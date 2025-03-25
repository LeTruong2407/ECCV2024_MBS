import os
import utils
import random
import logging
import argparse

import datetime
import time
import math

import numpy as np
from omegaconf import OmegaConf

from metrics import StreamSegMetrics

import torch
from torch.utils import data
import torch.nn.functional as F

from utils import ext_transforms as et
from utils.tasks import get_tasks

from datasets import VOCSegmentation
from datasets import ADESegmentation

from core import Segmenter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from utils import imutils
from utils.utils import AverageMeter

from segmentation_module import IncrementalSegmentationModule

# argment parser
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='./configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)), help="local_rank")
parser.add_argument('--log', default='test.log')
parser.add_argument('--backend', default='nccl')


#### NEST Change
# Thêm vào đầu file train.py của MBS (trước hàm warmup_nest)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torchvision.transforms import Resize, Compose
import PIL

def select(model_old, train_loader, old_classes, nb_new_classes, device):
    embedding_dim = model_old.encoder.d_model  # 768 với ViT-B
    new_classes_id = [x + old_classes for x in range(nb_new_classes)]
    bucket = torch.zeros(nb_new_classes, old_classes, embedding_dim, dtype=torch.float32).to(device)
    nums = torch.zeros(nb_new_classes, dtype=torch.long).to(device)

    model_old.eval()
    for cur_step, (images, labels, _) in enumerate(train_loader):  # MBS có thêm _ trong batch
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        resize = Compose([Resize((128, 128), interpolation=PIL.Image.NEAREST)])
        labels = resize(labels)
        labels = labels.reshape(-1)

        with torch.no_grad():
            outputs_old, features_old = model_old(images, ret_intermediate=True)
            outputs_old = F.interpolate(outputs_old, size=(128, 128), mode='bilinear', align_corners=False)
            outputs_old = torch.softmax(outputs_old, dim=1).permute(0, 2, 3, 1).reshape(-1, old_classes)

            pre_feature = features_old['pre_logits']  # Shape: (B, num_patches, d_model)
            pre_feature = F.interpolate(pre_feature, size=(128, 128), mode='bilinear', align_corners=False)
            pre_feature = pre_feature.permute(0, 2, 3, 1).reshape(-1, embedding_dim)

            # Trọng số cũ từ cls_emb của MaskTransformer
            imprinting_w = torch.cat([x.squeeze(0) for x in model_old.decoder.cls_emb[:-1]], dim=0)  # Shape: (old_classes, d_model)

            unique_elements = torch.unique(labels).tolist()
            intersection = list(set(unique_elements).intersection(new_classes_id))

            for new_class_id in intersection:
                new_class_mask = (labels == new_class_id)
                pre_feature1 = pre_feature.unsqueeze(1).repeat(1, imprinting_w.shape[0], 1)
                hadamard_product = pre_feature1 * imprinting_w
                hadamard_product = F.relu(hadamard_product)
                hadamard_product = torch.where(hadamard_product > 0, 1, 0)
                outputs_old1 = outputs_old.unsqueeze(-1).repeat(1, 1, hadamard_product.shape[-1])
                score = hadamard_product * outputs_old1
                cur_class_score = score[new_class_mask]
                if cur_class_score.shape[0] != 0:
                    bucket[new_class_id - old_classes] += torch.sum(cur_class_score, dim=0)
                    nums[new_class_id - old_classes] += cur_class_score.shape[0]
                del pre_feature1, outputs_old1, hadamard_product, new_class_mask, score, cur_class_score

    torch.distributed.all_reduce(bucket, op=distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(nums, op=distributed.ReduceOp.SUM)
    bucket = bucket / distributed.get_world_size()
    nums = nums / distributed.get_world_size()
    
    for i in range(nb_new_classes):
        if nums[i] > 0:
            bucket[i] /= nums[i]
    return bucket

# Trong file train.py của MBS (thay thế hàm warmup_nest hiện tại)
def warmup_nest(opts, model_prev, train_loader, device):
    """Giai đoạn warm-up của NeST để tính bucket từ pre_logits"""
    old_classes = sum(opts.num_classes[:-1])
    nb_new_classes = opts.num_classes[-1]
    
    # Tính bucket bằng hàm select
    with torch.no_grad():
        bucket = select(model_prev, train_loader, old_classes, nb_new_classes, device)
    return bucket


# calculate eta
def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

# logger function
def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

# train/val/test data prepare
## Thằng này lấy dataset gọi từ voc.yaml, các file yaml xong vào data_root và đọc file ảnh

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtResize(size=opts.dataset.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.dataset.crop_size, opts.dataset.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.train.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.dataset.crop_size),
            et.ExtCenterCrop(opts.dataset.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    if opts.dataset.name == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset.name == 'ade':
        dataset = ADESegmentation
    else:
        raise NotImplementedError
        
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    
    dataset_dict['val'] = dataset(opts=opts, image_set='val', transform=val_transform, cil_step=opts.curr_step)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    return dataset_dict
### Sau bước này là nó thu được dataset_dict là phần data mình vừa lấy ra nè

# validate function
def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            outputs, _, _, _ = model(images)
            
            if opts.train.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
                
        score = metrics.get_results()
        
    return score

# train function
def train(opts):
    writer = SummaryWriter('runs/'+ str(args.log))
    num_workers = 4 * len(opts.gpu_ids) ## opts nó đọc file 
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    # Get the target classes for the current task and step
    target_cls = get_tasks(opts.dataset.name, opts.task, opts.curr_step)
    
    # Calculate the number of classes for each step
    opts.num_classes = [len(get_tasks(opts.dataset.name, opts.task, step)) for step in range(opts.curr_step+1)]
    opts.num_classes = [1, opts.num_classes[0]-1] + opts.num_classes[1:]
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset.name, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset.name, opts.task, step)) for step in range(opts.curr_step+1))
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bg_label = 0
    
    if args.local_rank==0:
        print("==============================================")
        print(f"  task : {opts.task}")
        print(f"  step : {opts.curr_step}")
        print("  Device: %s" % device)
        print( "  opts : ")
        print(opts)
        print("==============================================")
# Khởi tạo model
    model = Segmenter(backbone=opts.train.backbone, num_classes=opts.num_classes, pretrained=True)
    
    if opts.curr_step > 0:
        model_prev = Segmenter(backbone=opts.train.backbone, num_classes=list(opts.num_classes)[:-1], pretrained=True)
    else:
        model_prev = None

    bucket = None
    if opts.curr_step > 0:
        dataset_dict = get_dataset(opts)
        train_loader_temp = data.DataLoader(
            dataset_dict['train'], 
            batch_size=opts.dataset.batch_size,
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        )
        bucket = warmup_nest(opts, model_prev, train_loader_temp, device)
    
    # Initialize the model with the specified backbone and number of classes
    model = Segmenter(backbone=opts.train.backbone, num_classes=opts.num_classes,
                pretrained=True)
    '''
    if opts.curr_step > 0:
        """ load previous model """
        model_prev = Segmenter(backbone=opts.train.backbone, num_classes=list(opts.num_classes)[:-1],
                pretrained=True)
    else:
        model_prev = None
    '''
    get_param = model.get_param_groups()
    
    if opts.curr_step > 0:
        param_group = [{"params": get_param[0], "lr": opts.optimizer.learning_rate*opts.optimizer.inc_lr}, # Encoder
                    {"params": get_param[1], "lr": opts.optimizer.learning_rate*opts.optimizer.inc_lr}, # Norm
                    {"params": get_param[2], "lr": opts.optimizer.learning_rate*opts.optimizer.inc_lr}] # Decoder
    else:
        param_group = [{"params": get_param[0], "lr": opts.optimizer.learning_rate}, # Encoder
                    {"params": get_param[1], "lr": opts.optimizer.learning_rate}, # Norm
                    {"params": get_param[2], "lr": opts.optimizer.learning_rate}] # Decoder
    
    # Initialize the optimizer with the parameter groups
    optimizer = torch.optim.SGD(params=param_group, 
                            lr=opts.optimizer.learning_rate,
                            weight_decay=opts.optimizer.weight_decay, 
                            momentum=0.9, 
                            nesterov=True)
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        
        if args.local_rank==0:
            print("Model saved as %s" % path)

    utils.mkdir('checkpoints')    
    # Restore
    best_score = -1
    cur_epochs = 0
    
    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
    
    # model load from checkpoint if opts_curr_step == 0 
    if opts.curr_step==0 and (opts.ckpt is not None and os.path.isfile(opts.ckpt)):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model.load_state_dict(checkpoint, strict=True)
        
        if args.local_rank==0:
                print("Curr_step is zero. Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
        ### Change
 
    # model load from checkpoint if opts_curr_step > 0
#    if opts.curr_step > 0:
     #   opts.ckpt = ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step-1)
    
    #    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
      #      checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
     #       model_prev.load_state_dict(checkpoint, strict=True)         
     #       
            # Transfer the background class token if weight transfer is enabled
     #       if opts.train.weight_transfer:
     #           curr_head_num = len(model.decoder.cls_emb) - 1
     #           class_token_param = model.state_dict()[f"decoder.cls_emb.{curr_head_num}"]
     #           for i in range(opts.num_classes[-1]):
      #              class_token_param[:, i] = checkpoint["decoder.cls_emb.0"]
                        
     #           checkpoint[f"decoder.cls_emb.{curr_head_num}"] = class_token_param
                    
     #       model.load_state_dict(checkpoint, strict=False)
                
   #         if args.local_rank==0:
 #               print("Model restored from %s" % opts.ckpt)
#            del checkpoint  # free memory
#        else:
#            if args.local_rank==0:
#                print("[!] Retrain")

        # Trong hàm train, sau khi khởi tạo model_prev

    if opts.curr_step > 0:
        opts.ckpt = ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step - 1)
    
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
            model_prev.load_state_dict(checkpoint, strict=True)
        
            if bucket is not None:
                imprinting_w = torch.cat([x for x in model_prev.decoder.cls_emb[:-1]], dim=1)
                imprinting_w = imprinting_w.squeeze(0)
                new_weight = torch.matmul(bucket.sum(dim=1).softmax(dim=1), bucket.mean(dim=1) * imprinting_w)
                curr_head_num = len(model.decoder.cls_emb) - 1
                model.decoder.cls_emb[curr_head_num].data = new_weight.unsqueeze(0)
                gamma = imprinting_w.norm(p=2).mean() / new_weight.norm(p=2).mean()
                model.decoder.cls_emb[curr_head_num].data *= gamma
        
            model.load_state_dict(checkpoint, strict=False)
            if args.local_rank == 0:
                print("Model restored from %s with NeST imprinting" % opts.ckpt)
            del checkpoint
        else:
            if args.local_rank == 0:
                print("[!] Retrain")

    if opts.curr_step > 0:
        model_prev.to(device)
        model_prev.eval()
    
        for param in model_prev.parameters():
            param.requires_grad = False
                
    if args.local_rank==0 and opts.curr_step>0:
        print("----------- trainable parameters --------------")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        print("-----------------------------------------------")
    
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[opts.gpu_ids[args.local_rank]], find_unused_parameters=True)
    model.train()
   
    dataset_dict = get_dataset(opts)
    train_sampler = DistributedSampler(dataset_dict['train'], shuffle=True)
    
    train_loader = data.DataLoader(
        dataset_dict['train'], 
        batch_size=opts.dataset.batch_size,
        sampler=train_sampler,  
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True, 
        prefetch_factor=4)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=opts.dataset.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.dataset.val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if args.local_rank==0:
        print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
        (opts.dataset.name, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    max_iters = opts.train.train_epochs * len(train_loader)
    val_interval = max(100, max_iters // 10)
    metrics = StreamSegMetrics(sum(opts.num_classes), dataset=opts.dataset.name)

    train_sampler.set_epoch(0)
            
    if args.local_rank==0:
        print(f"... train epoch : {opts.train.train_epochs} , iterations : {max_iters} , val_interval : {val_interval}")
    # Create a GradScaler for automatic mixed precision (AMP) training
    ## GradScaler dùng để loss, gradient trong sử dụng nhiều độ đo
    ## Amp là automatic mixing precision
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    # Set up the loss function based on the configuration
    if opts.train.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=opts.dataset.ignore_index, 
                                                           reduction='mean')
    elif opts.train.loss_type == 'ce_loss':                                                                                                                                                                                                                                                                       
        criterion = torch.nn.CrossEntropyLoss(ignore_index=opts.dataset.ignore_index, reduction='mean')
    
    # Set up additional loss functions for MBS if enabled
    if opts.train.MBS == True:
        # Separating Background-Class - output distillation, orthogonal loss
        ### Cái này define ra loss cho bước 3.5
        od_loss = utils.LabelGuidedOutputDistillation(reduction="mean", alpha=1.0).to(device)
        ortho_loss = utils.OtrthogonalLoss(reduction="mean", classes=target_cls).to(device)
    else:
        od_loss = utils.KnowledgeDistillationLoss(reduction="mean", alpha=1.0).to(device)
        ortho_loss = None
    # Adaptive Feature Distillation
    fd_loss = utils.AdaptiveFeatureDistillation(reduction="mean", alpha=1).to(device)

    criterion = criterion.to(device)
    cur_epochs = 0
    avg_loss = AverageMeter()
    
    for n_iter in range(max_iters):
        try:
            inputs, labels, _ = next(train_loader_iter)
        except:
            train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            inputs, labels, _ = next(train_loader_iter)
            cur_epochs += 1
        
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        origin_labels = labels.clone()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=opts.amp):
            ### Model được sử dụng từ decoder.py, áp dụng MaskTransformer
            ### output trả về là cái feature của img đầu vào, pathches chính là từng patch được chia nhỏ
            ### cls_seg_feat sẽ là trọng số của những class thông qua decoder
            ### cls_token là các class_embedding và ở đây nguyên lí sẽ là :
            ### Đầu tiên tạo ra class token = embedding, xong sau đó nối nó với pathches
            ## Và đưa qua MaskTransformer để học được thông tin các class-> thu được class_seg_feat
            outputs, patches, cls_seg_feat, cls_token = model(inputs)
            ### outputs chính là cái St dùng trong 3.3, 3.5
            lod = torch.zeros(1).to(device) 
            lfd_patches = torch.zeros(1).to(device)
            lfd = torch.zeros(1).to(device) ### lfd là ma trận số 0 có size là [1,] dùng để tính tổng loss trong step adaptive feature
            
            if opts.curr_step > 0:
                with torch.no_grad():
                    ### tạo ra pathces_prev để dùng cho Knowledge distillation 3.4, 3.3, 3.5
                    outputs_prev, patches_prev, cls_seg_feat_prev, _ = model_prev(inputs)
                    ## outputs_prev còn dùng trong 3.5 nữa, output của model cũ 
                    if opts.train.loss_type == 'bce_loss':
                        ## Pred_prob là prediction từ model cũ 3.3
                        pred_prob = torch.sigmoid(outputs_prev).detach()
                    else:
                        pred_prob = torch.softmax(outputs_prev, 1).detach()
                ## bg_label = 0, pred_label la gtri label
                ## Step này tạo ra pseudo_label 3.3
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                labels = torch.where((labels <= bg_label) & (pred_labels > bg_label) & (pred_scores >= opts.train.pseudo_thresh), 
                                        pred_labels, 
                                        labels)
                
                if opts.train.MBS:
                    ### Tạo ra cái object identifier 3.3
                    object_scores = torch.zeros(pred_prob.shape[0], 2, pred_prob.shape[2], pred_prob.shape[3]).to(device)
                    object_scores[:, 0] = pred_prob[:, 0]
                    object_scores[:, 1] = torch.sum(pred_prob[:, 1:], dim=1)
                    labels = torch.where((labels == 0) & (object_scores[:, 0] < object_scores[:, 1]), 
                                                opts.dataset.ignore_index, 
                                                labels)
                    # Bước này ở đây là đã có được labels chính là selective pseudo_label 3.3

                if opts.train.MBS:
                    ### Này là bước tính loss knowledge distillation 3.4
                    with torch.no_grad():
                        mask_origin = model_prev.get_masks()
                    HW = int(math.sqrt(patches.shape[1]))
                    ## bước downsample 3.4
                    label_temp = F.interpolate(labels.unsqueeze(1).float(), size=(HW, HW), mode='nearest').squeeze(1)
                    ## Tạo Reliability Map từ hàm make_scoremap ở file utils trong folder utils
                    pred_score_mask = utils.make_scoremap(mask_origin, label_temp, target_cls, bg_label, ignore_index=opts.dataset.ignore_index)
                    pred_scoremap = pred_score_mask.squeeze().reshape(-1, HW*HW)
                    ## lfd_patches tính loss dựa vào patches ở thời điểm hiện tại và patches ở thời điểm trước mô hình dự đoán
                    ## weights là dựa trên reliability map vừa tạo
                    lfd_patches = fd_loss(patches.unsqueeze(1), patches_prev.unsqueeze(1), weights=pred_scoremap.unsqueeze(-1).unsqueeze(1))
                else:
                    ## PP cũ ko dùng map này.
                    lfd_patches = fd_loss(patches, patches_prev, weights=1)
                ### Chỗ này cộng dồn loss L_afd
                lfd = lfd_patches + fd_loss(cls_seg_feat[:,:-len(target_cls)], cls_seg_feat_prev, weights=1)

                if opts.train.MBS:
                    ### Thực hiện tính loss ở phần 3.5
                    ## od_loss là cái LGKD dùng background_weight transfer
                    ## Theo pp thì này nó có thêm cái mask trong file loss.py sẽ là cái ground_truth ban đầu
                    ## thì ở đây nó dựa vào ground_truth xong nó sẽ lấy ra những vị trí là class mới ở output

                    lod = od_loss(outputs, outputs_prev, origin_labels) * opts.train.distill_args + ortho_loss(cls_token, weight=opts.num_classes[-1]/sum(opts.num_classes))
                else:
                    lod = od_loss(outputs, outputs_prev) * opts.train.distill_args      
            
            ## Này là loss ở 3.3 nè 
            ## Loss giữa output và label, label là thg SPL
            seg_loss = criterion(outputs, labels.type(torch.long))
            ## tính tổng tất cả các loss lại thôi
            loss_total = seg_loss + lfd + lod
        
        ## .scale chỉ đơn giản là scale giá trị loss lên vì nó nhỏ 
        scaler.scale(loss_total).backward()

        scaler.step(optimizer)
        ## avg_loss được gọi từ AverageMeter ở utils trong folder utils.
        ## avg_los gồm tính trung bình, tính tổng, số lượng bằng update
        ## avg_loss còn có phương thức reset về 0.
        avg_loss.update(loss_total.item()) 
        scaler.update()
        
        ## Log_iter để note lại sau 50 lần iters
        ### Local_rank là để set gpu nào chạy
        if (n_iter+1) % opts.train.log_iters == 0 and args.local_rank==0:
            delta, eta = cal_eta(time0, n_iter+1, max_iters)
            lr = optimizer.param_groups[0]['lr']
            logging.info("[Epochs: %d Iter: %d] Elasped: %s; ETA: %s; LR: %.3e; loss: %f; FD_loss: %f; OD_loss: %f"%(cur_epochs, n_iter+1, delta, eta, lr, avg_loss.avg, lfd.item(), 
                                                                                                                                 lod.item()))
            writer.add_scalar(f'loss/train_{opts.curr_step}', loss_total.item(), n_iter+1)
            writer.add_scalar(f'lr/train_{opts.curr_step}', lr, n_iter+1)
            record_inputs, record_outputs, record_labels = imutils.tensorboard_image(inputs=inputs, outputs=outputs, labels=labels, dataset=opts.dataset.name)
            
            writer.add_image(f"input/train_{opts.curr_step}", record_inputs, n_iter+1)
            writer.add_image(f"output/train_{opts.curr_step}", record_outputs, n_iter+1)
            writer.add_image(f"label/train_{opts.curr_step}", record_labels, n_iter+1)
        
        if (n_iter+1) % val_interval == 0:
            if args.local_rank==0:
                logging.info('Validating...')
            model.eval()
            val_score = validate(opts=opts, model=model, loader=val_loader, 
                              device=device, metrics=metrics)
            
            if args.local_rank==0:
                logging.info(metrics.to_str(val_score))
            model.train()
              
            writer.add_scalars(f'val/train_{opts.curr_step}', {"Overall Acc": val_score["Overall Acc"],
                                            "Mean Acc": val_score["Mean Acc"],
                                            "Mean IoU": val_score["Mean IoU"]}, n_iter+1)
            class_iou = list(val_score['Class IoU'].values())
            curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
            if args.local_rank==0:
                print("curr_val_score : %.4f" % (curr_score))
            
            if curr_score > best_score and args.local_rank==0:  # save best model
                print("... save best ckpt : ", curr_score)
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step))
        
    if args.local_rank==0:            
        print("... Training Done")
    time.sleep(2)
    
    if opts.curr_step >= 0:
        if args.local_rank==0:
            logging.info("... Testing Best Model")
        best_ckpt = ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step)
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))["model_state"]
        model.module.load_state_dict(checkpoint, strict=True)
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        if args.local_rank==0:
            logging.info(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(opts.dataset.name, opts.task, 0))
        
        if args.local_rank==0:
            logging.info(f"...from 1 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[1:first_cls]))
            logging.info(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
            logging.info(f"...from 1 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[1:first_cls]))
            logging.info(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))            
            

if __name__ == "__main__":
    
    args = parser.parse_args()
    opts = OmegaConf.load(args.config)
    random.seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed_all(opts.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    opts.dataset.batch_size = opts.dataset.batch_size // len(opts.gpu_ids)

    if args.local_rank == 0:
        setup_logger(filename=str(args.log)+'.log')
        logging.info('\nconfigs: %s' % opts)

    start_step = opts.curr_step
    total_step = len(get_tasks(opts.dataset.name, opts.task))

    torch.cuda.set_device(opts.gpu_ids[args.local_rank])
    dist.init_process_group(backend=args.backend,)
    for step in range(start_step, total_step):
        opts.curr_step = step
        train(opts=opts)
