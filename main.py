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

# argment parser
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='./configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--log', default='test.log')
parser.add_argument('--backend', default='nccl')

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

def get_base_model(model):
    """Helper to get the base model whether it's wrapped in DDP or not"""
    return model.module if hasattr(model, 'module') else model

def pre_tune_neST(opts, model, model_prev, train_loader, device, epochs=5):
    """NeST's pre-tuning phase to learn M_c and P_c"""
    base_model = get_base_model(model)
    base_model_prev = get_base_model(model_prev)

    base_model.train()
    base_model_prev.eval()

    # Freeze all parameters except NeST-specific ones
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model_prev.parameters():
        param.requires_grad = False

    # Initialize NeST parameters if not already done, move to device
    if not base_model.decoder.importance_matrices:
        base_model.decoder.init_nest_params(opts.num_classes, device=device)

    # NeST parameters to optimize
    nest_params = (
        list(base_model.decoder.importance_matrices) +
        list(base_model.decoder.projection_matrices) +
        [base_model.decoder.M0, base_model.decoder.P0]
    )
    for param in nest_params:
        param.requires_grad = True

    optimizer = torch.optim.SGD(nest_params, lr=opts.optimizer.learning_rate*0.1)

    # Get class indices (accounting for [1, 15, 1, ...] format)
    n_old = sum(opts.num_classes[:-1])  # Background + all old classes
    new_classes_start = n_old
    new_classes = list(range(new_classes_start, new_classes_start + opts.num_classes[-1]))

    # Unbiased cross-entropy loss
    def unbiased_cross_entropy(outputs, labels):
        valid_mask = labels != opts.dataset.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0).to(device)
        loss = F.cross_entropy(outputs, labels, ignore_index=opts.dataset.ignore_index, reduction='none')
        return loss[valid_mask].mean()

    # Set old_classifiers from previous model's cls_emb
    old_cls_emb = []
    for param in base_model_prev.decoder.cls_emb:
        num_classes_step = param.size(1)
        class_vectors = param.squeeze(0).split(1, dim=0)
        for vec in class_vectors:
            old_cls_emb.append(vec.t())  # [d_model, 1]
    base_model.decoder.old_classifiers = torch.cat(old_cls_emb, dim=1).to(device)  # [d_model, n_old]

    for epoch in range(epochs):
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get old model features and outputs
            with torch.no_grad():
                outputs_old, features_old = base_model_prev(images, ret_intermediate=True)
                features_old = features_old['pre_logits']  # [B, num_patches, d_model]

            # Generate classifier weights with gradients
            new_classifiers = base_model.decoder.generate_classifiers(opts.num_classes)

            # Forward pass with computed classifiers
            outputs, _, _, _ = base_model(images, classifiers=new_classifiers)  # masks, patches, cls_seg_feat, cls_token
            loss = unbiased_cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.local_rank == 0:
            print(f"Pre-tuning Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Finalize weights after pre-tuning (no gradients needed here)
    with torch.no_grad():
        new_classifiers = base_model.decoder.generate_classifiers(opts.num_classes)
        split_sizes = [1] + [opts.num_classes[1]] + [1] * (len(opts.num_classes) - 2)
        cls_emb_splits = new_classifiers.t().split(split_sizes, dim=0)
        for i, w in enumerate(cls_emb_splits):
            base_model.decoder.cls_emb[i].copy_(w.unsqueeze(0))

# train function
def train(opts):
    writer = SummaryWriter('runs/' + str(args.log))
    num_workers = 4 * len(opts.gpu_ids)
    
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
    
    if args.local_rank == 0:
        print("==============================================")
        print(f"  task : {opts.task}")
        print(f"  step : {opts.curr_step}")
        print("  Device: %s" % device)
        print("  opts : ")
        print(opts)
        print("==============================================")

    # Initialize the model
    model = Segmenter(backbone=opts.train.backbone, num_classes=opts.num_classes, pretrained=True)
    model = model.to(device)  # Move to device before pre-tuning

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
    
    if opts.curr_step > 0:
        """Load previous model"""
        model_prev = Segmenter(
            backbone=opts.train.backbone,
            num_classes=list(opts.num_classes)[:-1],  # [1, 10] for step 0
            pretrained=True
        )
        model_prev = model_prev.to(device)

        checkpoint_path = "checkpoints/vit_b_16_voc_10-1_step_0_overlap.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if args.local_rank == 0:
            print("Checkpoint keys:", list(checkpoint.keys()))
        model_prev.load_state_dict(checkpoint, strict=False)
        model_prev.eval()

        print(f"=> NeST pre-tuning for new classes at step {opts.curr_step}")
        pre_tune_neST(
            opts=opts,
            model=model,
            model_prev=model_prev,
            train_loader=train_loader,
            device=device,
            epochs=5
        )

        # Unfreeze parameters for main training after pre-tuning
        for param in model.parameters():
            param.requires_grad = True  # Unfreeze all for full training

    else:
        model_prev = None

    # DDP wrapping after pre-tuning
    model = DistributedDataParallel(model, device_ids=[opts.gpu_ids[args.local_rank]], find_unused_parameters=True)
    model.train()

    # Define param_groups after DDP wrapping
    nest_params = []
    if opts.curr_step > 0:
        nest_params = (
            list(model.module.decoder.importance_matrices) + 
            list(model.module.decoder.projection_matrices) +
            [model.module.decoder.M0, model.module.decoder.P0]
        )

    # Define param_groups (now includes all trainable parameters)
    param_groups = [
        {"params": model.module.encoder.parameters(), "lr": opts.optimizer.learning_rate},
        {"params": [p for p in model.module.decoder.parameters() 
                    if not any(id(p) == id(nest_p) for nest_p in nest_params)], 
         "lr": opts.optimizer.learning_rate}
    ]
    if nest_params:
        param_groups.append({
            "params": nest_params,
            "lr": opts.optimizer.learning_rate * 0.1  # Lower LR for NeST
        })

    optimizer = torch.optim.SGD(param_groups, 
                                momentum=0.9,
                                weight_decay=opts.optimizer.weight_decay)
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        
        if args.local_rank == 0:
            print("Model saved as %s" % path)

    utils.mkdir('checkpoints')    
    # Restore
    best_score = -1
    cur_epochs = 0
    
    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
    
    # Model load from checkpoint if opts_curr_step == 0 
    if opts.curr_step == 0 and (opts.ckpt is not None and os.path.isfile(opts.ckpt)):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model.module.load_state_dict(checkpoint, strict=True)
        
        if args.local_rank == 0:
            print("Curr_step is zero. Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    
# Model load from checkpoint if opts_curr_step > 0
    if opts.curr_step > 0:
        opts.ckpt = ckpt_str % (opts.train.backbone, opts.dataset.name, opts.task, opts.curr_step-1)
    
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
            if args.local_rank == 0:
                print("Checkpoint keys (model_state):", list(checkpoint.keys()))
            model_prev.load_state_dict(checkpoint, strict=False)         
            
            if opts.train.weight_transfer:
                curr_head_num = len(model.module.decoder.cls_emb) - 1
                class_token_param = model.module.state_dict()[f"decoder.cls_emb.{curr_head_num}"]
                if "decoder.cls_emb.0" in checkpoint:
                    for i in range(opts.num_classes[-1]):
                        class_token_param[:, i] = checkpoint["decoder.cls_emb.0"]
                elif "module.decoder.cls_emb.0" in checkpoint:  # Adjust for DDP prefix
                    for i in range(opts.num_classes[-1]):
                        class_token_param[:, i] = checkpoint["module.decoder.cls_emb.0"]
                elif "decoder.cls_emb" in checkpoint:
                    bg_token = checkpoint["decoder.cls_emb"][:, 0, :]
                    for i in range(opts.num_classes[-1]):
                        class_token_param[:, i] = bg_token.squeeze(0)
                else:
                    if args.local_rank == 0:
                        print("Warning: No suitable 'decoder.cls_emb' key found in checkpoint, skipping weight transfer")
                        
                checkpoint[f"decoder.cls_emb.{curr_head_num}"] = class_token_param
                    
            model.module.load_state_dict(checkpoint, strict=False)
                
            if args.local_rank == 0:
                print("Model restored from %s" % opts.ckpt)
            del checkpoint
        else:
            if args.local_rank == 0:
                print("[!] Retrain")

    if opts.curr_step > 0:
        model_prev.to(device)
        model_prev.eval()
        for param in model_prev.parameters():
            param.requires_grad = False
                
    if args.local_rank == 0:
        print("----------- trainable parameters --------------")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        print("-----------------------------------------------")
    # Ensure model is on device and in train mode after DDP (already done, but redundant calls removed)
    # model = model.to(device)  # Remove this redundant line
    # model = DistributedDataParallel(model, device_ids=[opts.gpu_ids[args.local_rank]], find_unused_parameters=True)  # Remove this redundant line
    # model.train()  # Remove this redundant line

    if args.local_rank == 0:
        print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
              (opts.dataset.name, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    max_iters = opts.train.train_epochs * len(train_loader)
    val_interval = max(100, max_iters // 10)
    metrics = StreamSegMetrics(sum(opts.num_classes), dataset=opts.dataset.name)

    train_sampler.set_epoch(0)
            
    if args.local_rank == 0:
        print(f"... train epoch : {opts.train.train_epochs} , iterations : {max_iters} , val_interval : {val_interval}")
    
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    if opts.train.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=opts.dataset.ignore_index, 
                                                           reduction='mean')
    elif opts.train.loss_type == 'ce_loss':                                                                                                                                                                                                                                                                       
        criterion = torch.nn.CrossEntropyLoss(ignore_index=opts.dataset.ignore_index, reduction='mean')

    if opts.train.MBS:
        fd_loss = utils.AdaptiveFeatureDistillation(reduction="mean", alpha=1).to(device)
    else:
        fd_loss = utils.KnowledgeDistillationLoss(reduction="mean", alpha=1.0).to(device)

    criterion = criterion.to(device)
    cur_epochs = 0
    avg_loss = AverageMeter()
    
    # Rest of your training loop (unchanged)
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
            outputs, patches, cls_seg_feat, cls_token = model(inputs)
            lod = torch.zeros(1).to(device)
            lfd_patches = torch.zeros(1).to(device)
            lfd = torch.zeros(1).to(device)
            
            if opts.curr_step > 0:
                with torch.no_grad():
                    outputs_prev, patches_prev, cls_seg_feat_prev, _ = model_prev(inputs)
                    pred_prob = torch.softmax(outputs_prev, 1).detach()

                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                labels = torch.where(
                    (labels <= bg_label) & (pred_labels > bg_label) & (pred_scores >= opts.train.pseudo_thresh),
                    pred_labels,
                    labels
                )

                HW = int(math.sqrt(patches.shape[1]))
                label_temp = F.interpolate(labels.unsqueeze(1).float(), size=(HW, HW), mode='nearest').squeeze(1)
                pred_score_mask = utils.make_scoremap(
                    model_prev.get_masks(), label_temp, target_cls, bg_label, ignore_index=opts.dataset.ignore_index
                )
                pred_scoremap = pred_score_mask.squeeze().reshape(-1, HW*HW)
                lfd = fd_loss(patches.unsqueeze(1), patches_prev.unsqueeze(1), weights=pred_scoremap.unsqueeze(-1).unsqueeze(1))

            seg_loss = criterion(outputs, labels.type(torch.long))
            loss_total = seg_loss + lfd

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        avg_loss.update(loss_total.item())
        scaler.update()
        
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
