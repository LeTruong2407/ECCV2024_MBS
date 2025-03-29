import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from .blocks import Block, FeedForward
from .utils import init_weights


class MaskLayerNorm(nn.Module):
    def __init__(self, num_class, eps=1e-5, elementwise_affine=True):
        super(MaskLayerNorm, self).__init__()
        self.num_class = num_class
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.ParameterList(nn.Parameter(torch.ones(c)) for c in self.num_class)
            self.bias = nn.ParameterList(nn.Parameter(torch.zeros(c)) for c in self.num_class)

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        normed_input = (input - mean) / (std + self.eps)
        
        weight = [weight for weight in self.weight]
        bias = [bias for bias in self.bias]
        
        weight = torch.cat(weight)
        bias = torch.cat(bias)
        
        if self.elementwise_affine:
            normed_input = normed_input * weight + bias
        
        return normed_input
    
class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x

class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.ParameterList(nn.Parameter(torch.randn(1, c, d_model)) for c in self.n_cls)
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.sum_cls = sum(n_cls)
        self.mask_norm = MaskLayerNorm(self.n_cls)
        
        self.patches = None
        self.cls_seg_feat = None
        self.cls_token = None
        
        self.apply(init_weights)
        for param in self.cls_emb:
            trunc_normal_(param, std=.02)
        
        # Add NeST components
        self.importance_matrices = nn.ParameterList()  # For M_c matrices
        self.projection_matrices = nn.ParameterList()  # For P_c matrices
        self.old_classifiers = None  # Will store W_old
        
        # Background specific parameters
        self.M0 = nn.Parameter(torch.randn(d_model, 1))  # For background
        self.P0 = nn.Parameter(torch.randn(1, 1))        # For background

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def return_feats(self):
        return self.patches, self.cls_seg_feat, self.cls_token

    def init_nest_params(self, num_classes_list, device=None):
        """Initialize NeST parameters based on num_classes list"""
        if len(num_classes_list) <= 2:  # Base step, no new classes yet
            return
        n_old = sum(num_classes_list[:-1])  # All previous classes including background
        n_new = num_classes_list[-1]  # Number of new classes in current step
        for _ in range(n_new):
            self.importance_matrices.append(nn.Parameter(torch.randn(self.d_model, n_old)))
            self.projection_matrices.append(nn.Parameter(torch.randn(n_old, 1)))
        # Initialize with small values
        for param in self.importance_matrices:
            nn.init.normal_(param, mean=0.0, std=0.02)
            if device is not None:
                param.data = param.data.to(device)
        for param in self.projection_matrices:
            nn.init.normal_(param, mean=0.0, std=0.02)
            if device is not None:
                param.data = param.data.to(device)
        if device is not None:
            self.M0.data = self.M0.data.to(device)
            self.P0.data = self.P0.data.to(device)

    def compute_importance_matrix(self, features_old, outputs_old, labels, new_class_idx):
        B, D, H, W = features_old.shape
        n_old = outputs_old.shape[1]

        p_u = features_old.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
        s_u = F.softmax(outputs_old, dim=1).permute(0, 2, 3, 1).reshape(-1, n_old)  # (B*H*W, n_old)

        W_old = self.old_classifiers  # (D, n_old)
        if W_old is None or W_old.shape[1] != n_old:
            raise ValueError("old_classifiers not properly initialized")

        H_u = p_u.unsqueeze(1) * W_old.unsqueeze(0)  # (B*H*W, n_old, D)
        H_u_mask = (H_u > 0).float()

        new_class_mask = (labels.reshape(-1) == new_class_idx)
        if new_class_mask.sum() == 0:
            return torch.zeros(D, n_old).to(features_old.device), torch.zeros(n_old, 1).to(features_old.device)

        M_c = (H_u_mask * s_u.unsqueeze(-1))[new_class_mask].mean(dim=0).t()  # (D, n_old)
        P_c = F.softmax(M_c.sum(dim=0), dim=0).unsqueeze(1)  # (n_old, 1)

        return M_c, P_c

    def generate_classifiers(self, num_classes_list):
        if self.old_classifiers is None:
            raise ValueError("old_classifiers must be set before generating new classifiers")

        new_classifiers = []
        # Background
        w0 = (torch.sigmoid(self.M0) * self.old_classifiers[:, 0:1]) @ torch.softmax(self.P0, dim=0)
        new_classifiers.append(w0)

        # Old classes (excluding background)
        n_old = sum(num_classes_list[:-1])
        new_classifiers.append(self.old_classifiers[:, 1:n_old])

        # New classes
        n_new = num_classes_list[-1]
        for i in range(n_new):
            M = torch.sigmoid(self.importance_matrices[i])
            P = torch.softmax(self.projection_matrices[i], dim=0)
            w_c = (M * self.old_classifiers) @ P
            new_classifiers.append(w_c)

        return torch.cat(new_classifiers, dim=1)

    def forward(self, x, im_size, classifiers=None):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)  # [batch_size, num_patches, d_model]
        B = x.size(0)  # Batch size

        if classifiers is None:
            cls_embeds = [cls_embed for cls_embed in self.cls_emb]
            cls_embeds = torch.cat(cls_embeds, dim=1)  # [1, n_cls, d_model]
        else:
            cls_embeds = classifiers.t().unsqueeze(0)  # [d_model, n_cls] -> [n_cls, d_model] -> [1, n_cls, d_model]
        
        cls_embeds = cls_embeds.expand(B, -1, -1)  # [batch_size, n_cls, d_model]
        self.cls_token = cls_embeds
        x = torch.cat((x, cls_embeds), 1)  # [batch_size, num_patches + n_cls, d_model]
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.decoder_norm(x)
        
        patches, cls_seg_feat = x[:, :-self.sum_cls], x[:, -self.sum_cls:]  # [batch_size, num_patches, d_model], [batch_size, n_cls, d_model]
        
        patches = patches @ self.proj_patch  # [batch_size, num_patches, d_model]
        cls_seg_feat = cls_seg_feat @ self.proj_classes  # [batch_size, n_cls, d_model]
        
        self.patches = patches
        self.cls_seg_feat = cls_seg_feat
        
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        
        masks = patches @ cls_seg_feat.transpose(1, 2)  # [batch_size, num_patches, n_cls]
        
        masks = self.mask_norm(masks)
    
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))  # [batch_size, n_cls, H, W]
        
        return masks
