from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import wandb
from torch.nn import init

from models.vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)
from models.weight_loaders import weight_loader_fn_dict
from models.x_transformers import AbsolutePositionalEmbedding, Attention


class GeneralMoE(nn.Module):
    def __init__(
            self,
            feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.layer = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))
        self.patch_agg = nn.Sequential(nn.Linear(257, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor):
        x = self.layer(x)
        f = self.patch_agg(x.permute(0, 1, 3, 2).contiguous())
        f = f.squeeze(-1)

        return f


class Gating(nn.Module):
    def __init__(
            self,
            num_experts: int = 8,
            feature_dim: int = 512*3,
    ):
        super().__init__()

        self.fc1 = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, num_experts))
        self.patch_agg = nn.Sequential(nn.Linear(257, 128), nn.ReLU(), nn.Linear(128, 1))
        self.fc2 = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x: torch.Tensor):
        N, T, L, C, d = x.size()
        x = x.view(N, T, L, C * d)

        x = self.fc1(x)
        x = x.permute(0, 1, 3, 2).contiguous()

        x = self.patch_agg(x)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        x = self.fc2(x)
        x = x.squeeze(-1)

        x = F.softmax(x, dim=-1)

        return x


class FirstLayerAdaptiveGating(nn.Module):
    def __init__(self,
                 feature_dim: int = 1024,
                 hidden_feature_dim: int = 512,
                 out_feature_dim: int = 512,
                 num_experts: int = 8,
                 num_next_experts: int = 4):
        super().__init__()

        self.fc_pre = nn.Sequential(nn.Linear(feature_dim, hidden_feature_dim), nn.ReLU())
        self.fc_post = nn.Linear(out_feature_dim, num_experts * num_next_experts)

        self.pos_embed = AbsolutePositionalEmbedding(dim=512, max_seq_len=257)
        self.attn = Attention(dim=512)

    def forward(self, xs: List[torch.Tensor]):
        num_samples = len(xs)
        n, t = xs[0].size(0), xs[0].size(1)
        ys = []
        for x in xs:
            y = rearrange(x, 'n t l c -> (n t) l c')
            y = self.fc_pre(y)
            y = y + self.pos_embed(y)
            ys.append(y)

        xs = torch.cat(ys, dim=1)
        xs = xs + self.attn(xs)
        xs = self.fc_post(xs)
        xs = rearrange(xs, '(n t) l c -> n t l c', n=n, t=t)
        xs = torch.chunk(xs, num_samples, dim=2)
        return xs


class SecondLayerAdaptiveGating(nn.Module):
    """
    Second layer gating mechanism for both single-sample and pariwise
    """
    def __init__(
            self,
            num_experts: int = 8,
            feature_dim: int = 512 * 3,
            num_frames: int = 16,
    ):
        super().__init__()

        self.fc1_pre = nn.Sequential(nn.Linear(feature_dim, 512), nn.LeakyReLU())
        self.fc1_post = nn.Linear(512, num_experts)
        self.patch_agg = nn.Sequential(nn.Linear(257, 128), nn.ReLU(), nn.Linear(128, 1))
        self.fc2 = nn.Sequential(nn.Linear(num_frames, 4), nn.LeakyReLU(), nn.Linear(4, 1))

        self.pos_embed = AbsolutePositionalEmbedding(dim=512, max_seq_len=257)
        self.attn = Attention(dim=512)

    def forward(self, xs: List[torch.Tensor]):
        """
        xs: List of 1 or 2 Tensors of shape (N, T, L, C, d)
        """
        num_samples = len(xs)
        n, t = xs[0].size(0), xs[0].size(1)

        ys = []
        for x in xs:
            y = rearrange(x, 'n t l c d -> (n t) l (c d)')
            y = self.fc1_pre(y)
            ys.append(y + self.pos_embed(y))

        xs = torch.cat(ys, dim=1)
        xs = xs + self.attn(xs)
        xs = self.fc1_post(xs)
        xs = rearrange(xs, '(n t) l cd -> n t cd l', n=n, t=t)

        xs = torch.chunk(xs, num_samples, dim=3)
        ys = []
        for x in xs:
            x = self.patch_agg(x)
            x = x.squeeze(-1).permute(0, 2, 1).contiguous()
            x = self.fc2(x)
            x = x.squeeze(-1)

            ys.append(F.softmax(x, dim=-1))

        return ys


class PoseMoE(nn.Module):
    def __init__(
            self,
            feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.layer = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))
        self.pose_agg = TransformerDecoderLayer(in_feature_dim=feature_dim, qkv_dim=768, num_heads=12, mlp_factor=4,
                                                mlp_dropout=0.5)
        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))

    def forward(self, x: torch.Tensor):
        N, T, L, C = x.size()
        cls_token = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)
        x = self.layer(x)
        f = self.pose_agg(cls_token, x.flatten(1, 2))

        return f[:, 0, :]


class FirstMoE(nn.Module):
    def __init__(
            self,
            num_experts: int = 8,
            num_next_experts: int = 4,
            feature_dim: int = 1024,
            hidden_feature_dim=512,
            out_feature_dim: int = 512,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_next_experts = num_next_experts
        self.feature_dim = feature_dim
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(feature_dim, hidden_feature_dim), nn.ReLU(),
                           nn.Linear(hidden_feature_dim, out_feature_dim)) for _ in range(num_experts)]
        )
        self.gating = FirstLayerAdaptiveGating(feature_dim=feature_dim, hidden_feature_dim=hidden_feature_dim,
                                               out_feature_dim=out_feature_dim, num_experts=num_experts,
                                               num_next_experts=num_next_experts)

    def forward(self, x: torch.Tensor):
        f = [self.experts[i](x) for i in range(self.num_experts)]
        concatenated_f = torch.cat([tensor.unsqueeze(4) for tensor in f], dim=4)

        w = self.gating([x])[0]
        B, T, C, N = w.size()
        w = w.view(B, T, C, self.num_experts, self.num_next_experts).contiguous()
        softmax_w = F.softmax(w, dim=3)
        softmax_w_expanded = softmax_w.unsqueeze(3).expand(-1, -1, -1, int(self.feature_dim / 2), -1, -1)

        result = torch.einsum('ijklnm,ijklmo->ijklno', concatenated_f.unsqueeze(-2), softmax_w_expanded)

        return softmax_w, result.squeeze(-2)
    
    def forward_two_inputs(self, x_1: torch.Tensor, x_2: torch.Tensor):
        f_1 = [self.experts[i](x_1) for i in range(self.num_experts)]
        concatenated_f_1 = torch.cat([tensor.unsqueeze(4) for tensor in f_1], dim=4)

        f_2 = [self.experts[i](x_2) for i in range(self.num_experts)]
        concatenated_f_2 = torch.cat([tensor.unsqueeze(4) for tensor in f_2], dim=4)

        w_1, w_2 = self.gating([x_1, x_2])
        B, T, C, N = w.size()

        w_1 = w_1.view(B, T, C, self.num_experts, self.num_next_experts).contiguous()
        w_2 = w_2.view(B, T, C, self.num_experts, self.num_next_experts).contiguous()

        softmax_w_1 = F.softmax(w_1, dim=3)
        softmax_w_2 = F.softmax(w_2, dim=3)

        softmax_w_expanded_1 = softmax_w_1.unsqueeze(3).expand(-1, -1, -1, int(self.feature_dim / 2), -1, -1)
        softmax_w_expanded_2 = softmax_w_2.unsqueeze(3).expand(-1, -1, -1, int(self.feature_dim / 2), -1, -1)

        result_1 = torch.einsum('ijklnm,ijklmo->ijklno', concatenated_f_1.unsqueeze(-2), softmax_w_expanded_1)
        result_2 = torch.einsum('ijklnm,ijklmo->ijklno', concatenated_f_2.unsqueeze(-2), softmax_w_expanded_2)

        return softmax_w_1, result_1.squeeze(-2), softamx_w_2, result_2.squeeze(-2)


class TemporalCrossAttention(nn.Module):

    def __init__(
            self,
            spatial_size: Tuple[int, int] = (14, 14),
            feature_dim: int = 768,
    ):
        super().__init__()

        self.spatial_size = spatial_size

        w_size = np.prod([x * 2 - 1 for x in spatial_size])
        self.w1 = nn.Parameter(torch.zeros([w_size, feature_dim]))
        self.w2 = nn.Parameter(torch.zeros([w_size, feature_dim]))

        idx_tensor = torch.zeros([np.prod(spatial_size) for _ in (0, 1)], dtype=torch.long)
        for q in range(np.prod(spatial_size)):
            qi, qj = q // spatial_size[1], q % spatial_size[1]
            for k in range(np.prod(spatial_size)):
                ki, kj = k // spatial_size[1], k % spatial_size[1]
                i_offs = qi - ki + spatial_size[0] - 1
                j_offs = qj - kj + spatial_size[1] - 1
                idx_tensor[q, k] = i_offs * (spatial_size[1] * 2 - 1) + j_offs
        self.idx_tensor = idx_tensor

    def forward_half(self, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        q, k = q[:, :, 1:], k[:, :, 1:]  # remove cls token

        assert q.size() == k.size()
        assert q.size(2) == np.prod(self.spatial_size)

        attn = torch.einsum('ntqhd,ntkhd->ntqkh', q / (q.size(-1) ** 0.5), k)
        attn = attn.softmax(dim=-2).mean(dim=-1)  # L, L, N, T

        self.idx_tensor = self.idx_tensor.to(w.device)
        w_unroll = w[self.idx_tensor]  # L, L, C
        ret = torch.einsum('ntqk,qkc->ntqc', attn, w_unroll)

        return ret

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        N, T, L, H, D = q.size()
        assert L == np.prod(self.spatial_size) + 1

        ret = torch.zeros([N, T, L, self.w1.size(-1)], device='cuda')
        ret[:, 1:, 1:, :] += self.forward_half(q[:, 1:, :, :, :], k[:, :-1, :, :, :], self.w1)
        ret[:, :-1, 1:, :] += self.forward_half(q[:, :-1, :, :, :], k[:, 1:, :, :, :], self.w2)

        return ret


class EVLDecoder(nn.Module):

    def __init__(
            self,
            num_frames: int = 8,
            spatial_size: Tuple[int, int] = (14, 14),
            num_layers: int = 4,
            in_feature_dim: int = 768,
            enable_temporal_conv: bool = True,
            enable_temporal_pos_embed: bool = True,
            enable_temporal_cross_attention: bool = True,
    ):
        super().__init__()

        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        # self.decoder_layers = nn.ModuleList(
        #    [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(num_layers)]
        # )
        self.decoder_layers = nn.Linear(in_feature_dim * num_layers, in_feature_dim)

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList(
                [nn.Conv1d(in_feature_dim, in_feature_dim, kernel_size=3, stride=1, padding=1, groups=in_feature_dim)
                 for _ in range(num_layers)]
            )
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(num_layers)]
            )
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList(
                [TemporalCrossAttention(spatial_size, in_feature_dim) for _ in range(num_layers)]
            )

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, in_features: List[Dict[str, torch.Tensor]]):
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        # xs = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        out_features = []
        for i in range(self.num_layers):
            frame_features = in_features[i]['out']

            if self.enable_temporal_conv:
                feat = in_features[i]['out']
                feat = feat.permute(0, 2, 3, 1).contiguous().flatten(0, 1)  # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C, T).permute(0, 3, 1, 2).contiguous()  # N, T, L, C
                frame_features += feat

            if self.enable_temporal_pos_embed:
                frame_features += self.temporal_pos_embed[i].view(1, T, 1, C)

            if self.enable_temporal_cross_attention:
                frame_features += self.cross_attention[i](in_features[i]['q'], in_features[i]['k'])

            # frame_features = frame_features.flatten(1, 2) # N, T * L, C
            # xs = self.decoder_layers[i](xs, frame_features)
            out_features.append(frame_features)
        out_features = torch.cat(out_features, dim=-1)
        out_features = self.decoder_layers(out_features)

        return out_features


class ResNet503D(nn.Module):

    def __init__(
            self,
            num_frames: int = 16,
            backbone_name: str = 'ViT-L/14-lnpre',
            backbone_type: str = 'clip',
            backbone_path: str = 'weights/ViT-L-14.pt',
            backbone_mode: str = 'freeze_fp16',
            decoder_num_layers: int = 4,
            enable_temporal_conv: bool = True,
            enable_temporal_pos_embed: bool = True,
            enable_temporal_cross_attention: bool = True,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers
        self.num_frames = num_frames

        backbone_config = self._create_backbone(backbone_name, backbone_type, backbone_path, backbone_mode)
        backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(
            x // y for x, y in zip(backbone_config['input_size'], backbone_config['patch_size']))

        self.decoder = EVLDecoder(
            num_frames=num_frames,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
        )

        #
        num_next_experts = 3
        num_experts = 8
        feature_dim = int(backbone_feature_dim / 2)
        self.moe_first = FirstMoE(num_experts=num_experts, num_next_experts=num_next_experts,
                                  feature_dim=backbone_feature_dim, hidden_feature_dim=feature_dim,
                                  out_feature_dim=feature_dim)
        self.moe_canonical = GeneralMoE(feature_dim=feature_dim)
        self.moe_appearance = GeneralMoE(feature_dim=feature_dim)
        self.moe_pose = PoseMoE(feature_dim=feature_dim)
        self.gating_second = SecondLayerAdaptiveGating(num_experts=num_next_experts, num_frames=num_frames)

        #
        self.fc = nn.Linear(feature_dim, 2048)
        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def _create_backbone(
            self,
            backbone_name: str,
            backbone_type: str,
            backbone_path: str,
            backbone_mode: str,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer2D(return_all_features=True, **vit_presets[backbone_name])
        backbone.load_state_dict(state_dict, strict=True)  # weight_loader_fn is expected to strip unused parameters

        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            self.backbone = [backbone]  # avoid backbone parameter registration

        return vit_presets[backbone_name]

    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            # freeze backbone
            self.backbone[0] = self.backbone[0].to(x.device)
            return self.backbone[0]
        else:
            # finetune backbone
            return self.backbone

    def intra_condition_consistency_loss(self, features, labels):
        # Calculate pairwise feature differences
        f_diff = features.unsqueeze(2) - features.unsqueeze(1)  # shape: [64, 8, 8, 512]
        f_diff = f_diff.pow(2).sum(-1)  # Sum over feature dimension, shape: [64, 8, 8]

        # Create mask for same subject (label-based) and different frames
        labels_expanded = labels.unsqueeze(1).expand(-1, self.num_frames)  # Expand labels for each frame, shape: [64, 8]
        mask_subject = labels_expanded.unsqueeze(2) == labels_expanded.unsqueeze(1)  # shape: [64, 8, 8]
        mask_frames = torch.eye(self.num_frames, device=features.device).bool().logical_not()  # Not the same frame, shape: [8, 8]
        mask = mask_subject & mask_frames.unsqueeze(0)  # Combine masks, shape: [64, 8, 8]

        # Apply mask and compute loss
        masked_diff = f_diff[mask]  # Apply mask to differences
        loss = masked_diff.mean()  # Mean over all valid differences
        return loss

    def inter_condition_consistency_loss(self, features, labels):

        features_mean = features.mean(dim=1)  # Average over frames to get one feature per condition per subject
        f_diff = features_mean.unsqueeze(1) - features_mean.unsqueeze(0)
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # Mask for same subject across conditions
        f_diff = f_diff.pow(2).sum(-1)  # Squared differences
        loss = (f_diff * mask).sum() / (mask.sum() + 1e-8)  # Apply mask and average
        return loss

    def gait_consistency_loss(self, features, labels):
        # Get unique labels (subjects) and create a dictionary to store features by subject
        unique_labels = labels.unique()
        subject_groups = {label.item(): [] for label in unique_labels}  # Convert label to integer for dict key

        # Group the features by label (subject)
        for i, label in enumerate(labels):
            subject_groups[label.item()].append(features[i])  # Convert label to integer for dict key

        # Calculate mean feature vector per subject across conditions
        mean_features = {}
        for label in unique_labels:
            subject_features = torch.stack(subject_groups[label.item()])  # Stack features for this subject
            mean_features[label.item()] = subject_features.mean(dim=0) # Mean across conditions

        # Calculate differences between features and the mean for each subject
        f_diff = []
        for label in unique_labels:
            subject_features = torch.stack(subject_groups[label.item()])  # Stack features for this subject
            subject_mean = subject_features.mean(dim=0, keepdim=True) # Mean across conditions
            subject_diff = (subject_features - subject_mean).pow(2).sum(-1)
            f_diff.append(subject_diff)

        # Stack all differences to compute the final loss
        f_diff = torch.cat(f_diff)  # Shape: [num_subjects, num_conditions]

        # Compute the loss: mean squared error within each subject group
        loss = f_diff.mean()  # Mean over all subjects and conditions
        return loss

    def denormalize(self, image, means=[0.48145466, 0.4578275, 0.40821073], stds=[0.26862954, 0.26130258, 0.27577711]):
        # Reverse the normalization for each channel
        for i in range(3):  # Assuming image is in (C, H, W) format
            image[i] = (image[i] * stds[i]) + means[i]
        return image

    def forward(self, x: torch.Tensor, pids: torch.Tensor, is_train: torch.bool, epoch: int, batch_idx: int):
        backbone = self._get_backbone(x)

        B, C, T, H, W = x.size()
        features = backbone(x.permute(0, 2, 1, 3, 4).flatten(0, 1))[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        f_features = self.decoder(features)

        # MoE first layer
        w_first, f_features = self.moe_first(f_features)
        canonical_features = self.moe_canonical(f_features[..., 0])
        appearance_features = self.moe_appearance(f_features[..., 1])
        gait_features = self.moe_pose(f_features[..., 2])

        #
        C_features = canonical_features.mean(1)
        A_features = appearance_features.mean(1)
        w = self.gating_second([f_features])[0]

        if is_train:
            loss_canonical_intra = self.intra_condition_consistency_loss(canonical_features, pids)
            loss_canonical_inter = self.inter_condition_consistency_loss(canonical_features, pids)
            loss_canonical = loss_canonical_intra + loss_canonical_inter
            loss_appearance = self.intra_condition_consistency_loss(appearance_features, pids)
            loss_gait = self.gait_consistency_loss(gait_features, pids)
            loss = loss_canonical + loss_appearance + loss_gait

        final_feature = w[:, [0]] * C_features + w[:, [1]] * A_features + w[:, [2]] * gait_features
        final_feature = self.bn(self.fc(final_feature))

        if is_train:
            return final_feature, A_features, loss
        else:
            return final_feature

    
    def forward_two_inputs(self, x1: torch.Tensor, x2: torch.Tensor, pids1: torch.Tensor, pids2: torch.Tensor, is_train: bool, epoch: int, batch_idx: int):
    # Shared backbone
    backbone = self._get_backbone(x1)
    B, C, T, H, W = x1.size()

    # Extract features from both inputs
    f1_raw = backbone(x1.permute(0, 2, 1, 3, 4).flatten(0, 1))[-self.decoder_num_layers:]
    f2_raw = backbone(x2.permute(0, 2, 1, 3, 4).flatten(0, 1))[-self.decoder_num_layers:]

    f1_raw = [
        dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in layer.items())
        for layer in f1_raw
    ]
    f2_raw = [
        dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in layer.items())
        for layer in f2_raw
    ]

    f1 = self.decoder(f1_raw)
    f2 = self.decoder(f2_raw)

    w1_first, f1, w2_first, f2 = self.moe_first.forward_two_inputs(f1, f2)

    # Apply second layer experts
    c1 = self.moe_canonical(f1[..., 0])
    a1 = self.moe_appearance(f1[..., 1])
    g1 = self.moe_pose(f1[..., 2])

    c2 = self.moe_canonical(f2[..., 0])
    a2 = self.moe_appearance(f2[..., 1])
    g2 = self.moe_pose(f2[..., 2])

    c1_mean, a1_mean = c1.mean(1), a1.mean(1)
    c2_mean, a2_mean = c2.mean(1), a2.mean(1)

    w1, w2 = self.gating_second.forward([f1, f2])

    f1_final = w1[:, [0]] * c1_mean + w1[:, [1]] * a1_mean + w1[:, [2]] * g1
    f2_final = w2[:, [0]] * c2_mean + w2[:, [1]] * a2_mean + w2[:, [2]] * g2

    f1_final = self.bn(self.fc(f1_final))
    f2_final = self.bn(self.fc(f2_final))

    if is_train:
        loss_c1 = self.intra_condition_consistency_loss(c1, pids1) + self.inter_condition_consistency_loss(c1, pids1)
        loss_c2 = self.intra_condition_consistency_loss(c2, pids2) + self.inter_condition_consistency_loss(c2, pids2)
        loss_a1 = self.intra_condition_consistency_loss(a1, pids1)
        loss_a2 = self.intra_condition_consistency_loss(a2, pids2)
        loss_g1 = self.gait_consistency_loss(g1, pids1)
        loss_g2 = self.gait_consistency_loss(g2, pids2)

        total_loss = loss_c1 + loss_c2 + loss_a1 + loss_a2 + loss_g1 + loss_g2

        return f1_final, f2_final, total_loss

    else:
        return f1_final, f2_final


def C2DResNet50(config, **kwargs):
    return ResNet503D()
