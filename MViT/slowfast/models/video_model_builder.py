# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""
import random
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import slowfast.utils.logging as logging
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
    prod,
)
from slowfast.utils import misc

from . import head_helper, head_helper_av, operators, resnet_helper, resnet_helper_av, stem_helper, stem_helper_av  # noqa
from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

logger = logging.get_logger(__name__)

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "r3d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
    "avslowfast": [
        [[1], [5], [1]],  # conv1 temp kernel for slow, fast and audio pathway.
        [[1], [3], [1]],  # res2 temp kernel for slow, fast and audio pathway.
        [[1], [3], [1]],  # res3 temp kernel for slow, fast and audio pathway.
        [[3], [3], [1]],  # res4 temp kernel for slow, fast and audio pathway.
        [[3], [3], [1]],  # res5 temp kernel for slow, fast and audio pathway.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "r3d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
    "avslowfast": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
}


class AVS(nn.Module):
    """
    Compute Audio-Visual synchronization loss.
    """
    
    def __init__(self, ref_dim, query_dim, proj_dim, num_gpus, num_shards):
        """
        Args:
            ref_dim (int): the channel dimension of the reference data point
                (usually a visual input).
            query_dim (int): the channel dimension of the query data point
                (usually an audio input).
            proj_dim (int): the channel dimension of the projected codes.
            num_gpus (int): number of gpus used.
            num_shards (int): number of shards used.
        """

        super(AVS, self).__init__()
        
        # initialize fc projection layers
        self.proj_dim = proj_dim
        self.ref_fc = nn.Linear(ref_dim, proj_dim, bias=True)
        self.query_fc = nn.Linear(query_dim, proj_dim, bias=True)
        self.num_gpus = num_gpus
        self.num_shards = num_shards
    
    
    def contrastive_loss(self, ref, pos, neg, audio_mask, margin):
        """
        Implement the contrastive loss used in https://arxiv.org/abs/1807.00230
        """
        N = torch.sum(audio_mask)
        
        pos_dist = ref - pos
        neg_dist = ref - neg
        pos_dist = pos_dist[audio_mask]
        neg_dist = neg_dist[audio_mask]
        
        pos_loss = torch.norm(pos_dist)**2
        neg_dist = torch.norm(neg_dist, dim=1)
        neg_loss = torch.sum(torch.clamp(margin - neg_dist, min=0)**2)
        loss = (pos_loss + neg_loss) / (2*N + 1e-8)
        return loss
        
        
    def forward(self, ref, pos, neg, audio_mask, norm='L2', margin=0.99):
        # reduce T, H, W dims
        ref = torch.mean(ref, (2, 3, 4))
        pos = torch.mean(pos, (2, 3, 4))
        neg = torch.mean(neg, (2, 3, 4))
        
        # projection
        ref = self.ref_fc(ref)
        pos = self.query_fc(pos)
        neg = self.query_fc(neg)
        
        # normalize
        if norm == 'L2':
            ref = torch.nn.functional.normalize(ref, p=2, dim=1)
            pos = torch.nn.functional.normalize(pos, p=2, dim=1)
            neg = torch.nn.functional.normalize(neg, p=2, dim=1)
            # scale data so that ||x-y||^2 fall in [0, 1]
            ref = ref * 0.5
            pos = pos * 0.5
            neg = neg * 0.5
        elif norm == 'Tanh':
            scale = 1.0 / self.proj_dim
            ref = torch.nn.functional.tanh(ref) * scale
            pos = torch.nn.functional.tanh(pos) * scale
            neg = torch.nn.functional.tanh(neg) * scale
        
        # contrstive loss
        loss = self.contrastive_loss(ref, pos, neg, audio_mask, margin)
        
        # scale the loss with nGPUs and shards
        # loss = loss / float(self.num_gpus * self.num_shards)
        loss = loss / float(self.num_shards)
        
        return loss


class FuseAV(nn.Module):
    """
    Fuses information from audio to visual pathways.
    """
    
    def __init__(
        self,
        # slow pathway
        dim_in_s,
        # fast pathway
        dim_in_f,
        fusion_conv_channel_ratio_f,
        fusion_kernel_f,
        alpha_f,
        # audio pathway
        dim_in_a,
        fusion_conv_channel_mode_a,
        fusion_conv_channel_dim_a,
        fusion_conv_channel_ratio_a,
        fusion_kernel_a,
        alpha_a,
        conv_num_a,
        # fusion connections
        use_fs_fusion,
        use_afs_fusion,
        # AVS
        use_avs,
        avs_proj_dim,
        # general params
        num_gpus=1,
        num_shards=1,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
    ):
        """
        Perform A2TS fusion described in AVSlowFast paper.

        Args:
            dim_in_s (int): channel dimension of the slow pathway.
            dim_in_f (int): channel dimension of the fast pathway.
            fusion_conv_channel_ratio_f (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel_f (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha_f (int): the frame rate ratio between the Fast and Slow pathway.
            dim_in_a (int): channel dimension of audio inputs.
            fusion_conv_channel_mode_a (str): 'ByDim' or 'ByRatio'. Decide how to
                compute intermediate feature dimension for Audiovisual fusion.
            fusion_conv_channel_dim_a (int): used when 'fusion_conv_channel_mode_a'
                == 'ByDim', decide intermediate feature dimension for Audiovisual fusion.
            fusion_conv_channel_ratio_a (float): used when 'fusion_conv_channel_mode_a'
                == 'ByRatio', decide intermediate feature dimension for Audiovisual fusion.
            fusion_kernel_a (int): kernel size of the convolution used to fuse
                from Audio pathway to SlowFast pathways.
            alpha_a (int): the frame rate ratio between the Audio and Slow pathway.
            conv_num_a (int): number of convs applied on audio, before fusing into
                SlowFast pathways.
            use_fs_fusion (bool): whether use Fast->Slow fusion.
            use_afs_fusion (bool): whether use Audio->SlowFast fusion.
            use_avs (bool): whether compute audiovisual synchronization loss.
            avs_proj_dim (int): channel dimension of the projection codes for audiovisual
                synchronization loss.
            num_gpus (int): number of gpus used.
            num_shards (int): number of shards used.            
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
        """
        super(FuseAV, self).__init__()
        self.conv_num_a = conv_num_a
        self.use_fs_fusion = use_fs_fusion
        self.use_afs_fusion = use_afs_fusion
                
        # perform F->S fusion
        if use_fs_fusion:
            self.conv_f2s = nn.Conv3d(
                dim_in_f,
                dim_in_f * fusion_conv_channel_ratio_f,
                kernel_size=[fusion_kernel_f, 1, 1],
                stride=[alpha_f, 1, 1],
                padding=[fusion_kernel_f // 2, 0, 0],
                bias=False,
            )
            self.bn_f2s = nn.BatchNorm3d(
                dim_in_f * fusion_conv_channel_ratio_f, eps=eps, momentum=bn_mmt
            )
            self.relu_f2s = nn.ReLU(inplace_relu)
        
        # perform A->FS fusion
        if fusion_conv_channel_mode_a == 'ByDim':
            afs_fusion_interm_dim = int(fusion_conv_channel_dim_a)
        elif fusion_conv_channel_mode_a == 'ByRatio':
            afs_fusion_interm_dim = int(dim_in_a * fusion_conv_channel_ratio_a)
        else:
            raise RuntimeError
        if use_afs_fusion:
            cur_dim_in = dim_in_a
            for idx in range(conv_num_a):
                if idx == conv_num_a - 1:
                    cur_stride = alpha_a
                    cur_dim_out = int(dim_in_f * fusion_conv_channel_ratio_f \
                                      + dim_in_s)
                else:
                    cur_stride = 1
                    cur_dim_out = afs_fusion_interm_dim
                conv_a2fs = nn.Conv3d(
                    cur_dim_in,
                    cur_dim_out,
                    kernel_size=[1, fusion_kernel_a, 1],
                    stride=[1, cur_stride, 1],
                    padding=[0, fusion_kernel_a // 2, 0],
                    bias=False,
                )
                bn_a2fs = nn.BatchNorm3d(
                    cur_dim_out, eps=eps, momentum=bn_mmt
                )
                relu_a2fs = nn.ReLU(inplace_relu)
                self.add_module('conv_a2fs_%d' % idx, conv_a2fs)
                self.add_module('bn_a2fs_%d' % idx, bn_a2fs)
                self.add_module('relu_a2fs_%d' % idx, relu_a2fs)
                cur_dim_in = cur_dim_out
        
        dim_in_a = int(dim_in_f * fusion_conv_channel_ratio_f + dim_in_s)
        
        # optionally compute audiovisual synchronization loss
        if use_avs:
            self.avs = AVS(
                dim_in_f * fusion_conv_channel_ratio_f + dim_in_s, 
                dim_in_a, 
                avs_proj_dim,
                num_gpus,
                num_shards,
            )
            
            
    def forward(self, x, get_misaligned_audio=False, mode='AFS'):
        """
        Forward function for audiovisual fusion, note that it currently only 
        supports A->FS fusion mode (which is the default used in AVSlowFast paper)
        Args:
            x (list): contains slow, fast and audio features
            get_misaligned_audio (bool): whether misaligned audio is carried in x
            mode (str):
                AFS  -- fuse audio, fast and slow
                AS   -- fuse audio and slow 
                FS   -- fuse fast and slow 
                NONE -- do not fuse at all
        """
        x_s = x[0]
        x_f = x[1]
        x_a = x[2]
        fuse = x_s
        cache = {}
        
        if mode != 'NONE':
            fs_proc, afs_proc = None, None
            
            # F->S
            if self.use_fs_fusion:
                fs_proc = self.conv_f2s(x_f)
                fs_proc = self.bn_f2s(fs_proc)
                fs_proc = self.relu_f2s(fs_proc)
                fuse = torch.cat([fuse, fs_proc], 1)
                cache['fs'] = fuse
                    
            # A->FS
            if self.use_afs_fusion:
                # [N C 1 T F] -> [N C 1 T 1]
                afs_proc = torch.mean(x_a, dim=-1, keepdim=True)
                for idx in range(self.conv_num_a):
                    conv = getattr(self, 'conv_a2fs_%d' % idx)
                    bn = getattr(self, 'bn_a2fs_%d' % idx)
                    relu = getattr(self, 'relu_a2fs_%d' % idx)
                    afs_proc = conv(afs_proc)
                    afs_proc = bn(afs_proc)
                    afs_proc = relu(afs_proc)
                if get_misaligned_audio:
                    afs_proc_pos, afs_proc_neg = torch.chunk(afs_proc, 2, dim=0)
                    cache['a_pos'] = afs_proc_pos
                    cache['a_neg'] = afs_proc_neg
                else:
                    afs_proc_pos = afs_proc
                # [N C 1 T 1] -> [N C T 1 1]
                afs_proc_pos = afs_proc_pos.permute(0, 1, 3, 2, 4) 
                if 'A' in mode:
                    fuse = afs_proc_pos + fuse
                else:
                    fuse = afs_proc_pos * 0.0 + fuse
        return [fuse, x_f, x_a], cache

class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]

@MODEL_REGISTRY.register()
class AVSlowFast(nn.Module):
    """
    Model builder for AVSlowFast network.
    Fanyi Xiao, Yong Jae Lee, Kristen Grauman, Jitendra Malik, Christoph Feichtenhofer.
    "Audiovisual Slowfast Networks for Video Recognition."
    https://arxiv.org/abs/2001.08740
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(AVSlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 3
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )


    def _construct_network(self, cfg):
        """
        Builds an AVSlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway, and the third one is the Audio 
            pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        
        self.DROPPATHWAY_RATE = cfg.SLOWFAST.DROPPATHWAY_RATE
        self.FS_FUSION = cfg.SLOWFAST.FS_FUSION
        self.AFS_FUSION = cfg.SLOWFAST.AFS_FUSION
        self.GET_MISALIGNED_AUDIO = cfg.DATA.GET_MISALIGNED_AUDIO
        self.AVS_FLAG = cfg.SLOWFAST.AVS_FLAG
        self.AVS_PROJ_DIM = cfg.SLOWFAST.AVS_PROJ_DIM
        self.AVS_VAR_THRESH = cfg.SLOWFAST.AVS_VAR_THRESH
        self.AVS_DUPLICATE_THRESH = cfg.SLOWFAST.AVS_DUPLICATE_THRESH
        
        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        tf_trans_func = [cfg.RESNET.TRANS_FUNC] * 2 + \
                        [cfg.RESNET.AUDIO_TRANS_FUNC]
        trans_func = [tf_trans_func] * cfg.RESNET.AUDIO_TRANS_NUM + \
            [cfg.RESNET.TRANS_FUNC] * (4 - cfg.RESNET.AUDIO_TRANS_NUM)

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        
        if cfg.SLOWFAST.AU_REDUCE_TF_DIM:
            tf_stride = 2
        else:
            tf_stride = 1
        tf_dim_reduction = 1

        self.s1 = stem_helper_av.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[
                width_per_group, 
                width_per_group // cfg.SLOWFAST.BETA_INV, 
                width_per_group // cfg.SLOWFAST.AU_BETA_INV
            ],
            kernel=[
                temp_kernel[0][0] + [7, 7], 
                temp_kernel[0][1] + [7, 7], 
                [temp_kernel[0][2] + [9, 1], temp_kernel[0][2] + [1, 9]],
            ],
            stride=[[1, 2, 2], [1, 2, 2], [[1, 1, 1], [1, 1, 1]]],
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
                [[temp_kernel[0][2][0] // 2, 4, 0], [temp_kernel[0][2][0] // 2, 0, 4]],
            ],
            stride_pool=[True, True, False],
        )
        
        if self.FS_FUSION[0] or self.AFS_FUSION[0]:
            self.s1_fuse = FuseAV(
                # Slow
                width_per_group,
                # Fast
                width_per_group // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[0],
                self.AFS_FUSION[0],
                # AVS
                self.AVS_FLAG[0],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )
        
        slow_dim = width_per_group + \
            (width_per_group // out_dim_ratio if self.FS_FUSION[0] else 0)
        self.s2 = resnet_helper_av.ResStage(
            dim_in=[
                slow_dim,
                width_per_group // cfg.SLOWFAST.BETA_INV,
                width_per_group // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 4 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner, 
                dim_inner // cfg.SLOWFAST.BETA_INV, 
                dim_inner // cfg.SLOWFAST.AU_BETA_INV
            ],
            temp_kernel_sizes=temp_kernel[1],
            stride=[1] * 3,
            num_blocks=[d2] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[0],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        if self.FS_FUSION[1] or self.AFS_FUSION[1]:
            self.s2_fuse = FuseAV(
                # Slow
                width_per_group * 4,
                # Fast
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 4 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[1],
                self.AFS_FUSION[1],
                # AVS
                self.AVS_FLAG[1],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)
            
        slow_dim = width_per_group * 4 + \
            (width_per_group * 4 // out_dim_ratio if self.FS_FUSION[1] else 0)
        self.s3 = resnet_helper_av.ResStage(
            dim_in=[
                slow_dim,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 4 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 8 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner * 2, 
                dim_inner * 2 // cfg.SLOWFAST.BETA_INV,
                dim_inner * 2 // cfg.SLOWFAST.AU_BETA_INV
            ],
            temp_kernel_sizes=temp_kernel[2],
            stride=[2, 2, tf_stride],
            num_blocks=[d3] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[1],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        tf_dim_reduction *= tf_stride
        
        if self.FS_FUSION[2] or self.AFS_FUSION[2]:
            self.s3_fuse = FuseAV(
                # Slow
                width_per_group * 8,
                # Fast
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 8 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[2],
                self.AFS_FUSION[2],
                # AVS
                self.AVS_FLAG[2],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )

        slow_dim = width_per_group * 8 + \
            (width_per_group * 8 // out_dim_ratio if self.FS_FUSION[2] else 0)
        self.s4 = resnet_helper_av.ResStage(
            dim_in=[
                slow_dim,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 8 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 16 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner * 4, 
                dim_inner * 4 // cfg.SLOWFAST.BETA_INV,
                dim_inner * 4 // cfg.SLOWFAST.AU_BETA_INV
            ],
            temp_kernel_sizes=temp_kernel[3],
            stride=[2, 2, tf_stride],
            num_blocks=[d4] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[2],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        tf_dim_reduction *= tf_stride
        
        if self.FS_FUSION[3] or self.AFS_FUSION[3]:
            self.s4_fuse = FuseAV(
                # Slow
                width_per_group * 16,
                # Fast
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 16 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[3],
                self.AFS_FUSION[3],
                # AVS
                self.AVS_FLAG[3],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )
        
        slow_dim = width_per_group * 16 + \
            (width_per_group * 16 // out_dim_ratio if self.FS_FUSION[3] else 0)
        self.s5 = resnet_helper_av.ResStage(
            dim_in=[
                slow_dim,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 16 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 32 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner * 8, 
                dim_inner * 8 // cfg.SLOWFAST.BETA_INV,
                dim_inner * 8 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            temp_kernel_sizes=temp_kernel[4],
            stride=[2, 2, tf_stride],
            num_blocks=[d5] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[3],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        tf_dim_reduction *= tf_stride
        
        # setup AVS for pool5 output
        if self.AVS_FLAG[4]:
            # this FuseAV object is used for compute AVS loss only
            self.s5_fuse = FuseAV(
                # Slow
                width_per_group * 32,
                # Fast
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 32 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                True,
                True,
                # AVS
                self.AVS_FLAG[4],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )

        self.head = head_helper_av.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 32 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
                [
                    1,
                    cfg.DATA.AUDIO_FRAME_NUM // tf_dim_reduction,
                    cfg.DATA.AUDIO_MEL_NUM // tf_dim_reduction,
                ],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )
    
    
    def freeze_bn(self, freeze_bn_affine):
        """
        Freeze the BN parameters
        """
        print("Freezing Mean/Var of BatchNorm.")
        if freeze_bn_affine:
            print("Freezing Weight/Bias of BatchNorm.")
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm1d) or \
                isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.BatchNorm3d):
                # if 'pathway2' in name or 'a2fs' in name:
                #     continue
                m.eval()
                if freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    
    
    def gen_fusion_avs_pattern(self):
        """
        This function generates a fusion pattern and a avs loss compute pattern.
        Specifically, fusion pattern is determined by both pre-defined fusion 
        connections between Slow/Fast/Audio, and the flag of whether to drop the 
        audio pathway, which is generated on the fly. 
        For AVS pattern, it is determined by fusion pattern. For example, if we 
        decided to have AFS fusion pattern like [False, False, True, True], 
        which means to have fusion between audio and visual after res3 and res4,
        and let's say our AFS_FUSION is [False, False, False, True], then we will 
        not compute AVS loss anywhere. This is because since we have fused audio
        into visual at res3, any visual features after this has already "seen" 
        audio features and the problem of telling whether audio and visual is in-sync
        will be trivial.
        """
        is_drop = self.training and random.random() < self.DROPPATHWAY_RATE
        fs_fusion = self.FS_FUSION
        afs_fusion = self.AFS_FUSION
        runtime_afs_fusion = []
        fusion_pattern, avs_pattern = [], []
        
        for idx in range(4):
            # If a junction has both audiovisual fusion and slowfast fusion,
            # we call it 'AFS'. If it only has slowfast fusion, we call it 'FS'.
            # If it only has audio and slow fusion, we call it 'AS'
            cur_fs_fusion = fs_fusion[idx]
            cur_afs_fusion = afs_fusion[idx] and not is_drop
            if cur_fs_fusion and cur_afs_fusion:
                fusion_pattern.append('AFS')
            elif cur_fs_fusion and not cur_afs_fusion:
                fusion_pattern.append('FS')
            elif not cur_fs_fusion and cur_afs_fusion:
                fusion_pattern.append('AS')
            else:
                fusion_pattern.append('NONE')
            runtime_afs_fusion.append(cur_afs_fusion)
        
        # compute the earliest audiovisual fusion, so that we don't do AVS
        # for any stage later than that
        earliest_afs = 4
        for idx in range(3, -1, -1):
            if runtime_afs_fusion[idx]:
                earliest_afs = idx
        
        for idx in range(5):
            if idx <= earliest_afs and self.AVS_FLAG[idx]:
                avs_pattern.append(True)
            else:
                avs_pattern.append(False)
        
        return fusion_pattern, avs_pattern
    
    
    def move_C_to_N(self, x):
        """
        Assume x is with shape [N C T H W], this function merges C into N which 
        results in shape [N*C 1 T H W]
        """
        N, C, T, H, W = x[2].size()
        x[2] = x[2].reshape(N*C, 1, T, H, W)
        return x
    
    
    def filter_duplicates(self, x):
        """
        Compute a valid mask for near-duplicates and near-zero audios
        """
        mask = None
        if self.GET_MISALIGNED_AUDIO:
            with torch.no_grad():
                audio = x[2]
                N, C, T, H, W = audio.size()
                audio = audio.reshape(N//2, C*2, -1)
                # remove pairs that are near-zero
                audio_std = torch.std(audio, dim=2) ** 2
                mask = audio_std > self.AVS_VAR_THRESH
                mask = mask[:, 0] * mask[:, 1]
                # remove pairs that are near-duplicate
                audio = F.normalize(audio, dim=2)
                similarity = audio[:, 0, :] * audio[:, 1, :]
                similarity = torch.sum(similarity, dim=1)
                similarity = similarity < self.AVS_DUPLICATE_THRESH
                # integrate var and dup mask
                mask = mask * similarity
                # mask = mask.float()
        return mask
    
    
    def get_pos_audio(self, x):
        """
        Slice the data and only take the first half 
        along the first dim for positive data
        """
        x[2], _ = torch.chunk(x[2], 2, dim=0)
        return x
    
    
    def avs_forward(self, features, audio_mask):
        """
        Forward for AVS loss
        """
        loss_list = {}
        avs_pattern = features['avs_pattern']
        for idx in range(5):
            if self.AVS_FLAG[idx]:
                a_pos = features['s{}_a_pos'.format(idx + 1)]
                a_neg = features['s{}_a_neg'.format(idx + 1)]
                fs = features['s{}_fs'.format(idx + 1)]
                fuse = getattr(self, 's{}_fuse'.format(idx + 1))
                avs = getattr(fuse, 'avs')
                loss = avs(fs, a_pos, a_neg, audio_mask)
                if not avs_pattern[idx]:
                    loss = loss * 0.0
                loss_list['s{}_avs'.format(idx + 1)] = loss
        return loss_list
        
        
    def forward(self, x):
        # generate fusion pattern
        fusion_pattern, avs_pattern = self.gen_fusion_avs_pattern()
        
        # tackle misaligned logmel
        if self.GET_MISALIGNED_AUDIO:
            x = self.move_C_to_N(x)
        
        # generate mask for audio
        audio_mask = self.filter_duplicates(x)
        
        # initialize feature list
        features = {'avs_pattern': avs_pattern}
        
        # execute forward
        x = self.s1(x)
        if self.FS_FUSION[0] or self.AFS_FUSION[0]:
            x, interm_feat = self.s1_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[0],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s1_'
            )
        x = self.s2(x)
        if self.FS_FUSION[1] or self.AFS_FUSION[1]:
            x, interm_feat = self.s2_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[1],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s2_'
            )        
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        if self.FS_FUSION[2] or self.AFS_FUSION[2]:
            x, interm_feat = self.s3_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[2],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s3_'
            )
        x = self.s4(x)
        if self.FS_FUSION[3] or self.AFS_FUSION[3]:
            x, interm_feat = self.s4_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[3],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s4_'
            )
        x = self.s5(x)
        if self.AVS_FLAG[4]:
            _, interm_feat = self.s5_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode='FS',
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s5_'
            )
        
        # drop the negative samples in audio
        if self.GET_MISALIGNED_AUDIO:
            x = self.get_pos_audio(x)
        
        x = self.head(x)
        
        if self.training and self.GET_MISALIGNED_AUDIO:
            # compute loss if in training
            loss = self.avs_forward(features, audio_mask)
            return x, loss
        else:
            return x


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.cfg = cfg
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
            )

    def forward(self, x, bboxes=None):
        x = x[:]  # avoid pass by reference
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        self.cfg = cfg

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        # Based on profiling data of activation size, s1 and s2 have the activation sizes
        # that are 4X larger than the second largest. Therefore, checkpointing them gives
        # best memory savings. Further tuning is possible for better memory saving and tradeoffs
        # with recomputing FLOPs.
        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)
            self.s1 = checkpoint_wrapper(s1)
            self.s2 = checkpoint_wrapper(s2)
        else:
            self.s1 = s1
            self.s2 = s2

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None]
                if cfg.MULTIGRID.SHORT_CYCLE
                or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
            )

    def forward(self, x, bboxes=None):
        x = x[:]  # avoid pass by reference
        x = self.s1(x)
        x = self.s2(x)
        y = []  # Don't modify x list in place due to activation checkpoint.
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            y.append(pool(x[pathway]))
        x = self.s3(y)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        # BUG record: missing training/eval mode judgement and fix on 11:10 26/Oct. 
        if cfg.TEST.PROCESS:
            if len(cfg.DATA.TEST_CROP_SIZE_RECT) != 0:
                # spat_sz_list = (int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE_RECT[0] / 32.0)),
                #                 int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE_RECT[1] / 32.0)))
                spatial_size = None
                spatial_size_list = cfg.DATA.TEST_CROP_SIZE_RECT
            else:
                # spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
                # spat_sz_list = [spat_sz, spat_sz]
                spatial_size = cfg.DATA.TEST_CROP_SIZE
                spatial_size_list = [cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE]
        else:
            if len(cfg.DATA.TRAIN_CROP_SIZE_RECT) != 0:
                # spat_sz_list = (int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE_RECT[0] / 32.0)),
                #                 int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE_RECT[1] / 32.0)))
                spatial_size = None
                spatial_size_list = cfg.DATA.TRAIN_CROP_SIZE_RECT
            else:
                # spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
                # spat_sz_list = [spat_sz, spat_sz]
                spatial_size = cfg.DATA.TRAIN_CROP_SIZE
                spatial_size_list = [cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE]
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = spatial_size_list[0] // self.patch_stride[1]
        self.W = spatial_size_list[1] // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        hw_switch_auto = cfg.DATA.TRAIN_CROP_SIZE_RECT_SWITCH_AUTO
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size_list[0], spatial_size_list[1]]
        # assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims
        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                separate_qkv=cfg.MVIT.SEPARATE_QKV,
                hw_switch_auto=hw_switch_auto,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)
            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]

            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.TransformerBasicHead(
                embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                cfg=cfg,
            )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append(["pos_embed"])
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):
        t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def forward(self, x, pm=None, bboxes=None, return_attn=False):
        if pm is not None:
            pm = torch.cat(pm)
            
        if pm is None or pm.sum() == 0:
            return self.forward_(x, bboxes, return_attn)
        else:
            ori_H, ori_W = self.H, self.W
            assert len(pm) == x[0].shape[0]
            pm_index = torch.where(pm == True)[0]
            lm_index = torch.where(pm == False)[0]
            self.H, self.W = ori_W, ori_H
            pm_x = [x[0][pm_index].transpose(-2,-1)]
            pm_x = self.forward_(pm_x, bboxes, return_attn)
            x_all = torch.empty((len(pm), *pm_x.shape[1:]), device=pm_x.device, dtype=pm_x.dtype)
            x_all[pm_index] = pm_x
            self.H, self.W = ori_H, ori_W
            if len(lm_index) != 0:
                lm_x = [x[0][lm_index]]
                lm_x = self.forward_(lm_x, bboxes, return_attn)
                x_all[lm_index] = lm_x
            return x_all

    def forward_(self, x, bboxes=None, return_attn=False):
        x = x[0]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        # try:
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        # except:
        #     print(bcthw)
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for i, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)

        if self.enable_detection:
            x = self.norm(x)
            if self.cls_embed_on:
                x = x[:, 1:]

            B, _, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

            x = self.head([x], bboxes)
        else:
            if self.use_mean_pooling:
                if self.cls_embed_on:
                    x = x[:, 1:]
                x = x.mean(1)
                x = self.norm(x)
            elif self.cls_embed_on:
                x = self.norm(x)
                x = x[:, 0]
            else:  # this is default, [norm->mean]
                x = self.norm(x)
                x = x.mean(1)
            x = self.head(x)

        return x
