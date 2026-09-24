import time
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Mapping, Any

import os
import copy
import math
import pyiqa
import einops
import numpy as np
from utils.utils import *

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
)

from .spaced_sampler import SpacedSampler
from einops import rearrange
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
)
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from utils.common import frozen_module

class CDDM(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            use_linear_in_transformer=False,
            control_model_ratio=1.0,        # ratio of the control model size compared to the base model. [0, 1]
            learn_embedding=True,
            control_scale=1.0
    ):
        super().__init__()

        self.learn_embedding = learn_embedding
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.control_scale = control_scale

        ################# start control model variations #################
        base_model = UNetModel(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
        )  # initialise control model from base model
        self.control_model = ControlModule(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
            control_model_ratio=control_model_ratio,
        )  # initialise pretrained model

        ################# end control model variations #################

        self.enc_zero_convs_out = nn.ModuleList([])

        self.middle_block_out = None
        self.middle_block_in = None

        self.dec_zero_convs_out = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        ################# Gather Channel Sizes #################
        for module in self.control_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

        ch_inout_ctr['mid'].append((self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
        ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

        for module in base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        ################# Build zero convolutions #################
        self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

        self.dec_zero_convs_out.append(
            self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(ch_inout_ctr['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
            )
        for i in range(len(ch_inout_ctr['enc'])):
            self.enc_zero_convs_out.append(self.make_zero_conv(
                in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
                    )

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list) * self.control_scale)

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, base_model, **kwargs):
        # DEBUG PRINTS
        # print(self.middle_block_out[0].weight.max(), self.middle_block_out[0].weight.min())
        # print(self.enc_zero_convs_out[0][0].weight.max(), self.enc_zero_convs_out[0][0].weight.min())
        # print(self.enc_zero_convs_out[-1][0].weight.max(), self.enc_zero_convs_out[-1][0].weight.min())
        # print(self.dec_zero_convs_out[0][0].weight.max(), self.dec_zero_convs_out[0][0].weight.min())
        # print(self.dec_zero_convs_out[-1][0].weight.max(), self.dec_zero_convs_out[-1][0].weight.min())
        # print()

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.control_model.time_embed(t_emb)
        emb_base = base_model.time_embed(t_emb)

        h_base = x.type(base_model.dtype)
        h_ctr = torch.cat((h_base, hint), dim=1)
        hs_base = []
        hs_ctr = []
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list)

        ###################### Cross Control  ######################

        # input blocks (encoder)
        for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
            h_base = module_base(h_base, emb_base, context)
            h_ctr = module_ctr(h_ctr, emb, context)
            
            h_base = h_base + next(it_enc_convs_out)(h_ctr, emb) * next(scales)
            # feat_ctr = next(it_enc_convs_out)(h_ctr, emb) * next(scales)
            # feat_ctr *= 0
            # h_base = h_base + feat_ctr

            hs_base.append(h_base)
            hs_ctr.append(h_ctr)

        # mid blocks (bottleneck)
        h_base = base_model.middle_block(h_base, emb_base, context)
        h_ctr = self.control_model.middle_block(h_ctr, emb, context)

        h_base = h_base + self.middle_block_out(h_ctr, emb) * next(scales)
        # feat_ctr = self.middle_block_out(h_ctr, emb) * next(scales)
        # feat_ctr *= 0
        # h_base = h_base + feat_ctr

        # output blocks (decoder)
        for module_base in base_model.output_blocks:
            h_base = h_base + next(it_dec_convs_out)(hs_ctr.pop(), emb) * next(scales)
            # feat_ctr = next(it_dec_convs_out)(hs_ctr.pop(), emb) * next(scales)
            # feat_ctr *= 0
            # h_base = h_base + feat_ctr

            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)

        return base_model.out(h_base)
    
class ControlModule(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        control_model_ratio=1.0,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)


        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        model_channels = int(model_channels * control_model_ratio)
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels+hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_head_channels = find_denominator(ch, self.num_head_channels)
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

def find_denominator(number, start):
    if start >= number:
        return number
    while (start != 0):
        residual = number % start
        if residual == 0:
            return start
        start -= 1

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # if find_denominator(channels, 32) < 32:
    #     print(f'[USING GROUPNORM OVER LESS CHANNELS ({find_denominator(channels, 32)}) FOR {channels} CHANNELS]')
    return GroupNorm_leq32(find_denominator(channels, 32), channels)

class GroupNorm_leq32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    
class DiffEIC(LatentDiffusion):

    def __init__(
        self, 
        control_stage_config: Mapping[str, Any], 
        control_key: str,
        sd_locked: bool,
        # synch_control: bool,
        learning_rate: float,
        aux_learning_rate: float,
        l_bpp_weight: float,
        l_guide_weight: float,
        l_semantic_weight: float,
        sl_t_type: str,
        sl_metric: str,
        sl_loc: str,
        sync_path: str, 
        synch_control: bool,
        ckpt_path_pre: str,
        preprocess_config: Mapping[str, Any],
        preprocess_semantic_config: Mapping[str, Any],
        preprocess_tag_config: Mapping[str, Any],
        calculate_metrics: Mapping[str, Any],
        c_ucg_rate: float = 0.1,
        c_cfg_scale: float = 7.5,
        roi_loss_enabled: bool = False,
        roi_weight: float = 4.0,
        roi_background_weight: float = 1.0,
        roi_semantic_enabled: bool = False,
        validation_sample_steps: int = 20,
        compact_checkpoint: bool = True,
        external_text_conditioning: bool = False,
        *args, 
        **kwargs
    ) -> "DiffEIC":
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.preprocess_model = instantiate_from_config(preprocess_config)
        self.preprocess_semantic_model = instantiate_from_config(preprocess_semantic_config)
        self.preprocess_tag_model = instantiate_from_config(preprocess_tag_config)
        self.c_ucg_rate = c_ucg_rate
        self.c_cfg_scale = c_cfg_scale
        self.roi_loss_enabled = roi_loss_enabled
        self.roi_weight = float(roi_weight)
        self.roi_background_weight = float(roi_background_weight)
        self.roi_semantic_enabled = roi_semantic_enabled
        self.validation_sample_steps = int(validation_sample_steps)
        self.compact_checkpoint = compact_checkpoint
        self.external_text_conditioning = external_text_conditioning
        if self.roi_weight <= 0 or self.roi_background_weight <= 0:
            raise ValueError("ROI and background weights must be positive")
        self._validation_outputs = []
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)
        if ckpt_path_pre is not None:
            self.load_preprocess_ckpt(ckpt_path_pre=ckpt_path_pre)

        self.control_key = control_key
        self.sd_locked = sd_locked
        if self.sd_locked:
            # Excluding SD from the optimizer is not enough: leaving
            # requires_grad=True retains parameter gradients, wastes VRAM and
            # makes compact checkpoints accidentally include the whole UNet.
            frozen_module(self.model.diffusion_model)

        self.learning_rate = learning_rate
        self.aux_learning_rate = aux_learning_rate
        self.l_bpp_weight = float(l_bpp_weight)
        self.l_guide_weight = float(l_guide_weight)
        self.l_semantic_weight = float(l_semantic_weight)
        assert sl_t_type in ['random', 'zero'] or sl_t_type.startswith('random_'), \
            "sl_t_type should be 'random' or 'zero' or 'random_x_x' where x_x indicates the range."
        assert sl_metric in ['cos', 'normalized_l2'], "sl_metric should be 'cos' or 'normalized_l2'"
        assert sl_loc.startswith('enc_') or sl_loc == 'mid', "sl_loc should be 'enc_1',  ..., 'enc_12', or 'mid'."
        self.sl_t_type = sl_t_type
        self.sl_metric = sl_metric
        self.sl_loc = sl_loc

        self.calculate_metrics = calculate_metrics
        # Register metrics as child modules so Lightning moves LPIPS and the
        # other metric networks to the same device as the codec.
        self.metric_funcs = torch.nn.ModuleDict()
        for _, opt in calculate_metrics.items(): 
            mopt = opt.copy()
            name = mopt.pop('type', None)
            mopt.pop('better', None)
            self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

    def apply_condition_encoder(self, control, x):
        c_latent, likelihoods, q_likelihoods = self.preprocess_model(control, x)
        return c_latent, likelihoods, q_likelihoods
    
    def encode_tag_ids(self, tag_ids, tag_codelength=13):
        """
        Encode tag IDs using fixed-length encoding.
        
        Args:
            tag_ids: List of tag IDs (integers)
            tag_codelength: Number of bits per tag (default: 13)
            
        Returns:
            bytes: Encoded tag data
        """
        # Convert tag_ids to integers if they're not already
        tag_ids = [int(tag) for tag in tag_ids]
        
        # Calculate total bits needed
        total_bits = len(tag_ids) * tag_codelength
        # Calculate number of bytes needed (round up)
        num_bytes = (total_bits + 7) // 8
        
        # Create a bytearray to store the encoded data
        encoded = bytearray(num_bytes)
        
        # Encode each tag ID
        bit_offset = 0
        for tag_id in tag_ids:
            # Ensure tag_id fits in tag_codelength bits
            assert 0 <= tag_id < (1 << tag_codelength), f"Tag ID {tag_id} exceeds {tag_codelength}-bit range"
            
            # Pack the tag_id into the byte array
            for i in range(tag_codelength):
                if tag_id & (1 << (tag_codelength - 1 - i)):
                    byte_idx = bit_offset // 8
                    bit_idx = bit_offset % 8
                    encoded[byte_idx] |= (1 << (7 - bit_idx))
                bit_offset += 1
        
        return bytes(encoded)
    
    def decode_tag_ids(self, encoded_bytes, num_tags, tag_codelength=13):
        """
        Decode tag IDs from fixed-length encoding.
        
        Args:
            encoded_bytes: Encoded tag data (bytes)
            num_tags: Number of tags to decode
            tag_codelength: Number of bits per tag (default: 13)
            
        Returns:
            List of tag IDs (integers)
        """
        tag_ids = []
        bit_offset = 0
        
        for _ in range(num_tags):
            tag_id = 0
            for i in range(tag_codelength):
                byte_idx = bit_offset // 8
                bit_idx = bit_offset % 8
                if byte_idx < len(encoded_bytes):
                    if encoded_bytes[byte_idx] & (1 << (7 - bit_idx)):
                        tag_id |= (1 << (tag_codelength - 1 - i))
                bit_offset += 1
            tag_ids.append(tag_id)
        
        return tag_ids
    
    @torch.no_grad()
    def apply_condition_compress(
        self, control, stream_path, H, W, tag_ids=None, tag_codelength=13,
        domain_metadata_codes=None,
    ):
        ref = self.encode_first_stage(control * 2 - 1).mode() * self.scale_factor
        out = self.preprocess_model.compress(control, ref)
        shape = out["shape"]
        with Path(stream_path).open("wb") as f:
            write_body(f, shape, out["strings"])
            
            # Write tag information if provided
            if tag_ids is not None:
                # Convert to integers if needed
                tag_ids = [int(tag) for tag in tag_ids]
                
                # Encode tags using fixed-length encoding
                encoded_tags = self.encode_tag_ids(tag_ids, tag_codelength)
                
                # Write number of tags (4 bytes)
                write_uints(f, (len(tag_ids),))
                # Write tag code length (4 bytes)
                write_uints(f, (tag_codelength,))
                # Write length of encoded data (4 bytes)
                write_uints(f, (len(encoded_tags),))
                # Write encoded tags
                write_bytes(f, encoded_tags)
            if domain_metadata_codes is not None:
                codes = bytes(int(code) & 0xFF for code in domain_metadata_codes)
                if len(codes) != control.shape[0]:
                    raise ValueError("one domain metadata code is required per image")
                write_bytes(f, codes)
        
        size = filesize(stream_path)
        bpp = float(size) * 8 / (H * W)
        return bpp

    @torch.no_grad()
    def apply_condition_decompress(
        self, stream_path, dec_tag_ids=False, dec_domain_metadata=False,
        domain_metadata_count=1,
    ):
        with Path(stream_path).open("rb") as f:
            strings, shape = read_body(f)
            
            # Decode tags if requested
            decoded_tag_ids = None
            if dec_tag_ids:
                try:
                    # Read number of tags (4 bytes)
                    num_tags = read_uints(f, 1)[0]
                    # Read tag code length (4 bytes)
                    tag_codelength = read_uints(f, 1)[0]
                    # Read length of encoded data (4 bytes)
                    encoded_length = read_uints(f, 1)[0]
                    # Read encoded tags
                    encoded_tags = read_bytes(f, encoded_length)
                    
                    # Decode tags
                    decoded_tag_ids = self.decode_tag_ids(encoded_tags, num_tags, tag_codelength)
                except:
                    # If tag data is not available, return None
                    decoded_tag_ids = None
            domain_metadata_codes = None
            if dec_domain_metadata:
                domain_metadata_codes = list(read_bytes(f, int(domain_metadata_count)))
        
        c_latent = self.preprocess_model.decompress(strings, shape)
        
        if dec_tag_ids and dec_domain_metadata:
            return c_latent, decoded_tag_ids, domain_metadata_codes
        if dec_tag_ids:
            return c_latent, decoded_tag_ids
        if dec_domain_metadata:
            return c_latent, domain_metadata_codes
        return c_latent
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs) 
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        c_latent, likelihoods, q_likelihoods = self.apply_condition_encoder(control, x)
        if (
            self.preprocess_semantic_model.enabled
            or self.preprocess_tag_model.enabled
            or self.external_text_conditioning
        ):
            if self.training and torch.rand(1) < self.c_ucg_rate:    # randomly drop for classifier free guidance
                # c_semantic = torch.zeros_like(c_semantic)
                drop_cond = True
            else:
                drop_cond = False
            if self.preprocess_semantic_model.enabled:
                c_semantic, bits_sem = self.preprocess_semantic_model(control)  # control: [0, 1]
                assert c_semantic is not None, "Semantic model is enabled but no semantic information is provided."
                # if self.training and torch.rand(1) < self.c_ucg_rate:    # randomly drop for classifier free guidance
                #     c_semantic = torch.zeros_like(c_semantic)
                if drop_cond:
                    c_semantic = torch.zeros_like(c_semantic)
                c = c_semantic   # overwrite c with c_semantic
            if self.preprocess_tag_model.enabled:
                c_tag, bits_tag = self.preprocess_tag_model(control)  # control: [0, 1]
                assert c_tag is not None, "Tag model is enabled but no tag information is provided."
                # self.cond_stage_model.encode(c)
                if drop_cond:
                    c_tag = [''] * len(c_tag)
                domain_prompts = batch.get("txt")
                if domain_prompts is not None:
                    if bs is not None:
                        domain_prompts = domain_prompts[:bs]
                    c_tag = [
                        ", ".join(part for part in (tag, prompt) if part)
                        for tag, prompt in zip(c_tag, domain_prompts)
                    ]
                c_tag = self.cond_stage_model.encode(c_tag)
                if self.preprocess_semantic_model.enabled:
                    # concat
                    c = torch.cat([c, c_tag], 1)
                else:
                    # replace
                    c = c_tag   # overwrite c with c_tag
            elif self.external_text_conditioning:
                # Cached RAM++ tags arrive through batch["txt"]. Re-encode
                # here so classifier-free dropout remains identical to the
                # original online TagGCM path instead of silently keeping the
                # conditioning for every training example.
                c_text = batch.get("txt")
                if c_text is None:
                    raise KeyError('external_text_conditioning requires batch["txt"]')
                if bs is not None:
                    c_text = c_text[:bs]
                if drop_cond:
                    c_text = [''] * len(c_text)
                c_text = self.cond_stage_model.encode(c_text)
                if self.preprocess_semantic_model.enabled:
                    c = torch.cat([c, c_text], 1)
                else:
                    c = c_text

        N , _, H, W = control.shape
        num_pixels = N * H * W
        bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
        q_bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in q_likelihoods)
        sem_bpp = bits_sem / num_pixels if self.preprocess_semantic_model.enabled else torch.tensor(0.).to(bpp.device)
        tag_bits = bits_tag if self.preprocess_tag_model.enabled else torch.tensor(0.0, device=bpp.device)
        cached_tag_bits = batch.get("tag_payload_bits")
        if cached_tag_bits is not None and not self.preprocess_tag_model.enabled:
            if bs is not None:
                cached_tag_bits = cached_tag_bits[:bs]
            tag_bits = tag_bits + torch.as_tensor(cached_tag_bits, device=bpp.device).float().sum()
        metadata_bits = batch.get("domain_metadata_bits")
        if metadata_bits is not None:
            if bs is not None:
                metadata_bits = metadata_bits[:bs]
            metadata_bits = torch.as_tensor(metadata_bits, device=bpp.device).float().sum()
            tag_bits = tag_bits + metadata_bits
        tag_bpp = tag_bits / num_pixels
        roi_mask = batch.get("roi_mask")
        if roi_mask is not None:
            if bs is not None:
                roi_mask = roi_mask[:bs]
            roi_mask = torch.as_tensor(roi_mask, device=control.device).float()
            if roi_mask.ndim == 3:
                roi_mask = roi_mask.unsqueeze(-1)
            roi_mask = einops.rearrange(roi_mask, 'b h w c -> b c h w').contiguous()
        # return x, dict(c_crossattn=[c], c_latent=[c_latent], bpp=bpp, q_bpp=q_bpp, control=[control])
        return x, dict(
            c_crossattn=[c], c_latent=[c_latent], control=[control], 
            bpp=bpp, q_bpp=q_bpp, sem_bpp=sem_bpp, tag_bpp=tag_bpp,
            roi_mask=roi_mask)
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1)

        eps = self.control_model(
            x=x_noisy, timesteps=t, context=cond_txt, hint=cond_hint, base_model=diffusion_model)
        
        return eps
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)
    
    @torch.no_grad()
    def log_images(self, batch, sample_steps=50, bs=2):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        # bpp = c["q_bpp"]
        bpp = c["q_bpp"] + c["sem_bpp"] + c["tag_bpp"]
        bpp_img = [f'{bpp:2f}']*4
        c_latent = c["c_latent"][0]
        control = c["control"][0]
        c = c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = control
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2
        
        samples = self.sample_log(
            cond={"c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )
        x_samples = self.decode_first_stage(samples)
        log["samples"] = (x_samples + 1) / 2

        return log, bpp
    
    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_latent"][0].shape
        shape = (b, self.channels, h, w)

        # if self.preprocess_semantic_model.enabled:
        if self.preprocess_semantic_model.enabled or self.preprocess_tag_model.enabled:
            unconditional_guidance_scale = self.c_cfg_scale
            # unconditional_conditioning = torch.zeros_like(cond["c_crossattn"][0])
            unconditional_conditioning = copy.deepcopy(cond)
            # unconditional_conditioning["c_crossattn"] = [torch.zeros_like(cond["c_crossattn"][0])]
            if self.preprocess_semantic_model.enabled:
                n_tokens = 256 // (self.preprocess_semantic_model.feature_postprocessor.postprocess_downscale**2)
                uncond_semantic = torch.zeros(b, n_tokens, cond["c_crossattn"][0].shape[-1]).to(self.device)
                uncond_crossattn = uncond_semantic
            if self.preprocess_tag_model.enabled:
                uncond_tag = self.cond_stage_model.encode([''] * b)
                if self.preprocess_semantic_model.enabled:
                    uncond_crossattn = torch.cat([uncond_crossattn, uncond_tag], 1)
                else:
                    uncond_crossattn = uncond_tag
            if self.external_text_conditioning and not self.preprocess_tag_model.enabled:
                uncond_crossattn = self.cond_stage_model.encode([''] * b)
            unconditional_conditioning["c_crossattn"] = [uncond_crossattn]

            samples = sampler.sample(
                steps, shape, cond, 
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning
            )
        else:
            samples = sampler.sample(
                steps, shape, cond, 
                unconditional_guidance_scale=1.0, unconditional_conditioning=None
            )
        return samples
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(param for name, param in self.preprocess_model.named_parameters() 
                       if not name.endswith('.quantiles'))
        params += list(param for name, param in self.preprocess_semantic_model.named_parameters() 
                       if name.startswith('final_layers'))  # only update the final layers of semantic model
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        params = [param for param in params if param.requires_grad]
        aux_params = [
            param for name, param in self.preprocess_model.named_parameters()
            if name.endswith('.quantiles') and param.requires_grad
        ]
        groups = [{"params": params, "lr": lr, "name": "main"}]
        if aux_params:
            groups.append({"params": aux_params, "lr": self.aux_learning_rate, "name": "entropy_aux"})
        # One optimizer keeps Lightning 2 automatic optimisation, native
        # gradient accumulation and an unambiguous global_step. CompressAI's
        # quantiles only receive gradients from aux_loss below.
        return torch.optim.AdamW(groups, lr=lr)

    def _weighted_spatial_mean(self, loss_map, roi_mask):
        if roi_mask is None:
            return loss_map.mean()
        mask = F.interpolate(roi_mask, size=loss_map.shape[-2:], mode="nearest")
        mask = mask.clamp(0, 1)
        weights = self.roi_background_weight + (self.roi_weight - self.roi_background_weight) * mask
        if loss_map.shape[1] != 1:
            loss_map = loss_map.mean(dim=1, keepdim=True)
        return (loss_map * weights).sum() / weights.sum().clamp_min(1e-8)

    def _region_spatial_mean(self, loss_map, roi_mask, foreground):
        if roi_mask is None:
            return loss_map.new_zeros(())
        mask = F.interpolate(roi_mask, size=loss_map.shape[-2:], mode="nearest").clamp(0, 1)
        if not foreground:
            mask = 1.0 - mask
        if loss_map.shape[1] != 1:
            loss_map = loss_map.mean(dim=1, keepdim=True)
        denominator = mask.sum()
        if denominator <= 0:
            return loss_map.new_zeros(())
        return (loss_map * mask).sum() / denominator
    
    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        prefix = 'T' if self.training else 'V'

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # diffusion loss
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/l_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # bpp loss
        loss_bpp = cond['bpp']
        guide_bpp = cond['q_bpp']
        loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
        loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp
        loss_dict.update({f'{prefix}/sem_bpp': cond['sem_bpp'].mean()})
        loss_dict.update({f'{prefix}/tag_bpp': cond['tag_bpp'].mean()})
        loss_dict.update({f'{prefix}/total_bpp': guide_bpp+cond['sem_bpp'].mean()+cond['tag_bpp'].mean()})

        # guide loss
        c_latent = cond['c_latent'][0][:,:4,:,:]
        loss_guide_map = self.get_loss(c_latent, x_start, mean=False)
        loss_guide = self._weighted_spatial_mean(
            loss_guide_map,
            cond.get('roi_mask') if self.roi_loss_enabled else None,
        )
        loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
        if self.roi_loss_enabled:
            loss_dict.update({
                f'{prefix}/l_guide_roi': self._region_spatial_mean(
                    loss_guide_map, cond.get('roi_mask'), foreground=True),
                f'{prefix}/l_guide_bg': self._region_spatial_mean(
                    loss_guide_map, cond.get('roi_mask'), foreground=False),
                f'{prefix}/roi_fraction': cond['roi_mask'].float().mean()
                    if cond.get('roi_mask') is not None else loss_guide.new_zeros(()),
            })
        loss += self.l_guide_weight * loss_guide
        # loss_dict.update({f'{prefix}/loss': loss})

        # semantic loss
        if self.l_semantic_weight != 0:
            cond_txt = torch.cat(cond['c_crossattn'], 1)    # text embedding
            cond_hint = torch.cat(cond['c_latent'], 1)      # compressed latent
            cond_hint_ori = x_start                         # clean latent
            if self.sl_t_type == 'zero':
                t_tmp = torch.zeros_like(t)
            elif self.sl_t_type == 'random':
                # create noisy x and x_hat with the same t and noise
                cond_hint = self.q_sample(x_start=cond_hint, t=t, noise=noise)
                cond_hint_ori = self.q_sample(x_start=cond_hint_ori, t=t, noise=noise)
                t_tmp = t
            elif self.sl_t_type.startswith('random_'):
                lower_bound, upper_bound = float(self.sl_t_type.split('_')[1]), float(self.sl_t_type.split('_')[2])
                lower_bound, upper_bound = int(lower_bound * self.num_timesteps), int(upper_bound * self.num_timesteps)
                t_tmp = torch.randint(lower_bound, upper_bound, (t.shape[0],)).to(t.device)
                cond_hint = self.q_sample(x_start=cond_hint, t=t_tmp, noise=noise)
                cond_hint_ori = self.q_sample(x_start=cond_hint_ori, t=t_tmp, noise=noise)
            else:
                raise NotImplementedError()
            features_enc, features_mid = self.model.diffusion_model.get_encode_features(
                cond_hint, t_tmp, cond_txt)
            features_enc_ori, features_mid_ori = self.model.diffusion_model.get_encode_features(
                cond_hint_ori, t_tmp, cond_txt)
            if self.sl_loc == 'mid':
                sl_x_ori = features_mid_ori
                sl_x = features_mid
            elif self.sl_loc.startswith('enc'):
                enc_idx = int(self.sl_loc.split('_')[1]) - 1
                sl_x_ori = features_enc_ori[enc_idx]
                sl_x = features_enc[enc_idx]
            loss_semantic_map = self.get_loss_semantic(sl_x_ori, sl_x, self.sl_metric, mean=False)
            loss_semantic = self._weighted_spatial_mean(
                loss_semantic_map.unsqueeze(1),
                cond.get('roi_mask') if self.roi_semantic_enabled else None,
            )
            loss_dict.update({f'{prefix}/l_semantic': loss_semantic})
            loss += self.l_semantic_weight * loss_semantic
            # Visualize cond_hint and cond_hint_ori: t->0 corresponds to clean image
            # import torchvision
            # torchvision.utils.save_image(cond_hint_ori[:,:3], 'cond_hint_ori.png', normalize=True)
            # torchvision.utils.save_image(cond_hint[:,:3], 'cond_hint.png', normalize=True)
            # torchvision.utils.save_image(cond_hint_ori_noisy[:,:3], 'cond_hint_ori_noisy.png', normalize=True)
            # torchvision.utils.save_image(cond_hint_noisy[:,:3], 'cond_hint_noisy.png', normalize=True)
            # torchvision.utils.save_image(x_noisy[:,:3], 'x_noisy.png', normalize=True)

        loss_dict.update({f'{prefix}/loss': loss})  # total loss

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)
        aux_loss = self.preprocess_model.aux_loss()
        total_loss = loss + aux_loss
        loss_dict["T/aux_loss"] = aux_loss
        loss_dict["T/optim_loss"] = total_loss
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return total_loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = []
        log, bpp = self.log_images(batch, sample_steps=self.validation_sample_steps, bs=1)
        out.append(bpp.cpu())
        # save images
        save_dir = os.path.join(self.logger.save_dir, "validation", f'{self.global_step}')
        os.makedirs(save_dir, exist_ok=True)
        image = log["samples"].detach().cpu()
        if image.dtype == torch.bfloat16:
            image = image.float()
        image = image.numpy().squeeze().transpose(1,2,0)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        path = os.path.join(save_dir, f'{batch_idx}.png')
        Image.fromarray(image).save(path)

        control = log["control"].detach().cpu()
        control = control.numpy().squeeze().transpose(1,2,0)
        control = (control * 255).clip(0, 255).astype(np.uint8)

        metric_data = [img2tensor(image).unsqueeze(0) / 255.0, img2tensor(control).unsqueeze(0) / 255.0]

        for name, _ in self.calculate_metrics.items():
            out.append(self.metric_funcs[name](*metric_data))
        
        values = torch.stack([torch.as_tensor(value, device=self.device).float().mean() for value in out])
        self._validation_outputs.append(values)
        return values

    def on_validation_epoch_start(self):
        self._validation_outputs.clear()

    def on_validation_epoch_end(self):
        if not self._validation_outputs:
            return
        avg_out = torch.stack(self._validation_outputs).mean(dim=0)
        self.log("avg_bpp", avg_out[0],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        for i, (name, _) in enumerate(self.calculate_metrics.items()):
            self.log(f"avg_{name}", avg_out[i+1],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self._validation_outputs.clear()
        
    def load_preprocess_ckpt(self, ckpt_path_pre):
        try:
            ckpt = torch.load(ckpt_path_pre, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path_pre, map_location="cpu")
        self.preprocess_model.load_state_dict(ckpt)
        print(['CONTROL WEIGHTS LOADED'])
        del ckpt
        
    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        try:
            ckpt_base = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            ckpt_base = torch.load(path, map_location="cpu")

        if synch_control:
            # add copy for control_module weights from the base model
            for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        if ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape:
                            if len(ckpt_base['state_dict'][key].shape) == 1:
                                dim = 0
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim]
                            else:
                                dim = 0
                                control_dim_0 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                dim = 1
                                control_dim_1 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim_0, :control_dim_1, ...]
                        else:
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]
            
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')
        del ckpt_base

    def get_loss_semantic(self, x, y, loss_type='cos', mean=True):
        if loss_type == 'cos':
            cos_sim = F.cosine_similarity(x, y, dim=1)
            loss = 1 - cos_sim
            return loss.mean() if mean else loss
        elif loss_type == 'normalized_l2':  # equal to 2*(1 - cosine_similarity)
                                            # x̂₁ = x₁ / ||x₁|| 
                                            # x̂₂ = x₂ / ||x₂||
                                            # ||x̂₁ - x̂₂||² = (x̂₁ - x̂₂)ᵀ(x̂₁ - x̂₂)
                                            # = x̂₁ᵀx̂₁ + x̂₂ᵀx̂₂ - 2x̂₁ᵀx̂₂
                                            # x̂₁ᵀx̂₁ = ||x̂₁||² = 1
                                            # x̂₂ᵀx̂₂ = ||x̂₂||² = 1
                                            # x̂₁ᵀx̂₂ = cos(x₁,x₂)
                                            # ||x̂₁ - x̂₂||² = 1 + 1 - 2cos(x₁,x₂)
                                            #              = 2(1 - cos(x₁,x₂))
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            loss = torch.sum((x_norm - y_norm) ** 2, dim=1)
            return loss.mean() if mean else loss
        else:
            raise NotImplementedError('loss_type [{}] not implemented'.format(loss_type))

    def _compact_state_keys(self):
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        prefixes = (
            "control_model.",
            "preprocess_model.",
            "preprocess_semantic_model.final_layers.",
        )
        return {
            key
            for key in self.state_dict()
            if key in trainable or key.startswith(prefixes)
        }

    def on_save_checkpoint(self, checkpoint):
        if not self.compact_checkpoint:
            return
        keep = self._compact_state_keys()
        checkpoint["state_dict"] = {
            key: value for key, value in checkpoint["state_dict"].items() if key in keep
        }
        checkpoint["wild_diff_compact"] = True
        checkpoint["wild_diff_compact_keys"] = sorted(keep)

    def on_load_checkpoint(self, checkpoint):
        if not checkpoint.get("wild_diff_compact"):
            return
        saved = checkpoint.get("state_dict", {})
        expected = self._compact_state_keys()
        missing_trainable = sorted(expected - set(saved))
        if missing_trainable:
            preview = ", ".join(missing_trainable[:10])
            raise RuntimeError(f"compact checkpoint is missing trainable state: {preview}")
        # Lightning restores model state strictly. Fill omitted frozen weights
        # with the already-instantiated SD/VAE/RAM++ state without cloning.
        for key, value in self.state_dict().items():
            saved.setdefault(key, value)
