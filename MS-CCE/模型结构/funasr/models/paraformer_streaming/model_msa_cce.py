#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
ParaformerStreaming with MSA-CCE (Multi-Scale Adaptive Causal Context Embedding)

针对靶场指令音频中枪声短促（0.3-0.5秒）、信息分布高度不均匀的特点，
在原CCE基础上引入多尺度因果卷积和自适应注入权重。

LFR参数: lfr_m=7, lfr_n=6, frame_shift=10ms → LFR后帧移=60ms
    k=3  → 180ms (枪声瞬态)
    k=6  → 360ms (口令语流)
    k=9  → 540ms (句子级上下文)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.register import tables
from funasr.models.paraformer_streaming.model import ParaformerStreaming


class MultiScaleAdaptiveCCE(nn.Module):

    def __init__(
        self,
        dim: int = 560,
        kernel_sizes: list = None,
        embed_weight_range: tuple = (0.3, 1.5),
        chunk_size: int = 16,
        gate_reduction: int = 4,
        adaptive: bool = True,
        fixed_weight: float = 0.8,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 6, 9]

        self.dim = dim
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)
        self.w_min, self.w_max = embed_weight_range
        self.chunk_size = chunk_size
        self.adaptive = adaptive
        self.fixed_weight = fixed_weight

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(dim, dim, k, stride=1, padding=0, bias=True)
            for k in kernel_sizes
        ])

        gate_dim = self.num_branches * dim
        hidden_dim = max(gate_dim // gate_reduction, 32)
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_branches),
        )

        self.adaptive_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.relu = nn.ReLU()

        for conv in self.conv_branches:
            nn.init.xavier_uniform_(conv.weight, gain=0.01)
            nn.init.zeros_(conv.bias)

    def _compute_chunk_stats(self, x_t, T):
        B, D, _ = x_t.shape
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        stats_list = []

        for j in range(num_chunks):
            start = j * self.chunk_size
            end = min(start + self.chunk_size, T)
            if start >= T:
                break
            chunk = x_t[:, :, start:end]
            chunk_len = end - start
            chunk_mean = chunk.mean(dim=2)
            chunk_var = chunk.var(dim=2, unbiased=False)
            chunk_std = torch.sqrt(chunk_var + 1e-8)
            if chunk_len < 2:
                chunk_std = torch.zeros_like(chunk_mean)
            chunk_max = chunk.max(dim=2)[0]
            stat = torch.stack([chunk_mean, chunk_std, chunk_max], dim=-1).mean(dim=1)
            stats_list.append(stat)

        return torch.stack(stats_list, dim=1)

    def forward(self, x):
        B, T, D = x.shape

        if T < 1:
            return x.clone()

        x_t = x.transpose(1, 2)

        branch_outputs = []
        for i, (conv, k) in enumerate(zip(self.conv_branches, self.kernel_sizes)):
            x_padded = F.pad(x_t, (k - 1, 0))
            branch_out = conv(x_padded)
            branch_outputs.append(branch_out)

        branch_cat = torch.cat(branch_outputs, dim=1)
        branch_cat_t = branch_cat.transpose(1, 2)
        gate_logits = self.fusion_gate(branch_cat_t)
        gate = torch.softmax(gate_logits, dim=-1)

        fused = torch.zeros_like(x_t)
        for i in range(self.num_branches):
            fused += gate[:, :, i:i + 1].transpose(1, 2) * branch_outputs[i]

        fused = self.relu(fused)

        result = torch.zeros_like(x_t)
        for pos in range(0, T, self.chunk_size):
            if pos < fused.shape[2]:
                result[:, :, pos] = fused[:, :, pos]

        if self.adaptive:
            chunk_stats = self._compute_chunk_stats(x_t, T)
            raw_weight = torch.sigmoid(self.adaptive_mlp(chunk_stats))
            w = self.w_min + (self.w_max - self.w_min) * raw_weight

            x_t_out = x_t.clone()
            num_chunks = chunk_stats.shape[1]
            for j in range(num_chunks):
                pos = j * self.chunk_size
                if pos < T:
                    x_t_out[:, :, pos] = x_t[:, :, pos] + w[:, j, 0].unsqueeze(-1) * result[:, :, pos]
        else:
            x_t_out = x_t + self.fixed_weight * result

        return x_t_out.transpose(1, 2)


@tables.register("model_classes", "ParaformerStreamingMSACCE")
class ParaformerStreamingMSACCE(ParaformerStreaming):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cce = kwargs.get("use_cce", True)
        self.cce_dim = kwargs.get("cce_dim", 560)
        self.cce_kernel_sizes = kwargs.get("cce_kernel_sizes", [3, 6, 9])
        self.cce_embed_weight_min = kwargs.get("cce_embed_weight_min", 0.3)
        self.cce_embed_weight_max = kwargs.get("cce_embed_weight_max", 1.5)
        self.cce_chunk_size = kwargs.get("cce_chunk_size", 16)

        if self.use_cce:
            cce_module = MultiScaleAdaptiveCCE(
                dim=self.cce_dim,
                kernel_sizes=self.cce_kernel_sizes,
                embed_weight_range=(self.cce_embed_weight_min, self.cce_embed_weight_max),
                chunk_size=self.cce_chunk_size,
            )
            self.add_module('cce_module', cce_module)

            total_params = sum(p.numel() for p in self.cce_module.parameters())
            gate_params = sum(p.numel() for p in self.cce_module.fusion_gate.parameters())
            adapt_params = sum(p.numel() for p in self.cce_module.adaptive_mlp.parameters())
            conv_params = sum(p.numel() for p in self.cce_module.conv_branches.parameters())
            print(f"[MSA-CCE] MultiScaleAdaptiveCCE initialized:")
            print(f"  dim={self.cce_dim}, kernels={self.cce_kernel_sizes}, "
                  f"weight_range=[{self.cce_embed_weight_min}, {self.cce_embed_weight_max}]")
            print(f"  params: total={total_params:,}, conv={conv_params:,}, "
                  f"gate={gate_params:,}, adapt={adapt_params:,}")
        else:
            self.cce_module = None

    def encode_chunk(self, speech, speech_lengths, cache=None, **kwargs):
        from torch.cuda.amp import autocast

        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        if (self.use_cce
            and self.cce_module is not None
            and speech is not None
            and len(speech.shape) == 3
            and speech.shape[-1] == self.cce_dim):
            speech = self.cce_module(speech)

        encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(
            speech, speech_lengths, cache=cache["encoder"]
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, torch.tensor([encoder_out.size(1)])

    def encode(self, speech, speech_lengths, **kwargs):
        from torch.cuda.amp import autocast

        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        if (self.use_cce
            and self.cce_module is not None
            and speech is not None
            and len(speech.shape) == 3
            and speech.shape[-1] == self.cce_dim):
            speech = self.cce_module(speech)

        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens


@tables.register("model_classes", "ParaformerStreamingMSCce")
class ParaformerStreamingMSCce(ParaformerStreaming):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cce = kwargs.get("use_cce", True)
        self.cce_dim = kwargs.get("cce_dim", 560)
        self.cce_kernel_sizes = kwargs.get("cce_kernel_sizes", [3, 6, 9])
        self.cce_fixed_weight = kwargs.get("cce_fixed_weight", 0.8)
        self.cce_chunk_size = kwargs.get("cce_chunk_size", 16)

        if self.use_cce:
            cce_module = MultiScaleAdaptiveCCE(
                dim=self.cce_dim,
                kernel_sizes=self.cce_kernel_sizes,
                embed_weight_range=(0.3, 1.5),
                chunk_size=self.cce_chunk_size,
                adaptive=False,
                fixed_weight=self.cce_fixed_weight,
            )
            self.add_module('cce_module', cce_module)

            total_params = sum(p.numel() for p in self.cce_module.parameters())
            print(f"[MS-CCE] MultiScaleCCE initialized (no adaptive):")
            print(f"  dim={self.cce_dim}, kernels={self.cce_kernel_sizes}, "
                  f"fixed_weight={self.cce_fixed_weight}")
            print(f"  params: total={total_params:,}")

    def encode_chunk(self, speech, speech_lengths, cache=None, **kwargs):
        from torch.cuda.amp import autocast

        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        if (self.use_cce
            and self.cce_module is not None
            and speech is not None
            and len(speech.shape) == 3
            and speech.shape[-1] == self.cce_dim):
            speech = self.cce_module(speech)

        encoder_out, encoder_out_lens, _ = self.encoder.forward_chunk(
            speech, speech_lengths, cache=cache["encoder"]
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, torch.tensor([encoder_out.size(1)])

    def encode(self, speech, speech_lengths, **kwargs):
        from torch.cuda.amp import autocast

        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        if (self.use_cce
            and self.cce_module is not None
            and speech is not None
            and len(speech.shape) == 3
            and speech.shape[-1] == self.cce_dim):
            speech = self.cce_module(speech)

        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

