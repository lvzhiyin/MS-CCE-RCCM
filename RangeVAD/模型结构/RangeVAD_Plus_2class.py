"""
RangeVAD-Plus 2分类消融模型

架构: Spectral Encoder → DFSMN×3 → BiLSTM → 2分类Head
与 RangeVAD-Plus 三分类版完全相同，仅:
  1. 移除 InvertedResidual 模块
  2. 输出头改为2分类 [非语音(0), 语音(1)]

用于验证三分类(区分干净/带噪/非语音)相比二分类的GFAR优势。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDSConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=0,
                                    dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        x_padded = F.pad(x, (self.padding, 0))
        x = self.depthwise(x_padded)
        x = self.pointwise(x)
        return x


class DFSMNBlock(nn.Module):
    def __init__(self, dim, look_back=20, dropout=0.1):
        super().__init__()
        self.look_back = look_back
        self.linear = nn.Linear(dim, dim)
        self.memory_weights = nn.Parameter(torch.randn(look_back) * 0.01)
        self.dw_conv = CausalDSConv1d(dim, dim, kernel_size=3)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        h = self.linear(x)
        memory_out = torch.zeros_like(h)
        for i in range(1, self.look_back + 1):
            shifted = F.pad(h[:, :-i, :], (0, 0, i, 0))
            memory_out = memory_out + self.memory_weights[i - 1] * shifted
        h = h + memory_out
        h_conv = self.dw_conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + h_conv
        h = self.dropout(h)
        h = self.norm(h + x)
        return h


class RangeVADPlus2Class(nn.Module):
    """
    2分类消融版: Spectral Encoder → DFSMN×3 → BiLSTM → 2分类Head
    """

    def __init__(self, mel_dim=80, hidden_dim=64, dfsmn_blocks=3, look_back=20):
        super().__init__()

        # Stage 1: Spectral Encoder
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.stem_padding = 4

        # Stage 2: DFSMN Backbone
        self.dfsmn_blocks = nn.ModuleList([
            DFSMNBlock(hidden_dim, look_back=look_back)
            for _ in range(dfsmn_blocks)
        ])

        # Stage 3: BiLSTM Context Fusion
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.2)

        # Stage 4: 2-Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"RangeVAD-Plus-2Class 总参数量: {total:,} ({total/1000:.1f}K)")
        stages = [
            ("SpectralEncoder", self.spectral_encoder),
            ("DFSMN×3", self.dfsmn_blocks),
            ("BiLSTM", self.bilstm),
            ("2ClassHead", self.classifier),
        ]
        for name, module in stages:
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:<20} {n:>8,} ({n/1000:>5.1f}K)")

    def forward(self, x):
        B, T, _ = x.shape

        x = x.transpose(1, 2)
        x = F.pad(x, (self.stem_padding, 0))
        x = self.spectral_encoder(x)
        x = x.transpose(1, 2)

        for dfsmn in self.dfsmn_blocks:
            x = dfsmn(x)

        x, _ = self.bilstm(x)
        x = self.lstm_dropout(x)

        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = RangeVADPlus2Class()
    x = torch.randn(4, 100, 80)
    logits = model(x)
    print(f"输入: {x.shape} → 输出: {logits.shape}")
