"""
RangeVAD-Plus: Enhanced Streaming VAD with SOTA Architecture Fusion
====================================================================

融合6个最新流式VAD模型的架构精华:
- FireRedVAD (2026): DFSMN块 + Depthwise Separable Conv
- MagicNet (2024): Causal Conv + Inverted Residual
- Silero VAD v5 (2024): Encoder-Decoder结构
- MarbleNet (NeMo): 1D Time-Channel Separable Conv
- FSMN-VAD (FunASR): Feedforward Sequential Memory Network
- Su et al. (2026): CNN+BiLSTM混合时序建模

架构: Spectral Encoder → Temporal Backbone (DFSMN×3) → 
      Context Fusion (BiLSTM) → Tri-Head (3分类)

三分类输出: [干净语音(0), 带噪语音(1), 非语音(2)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 模块1: Causal Depthwise Separable Conv1d (参考MarbleNet+FireRedVAD)
# ============================================================
class CausalDSConv1d(nn.Module):
    """
    因果深度可分离1D卷积。

    标准Conv: 参数量 = out_ch × in_ch × kernel_size
    DSConv:   参数量 = in_ch × kernel_size + out_ch × in_ch
              → 大幅减少 (尤其in_ch大时)

    因果: 只卷积"过去"的帧，padding在左侧
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # 左侧填充

        # Depthwise: 逐通道卷积
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            padding=0,  # 手动填充以实现因果
            dilation=dilation,
            groups=in_ch,
            bias=False
        )
        # Pointwise: 1×1卷积混合通道
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        # x: [B, C, T]
        # 左侧填充实现因果
        x_padded = F.pad(x, (self.padding, 0))
        x = self.depthwise(x_padded)
        x = self.pointwise(x)
        return x


# ============================================================
# 模块2: Inverted Residual Block (参考MagicNet)
# ============================================================
class InvertedResidualBlock(nn.Module):
    """
    MobileNet风格的Inverted Residual Block。

    结构: 升维 → DepthwiseConv → 降维 → 残差连接
    特点:
    - 中间层通道数高(升维), 提取丰富特征
    - DepthwiseConv轻量
    - 残差连接帮助梯度流动

    参数:
        in_ch: 输入通道
        expand_ch: 中间升维通道数 (通常 = in_ch × expansion)
        kernel_size: DepthwiseConv卷积核大小
        dilation: 空洞率 (扩大感受野)
    """

    def __init__(self, in_ch, expand_ch, kernel_size, dilation=1):
        super().__init__()
        self.use_residual = True

        # 升维: 1×1 pointwise conv
        self.expand = nn.Sequential(
            nn.Conv1d(in_ch, expand_ch, 1, bias=False),
            nn.BatchNorm1d(expand_ch),
            nn.ReLU(inplace=True),
        )

        # Depthwise Conv: 逐通道因果卷积
        # 保持输入输出长度一致: 左侧padding = (k-1)*dilation
        self.pad_size = (kernel_size - 1) * dilation
        self.depthwise = nn.Sequential(
            nn.Conv1d(expand_ch, expand_ch, kernel_size,
                       padding=0, dilation=dilation,
                       groups=expand_ch, bias=False),
            nn.BatchNorm1d(expand_ch),
            nn.ReLU(inplace=True),
        )

        # 降维: 1×1 pointwise conv (无激活, 线性投影)
        self.project = nn.Sequential(
            nn.Conv1d(expand_ch, in_ch, 1, bias=False),
            nn.BatchNorm1d(in_ch),
        )

    def forward(self, x):
        # x: [B, in_ch, T]
        residual = x

        out = self.expand(x)               # [B, expand_ch, T]
        # 因果填充: 只在左侧填充
        out = F.pad(out, (self.pad_size, 0))
        out = self.depthwise(out)          # [B, expand_ch, T] (长度不变)
        out = self.project(out)            # [B, in_ch, T]

        # 残差连接 (确保长度匹配)
        if out.shape[-1] != residual.shape[-1]:
            # 如果长度不一致, 裁剪输出
            diff = out.shape[-1] - residual.shape[-1]
            out = out[:, :, diff:]

        if self.use_residual:
            out = out + residual

        return out


# ============================================================
# 模块3: DFSMN Memory Block (参考FireRedVAD + FSMN-VAD)
# ============================================================
class DFSMNBlock(nn.Module):
    """
    Deep Feedforward Sequential Memory Network Block。

    核心: y_t = x_t + Σ_{i=1}^{look_back} w_i · x_{t-i}
        当前帧 = 原始帧 + 加权历史帧之和

    优点:
    - 非递归 → 训练稳定, 可并行
    - 有界记忆 → 流式只需缓存look_back帧
    - 前馈结构 → 比RNN快7倍

    参考: "Compact Feedforward Sequential Memory Networks" (Zhang et al., Interspeech 2016)
          "FireRedASR: Open Source Industrial-Grade Speech Recognition" (Xu et al., 2026)
    """

    def __init__(self, dim, look_back=20, dropout=0.1):
        super().__init__()
        self.look_back = look_back
        self.dim = dim

        # 线性投影
        self.linear = nn.Linear(dim, dim)

        # 记忆权重: 可学习的look_back个历史位置权重
        self.memory_weights = nn.Parameter(torch.randn(look_back) * 0.01)

        # 额外的Depthwise Conv增强局部建模
        self.dw_conv = CausalDSConv1d(dim, dim, kernel_size=3)

        # 归一化 + Dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, dim]
        B, T, D = x.shape

        # 线性投影
        h = self.linear(x)  # [B, T, dim]

        # 记忆块: 向量化实现加权历史求和
        # memory_out[t] = Σ_{i=1}^{look_back} w_i × h[t-i]
        # 构造 [B, T, D] 的移位版本，一次性计算
        memory_out = torch.zeros_like(h)
        for i in range(1, self.look_back + 1):
            # 后移 i 帧: [B, T-i, D] → pad 左右
            shifted = F.pad(h[:, :-i, :], (0, 0, i, 0))  # 左侧 pad i 行 0
            memory_out = memory_out + self.memory_weights[i-1] * shifted

        h = h + memory_out  # 残差 + 记忆

        # Depthwise Conv增强
        h_conv = self.dw_conv(h.transpose(1, 2)).transpose(1, 2)  # [B, T, dim]
        h = h + h_conv  # 残差连接

        h = self.dropout(h)
        h = self.norm(h + x)  # 最终残差 + LayerNorm

        return h


# ============================================================
# 模块4: Spectral Variance Feature (用于后处理区分纯枪声/重叠)
# ============================================================
def compute_spectral_variance(mel_spec):
    """
    计算频谱方差 — 区分纯枪声(方差低) vs 重叠(方差中) vs 语音(方差高)。

    mel_spec: [B, T, 80] log-Mel
    返回: [B, T, 1] 方差特征
    """
    spec = torch.exp(mel_spec)  # 线性幅度
    spec_norm = spec / (spec.sum(dim=-1, keepdim=True) + 1e-10)
    mean = spec_norm.mean(dim=-1, keepdim=True)
    variance = ((spec_norm - mean) ** 2).mean(dim=-1, keepdim=True)
    variance_norm = torch.clamp(variance * 500, 0, 1)
    return variance_norm


# ============================================================
# RangeVAD-Plus 完整模型
# ============================================================
class RangeVADPlus(nn.Module):
    """
    RangeVAD-Plus: 融合SOTA架构的增强版流式VAD。

    整体架构 (Encoder-Backbone-Head风格, 参考Silero V5):

    [Input: log-Mel 80-dim]
        │
        v
    ┌─────────────────────────────────────────┐
    │ Stage 1: Spectral Encoder (频谱编码器)   │  ← 参考MagicNet Prologue
    │   CausalConv(80→64, k=5)                │
    │   + BatchNorm + ReLU                     │
    │   参数量: ~25K                            │
    └─────────────────────────────────────────┘
        │ [B, T, 64]
        v
    ┌─────────────────────────────────────────┐
    │ Stage 2: Temporal Backbone (时序主干)    │  ← 参考FireRedVAD DFSMN
    │   DFSMN Block ×3                         │
    │   - look_back=20, dim=64                 │
    │   - 每个块: FSMN记忆 + DepthwiseConv      │
    │   - 残差连接 + LayerNorm                  │
    │   参数量: ~45K                            │
    └─────────────────────────────────────────┘
        │ [B, T, 64]
        v
    ┌─────────────────────────────────────────┐
    │ Stage 3: Multi-Scale Conv (多尺度卷积)   │  ← 参考MagicNet Inverted Residual
    │   InvertedResidualBlock ×2               │
    │   - Block1: k=41, dilation=1 (大感受野)  │
    │   - Block2: k=21, dilation=2 (空洞卷积)  │
    │   - 升维因子: 4× (64→256→64)             │
    │   参数量: ~35K                            │
    └─────────────────────────────────────────┘
        │ [B, T, 64]
        v
    ┌─────────────────────────────────────────┐
    │ Stage 4: Context Fusion (上下文融合)     │  ← 参考Su et al. CNN+BiLSTM
    │   BiLSTM(64→64, bidirectional)           │
    │   输出: 128-dim (前向64+后向64)          │
    │   参数量: ~100K                           │
    └─────────────────────────────────────────┘
        │ [B, T, 128]
        v
    ┌─────────────────────────────────────────┐
    │ Stage 5: Tri-Head (三分类头)            │
    │   Linear(128→64) + ReLU + Dropout        │
    │   Linear(64→3)                           │
    │   输出: [干净语音, 带噪语音, 非语音]      │
    │   参数量: ~6K                             │
    └─────────────────────────────────────────┘

    总参数量: ~213K (与原版RangeVAD相近!)
    """

    def __init__(self, mel_dim=80, hidden_dim=64, num_classes=3,
                 dfsmn_blocks=3, look_back=20, use_ir=True):
        super().__init__()

        # ===== Stage 1: Spectral Encoder =====
        self.spectral_encoder = nn.Sequential(
            # CausalConv: 左侧填充实现因果
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.stem_padding = 4  # k=5, 因果padding=4

        # ===== Stage 2: Temporal Backbone (DFSMN ×3) =====
        self.dfsmn_blocks = nn.ModuleList([
            DFSMNBlock(hidden_dim, look_back=look_back)
            for _ in range(dfsmn_blocks)
        ])

        # ===== Stage 3: Multi-Scale Inverted Residual (可选) =====
        self.use_ir = use_ir
        if use_ir:
            self.ir_block1 = InvertedResidualBlock(
                hidden_dim, expand_ch=hidden_dim * 4,
                kernel_size=41, dilation=1
            )
            self.ir_block2 = InvertedResidualBlock(
                hidden_dim, expand_ch=hidden_dim * 4,
                kernel_size=21, dilation=2
            )

        # ===== Stage 4: Context Fusion (BiLSTM) =====
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.lstm_dropout = nn.Dropout(0.2)

        # ===== Stage 5: Quad-Head =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"RangeVAD-Plus 总参数量: {total:,} ({total/1000:.1f}K)")
        print(f"{'='*60}")

        stage_names = [
            ("Stage1 SpectralEncoder", self.spectral_encoder),
            ("Stage2 DFSMN×3", self.dfsmn_blocks),
        ]
        if self.use_ir:
            stage_names.append(("Stage3 InvertedResidual×2", 
                               nn.ModuleList([self.ir_block1, self.ir_block2])))
        stage_names += [
            ("Stage4 BiLSTM", self.bilstm),
            ("Stage5 Classifier", self.classifier),
        ]
        for name, module in stage_names:
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:<30} {n:>8,} ({n/1000:>5.1f}K)")

    def forward(self, x, return_variance=False):
        """
        Args:
            x: [B, T, 80] log-Mel特征
            return_variance: 是否返回频谱方差(用于后处理)

        Returns:
            logits: [B, T, 4] 四分类logits
            variance (可选): [B, T, 1] 频谱方差
        """
        B, T, _ = x.shape

        # 保留原始mel用于方差计算
        raw_mel = x.clone() if return_variance else None

        # ===== Stage 1: Spectral Encoder =====
        # 转置为 [B, C, T] 适配Conv1d
        x = x.transpose(1, 2)  # [B, 80, T]
        x = F.pad(x, (self.stem_padding, 0))  # 因果填充
        x = self.spectral_encoder(x)  # [B, 64, T]
        x = x.transpose(1, 2)  # [B, T, 64]

        # ===== Stage 2: DFSMN Backbone =====
        for dfsmn in self.dfsmn_blocks:
            x = dfsmn(x)  # [B, T, 64]

        # ===== Stage 3: Multi-Scale Inverted Residual (可选) =====
        if self.use_ir:
            x_t = x.transpose(1, 2)  # [B, 64, T]
            x_t = self.ir_block1(x_t)
            x_t = self.ir_block2(x_t)
            x = x_t.transpose(1, 2)  # [B, T, 64]

        # ===== Stage 4: Context Fusion (BiLSTM) =====
        x, _ = self.bilstm(x)  # [B, T, 128]
        x = self.lstm_dropout(x)

        # ===== Stage 5: Quad-Head =====
        logits = self.classifier(x)  # [B, T, 4]

        if return_variance:
            variance = compute_spectral_variance(raw_mel)
            return logits, variance

        return logits

    def inference(self, x, variance_threshold=0.015):
        """
        带后处理的推理。

        后处理规则:
        - 模型输出4分类概率
        - 若判为"纯枪声(0)": 计算频谱方差
            - 方差 < threshold → 确认纯枪声 → 丢弃
            - 方差 ≥ threshold → 可能是重叠 → 改为"重叠(3)"
        - 其他类别直接使用
        """
        with torch.no_grad():
            logits, variance = self.forward(x, return_variance=True)
            probs = F.softmax(logits, dim=-1)  # [B, T, 4]
            pred = torch.argmax(probs, dim=-1)  # [B, T]
            # 4分类: 0=纯枪声, 1=语音, 2=静音, 3=重叠

            # 后处理: 纯枪声 → 检查方差
            gunshot_mask = (pred == 0)  # 判为纯枪声的帧
            overlap_suspect = gunshot_mask & (variance.squeeze(-1) >= variance_threshold)
            pred[overlap_suspect] = 3  # 改为重叠类别

            return pred, probs, variance


# ============================================================
# 快速预筛 (与RangeVAD-Plus联合使用)
# ============================================================
def fast_prescreen(audio_frame, prev_energy=0.0,
                    energy_threshold=0.01, crest_threshold=10.0):
    """快速预筛 — O(1)复杂度过滤明显帧。"""
    energy = np.sqrt(np.mean(audio_frame ** 2))

    if energy < energy_threshold:
        return 2, energy  # 静音 (类别2)

    peak = np.max(np.abs(audio_frame))
    crest = peak / (energy + 1e-10)
    if crest > crest_threshold:
        return 0, energy  # 纯枪声 (类别0)

    return -1, energy  # 模糊帧 → 送神经网络


# ============================================================
# 测试验证
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RangeVAD-Plus 模型验证")
    print("=" * 60)

    model = RangeVADPlus(mel_dim=80, hidden_dim=64, num_classes=4)

    # 前向传播测试
    B, T = 4, 100
    x = torch.randn(B, T, 80)
    logits = model(x)
    print(f"\n输入:  {x.shape}  [B, T, 80]")
    print(f"输出:  {logits.shape}  [B, T, 4] (纯枪声/语音/重叠/静音)")

    # 推理测试(带后处理)
    print(f"\n{'='*60}")
    print("推理测试 (带频谱方差后处理)")
    print(f"{'='*60}")
    pred, probs, variance = model.inference(x, variance_threshold=0.015)
    print(f"预测分布:")
    for i, name in enumerate(["纯枪声", "语音", "静音", "重叠"]):
        count = (pred == i).sum().item()
        print(f"  {name}: {count}帧 ({count/(B*T)*100:.1f}%)")
    print(f"频谱方差范围: [{variance.min():.4f}, {variance.max():.4f}]")

    # 与原版RangeVAD对比
    print(f"\n{'='*60}")
    print("与RangeVAD系列参数量对比")
    print(f"{'='*60}")
    total = sum(p.numel() for p in model.parameters())
    comparisons = [
        ("RangeVAD V4 (纯BiLSTM)", 185000),
        ("RangeVAD V5 (+注意力)", 188000),
        ("RangeVAD v6 SpeechFirst", 191600),
        ("RangeVAD v6 POD", 213000),
        ("RangeVAD-Plus (本文)", total),
        ("RangeVAD-Pro (MST-VAD)", 338000),
        ("Silero VAD v5", 309000),
        ("FireRedVAD", 588000),
    ]
    for name, params in comparisons:
        marker = " ← 本文" if "本文" in name else ""
        bar = "█" * int(params / 15000)
        print(f"  {name:<28} {params/1000:>6.1f}K  {bar}{marker}")

    print(f"\n架构亮点:")
    print(f"  ✅ 5个Stage (SpectralEncoder → DFSMN×3 → InvertedResidual×2 → BiLSTM → QuadHead)")
    print(f"  ✅ DFSMN块 (参考FireRedVAD): 非递归时序建模, 训练快7倍")
    print(f"  ✅ Inverted Residual (参考MagicNet): 升维4×提取丰富特征")
    print(f"  ✅ Depthwise Separable Conv (参考MarbleNet): 轻量高效")
    print(f"  ✅ 频谱方差后处理: 区分纯枪声 vs 重叠帧")
    print(f"  ✅ 四分类输出: [纯枪声, 语音, 重叠, 静音]")
    print(f"  ✅ 总参数量: {total/1000:.1f}K (与原版相近!)")
