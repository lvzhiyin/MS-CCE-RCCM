# #!/usr/bin/env python3
# # -*- encoding: utf-8 -*-
# """
# LightSACM - 轻量级 ASR 纠错模型 (基于论文第四章 SACM 内核)

# 配置:
# - 词表大小: 3000
# - d_model: 128
# - 注意力头数: 4
# - Encoder/Decoder 层数: 3
# - 最大序列长度: 30
# - 前馈维度: 512
# - 总参数量: ~2.0M

# 用途:
# - 非流式整句纠错
# - 输入: ASR 识别文本（可能含错误）
# - 输出: 纠正后的干净文本
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class PositionalEncoding(nn.Module):
#     """位置编码"""

#     def __init__(self, d_model, max_len=30, dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: [B, T, D]
#         Returns:
#             x + positional_encoding
#         """
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)


# class LightSACM(nn.Module):
#     """
#     轻量级 SACM 纠错模型

#     非自回归 Transformer 编解码器架构:
#     - Encoder: 编码 ASR 错误文本
#     - Decoder: 生成纠正文本
#     """

#     def __init__(
#         self,
#         vocab_size=3000,
#         d_model=128,
#         nhead=4,
#         num_encoder_layers=3,
#         num_decoder_layers=3,
#         dim_feedforward=512,
#         max_seq_len=30,
#         dropout=0.1,
#         pad_idx=0,
#     ):
#         super().__init__()

#         self.vocab_size = vocab_size
#         self.d_model = d_model
#         self.pad_idx = pad_idx
#         self.max_seq_len = max_seq_len

#         # Embedding 层
#         self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
#         self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

#         # Transformer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#         )
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

#         # 输出层
#         self.output_proj = nn.Linear(d_model, vocab_size)

#         # 初始化权重
#         self._init_weights()

#         # 打印参数信息
#         total_params = sum(p.numel() for p in self.parameters())
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print(f"[LightSACM] 模型初始化完成:")
#         print(f"  - 词表大小: {vocab_size}")
#         print(f"  - d_model: {d_model}")
#         print(f"  - 注意力头: {nhead}")
#         print(f"  - Encoder层数: {num_encoder_layers}")
#         print(f"  - Decoder层数: {num_decoder_layers}")
#         print(f"  - 最大序列长度: {max_seq_len}")
#         print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
#         print(f"  - 可训练参数: {trainable_params:,}")

#     def _init_weights(self):
#         """初始化权重"""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def create_mask(self, src, tgt):
#         """创建 mask"""
#         src_key_padding_mask = (src == self.pad_idx)
#         tgt_key_padding_mask = (tgt == self.pad_idx)

#         # Decoder causal mask
#         tgt_len = tgt.size(1)
#         causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()

#         return src_key_padding_mask, tgt_key_padding_mask, causal_mask

#     def forward(self, src, tgt):
#         """
#         前向传播

#         Args:
#             src: [B, S] 源序列 (ASR 错误文本)
#             tgt: [B, T] 目标序列 (正确文本，带 <sos> 标记)
#         Returns:
#             logits: [B, T, V] 输出 logits
#         """
#         # Embedding
#         src_emb = self.embedding(src)  # [B, S, D]
#         tgt_emb = self.embedding(tgt)  # [B, T, D]

#         # 位置编码
#         src_emb = self.pos_encoding(src_emb)
#         tgt_emb = self.pos_encoding(tgt_emb)

#         # 创建 mask
#         src_pad_mask, tgt_pad_mask, causal_mask = self.create_mask(src, tgt)

#         # Encoder
#         memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)  # [B, S, D]

#         # Decoder
#         dec_out = self.decoder(
#             tgt_emb,
#             memory,
#             tgt_mask=causal_mask,
#             tgt_key_padding_mask=tgt_pad_mask,
#             memory_key_padding_mask=src_pad_mask,
#         )  # [B, T, D]

#         # 输出投影
#         logits = self.output_proj(dec_out)  # [B, T, V]

#         return logits

#     def generate(self, src, tokenizer, max_len=None, device='cuda'):
#         """
#         推理生成（非自回归：直接输出完整序列）

#         Args:
#             src: [B, S] 或 [S] 源序列
#             tokenizer: 分词器
#             max_len: 最大生成长度
#             device: 设备
#         Returns:
#             output_ids: [B, T] 生成的 token IDs
#         """
#         if max_len is None:
#             max_len = self.max_seq_len

#         if src.dim() == 1:
#             src = src.unsqueeze(0)

#         B = src.size(0)
#         device = src.device

#         # 计算实际源序列长度（去掉 padding）
#         actual_src_len = (src != self.pad_idx).sum(dim=1).max().item()

#         # Encoder
#         src_emb = self.embedding(src)
#         src_emb = self.pos_encoding(src_emb)
#         src_pad_mask = (src == self.pad_idx)
#         memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

#         # 初始化目标输入为 <sos>
#         sos_id = tokenizer.sos_token_id if hasattr(tokenizer, 'sos_token_id') else 1
#         eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2

#         # 非自回归：直接用 <sos> 填充整个目标序列
#         tgt = torch.full((B, max_len), sos_id, dtype=torch.long, device=device)

#         # Decoder
#         tgt_emb = self.embedding(tgt)
#         tgt_emb = self.pos_encoding(tgt_emb)

#         tgt_pad_mask = (tgt == self.pad_idx)
#         causal_mask = torch.triu(torch.ones(max_len, max_len, device=device), diagonal=1).bool()

#         dec_out = self.decoder(
#             tgt_emb,
#             memory,
#             tgt_mask=causal_mask,
#             tgt_key_padding_mask=tgt_pad_mask,
#             memory_key_padding_mask=None,
#         )

#         # 输出投影
#         logits = self.output_proj(dec_out)  # [B, max_len, V]

#         # Greedy decoding
#         output_ids = logits.argmax(dim=-1)  # [B, max_len]

#         return output_ids


# class CharTokenizer:
#     """字符级分词器"""

#     def __init__(self, vocab_file=None, max_vocab=3000):
#         self.pad_token = '<PAD>'
#         self.unk_token = '<UNK>'
#         self.sos_token = '<SOS>'
#         self.eos_token = '<EOS>'

#         self.pad_token_id = 0
#         self.unk_token_id = 1
#         self.sos_token_id = 2
#         self.eos_token_id = 3

#         self.token2id = {
#             self.pad_token: self.pad_token_id,
#             self.unk_token: self.unk_token_id,
#             self.sos_token: self.sos_token_id,
#             self.eos_token: self.eos_token_id,
#         }
#         self.id2token = {v: k for k, v in self.token2id.items()}

#         if vocab_file is not None:
#             self.load_vocab(vocab_file, max_vocab)

#     def load_vocab(self, texts, max_vocab=3000):
#         """从文本构建词表"""
#         char_freq = {}
#         for text in texts:
#             for char in text:
#                 char_freq[char] = char_freq.get(char, 0) + 1

#         # 按频率排序，取 top N
#         sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab - 4]

#         idx = len(self.token2id)
#         for char, _ in sorted_chars:
#             if char not in self.token2id:
#                 self.token2id[char] = idx
#                 self.id2token[idx] = char
#                 idx += 1

#         print(f"[CharTokenizer] 词表大小: {len(self.token2id)}")

#     def encode(self, text, add_sos=False, add_eos=False):
#         """编码文本"""
#         ids = []
#         if add_sos:
#             ids.append(self.sos_token_id)

#         for char in text:
#             ids.append(self.token2id.get(char, self.unk_token_id))

#         if add_eos:
#             ids.append(self.eos_token_id)

#         return ids

#     def decode(self, ids, remove_special=True):
#         """解码 ID 序列"""
#         chars = []
#         for id_ in ids:
#             if remove_special and id_ in [self.pad_token_id, self.unk_token_id, self.sos_token_id, self.eos_token_id]:
#                 continue
#             token = self.id2token.get(id_, self.unk_token)
#             chars.append(token)
#         return ''.join(chars)


# def test_light_rccm():
#     """测试模型"""
#     print("=" * 60)
#     print("LightSACM 测试")
#     print("=" * 60)

#     # 参数配置
#     config = {
#         'vocab_size': 3000,
#         'd_model': 128,
#         'nhead': 4,
#         'num_encoder_layers': 3,
#         'num_decoder_layers': 3,
#         'dim_feedforward': 512,
#         'max_seq_len': 30,
#         'dropout': 0.1,
#     }

#     # 创建模型
#     model = LightSACM(**config)

#     # 测试输入
#     B, S, T = 2, 15, 20
#     src = torch.randint(0, 3000, (B, S))
#     tgt = torch.randint(0, 3000, (B, T))

#     # 前向传播
#     model.eval()
#     with torch.no_grad():
#         logits = model(src, tgt)

#     print(f"\n输入形状:")
#     print(f"  src: {src.shape}")
#     print(f"  tgt: {tgt.shape}")
#     print(f"输出形状:")
#     print(f"  logits: {logits.shape}")

#     assert logits.shape == (B, T, 3000), f"输出形状错误: {logits.shape}"

#     print("\n✅ 测试通过!")


# if __name__ == "__main__":
#     test_light_rccm()








# !/usr/bin/env python3
# -*- encoding: utf-8 -*-
# """
# LightSACM - 轻量级 ASR 纠错模型 (基于论文第四章 SACM 内核)

# 配置:
# - 词表大小: 3000
# - d_model: 128
# - 注意力头数: 4
# - Encoder/Decoder 层数: 3
# - 最大序列长度: 30
# - 前馈维度: 512
# - 总参数量: ~2.0M

# 用途:
# - 非流式整句纠错
# - 输入: ASR 识别文本（可能含错误）
# - 输出: 纠正后的干净文本
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class PositionalEncoding(nn.Module):
#     """位置编码"""

#     def __init__(self, d_model, max_len=30, dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: [B, T, D]
#         Returns:
#             x + positional_encoding
#         """
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)


# class LightSACM(nn.Module):
#     """
#     轻量级 SACM 纠错模型

#     非自回归 Transformer 编解码器架构:
#     - Encoder: 编码 ASR 错误文本
#     - Decoder: 生成纠正文本
#     """

#     def __init__(
#         self,
#         vocab_size=3000,
#         d_model=128,
#         nhead=4,
#         num_encoder_layers=3,
#         num_decoder_layers=3,
#         dim_feedforward=512,
#         max_seq_len=30,
#         dropout=0.1,
#         pad_idx=0,
#     ):
#         super().__init__()

#         self.vocab_size = vocab_size
#         self.d_model = d_model
#         self.pad_idx = pad_idx
#         self.max_seq_len = max_seq_len

#         # Embedding 层
#         self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
#         self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

#         # Transformer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#         )
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

#         # 输出层
#         self.output_proj = nn.Linear(d_model, vocab_size)

#         # 初始化权重
#         self._init_weights()

#         # 打印参数信息
#         total_params = sum(p.numel() for p in self.parameters())
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print(f"[LightSACM] 模型初始化完成:")
#         print(f"  - 词表大小: {vocab_size}")
#         print(f"  - d_model: {d_model}")
#         print(f"  - 注意力头: {nhead}")
#         print(f"  - Encoder层数: {num_encoder_layers}")
#         print(f"  - Decoder层数: {num_decoder_layers}")
#         print(f"  - 最大序列长度: {max_seq_len}")
#         print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
#         print(f"  - 可训练参数: {trainable_params:,}")

#     def _init_weights(self):
#         """初始化权重"""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def create_mask(self, src, tgt):
#         """创建 mask"""
#         src_key_padding_mask = (src == self.pad_idx)
#         tgt_key_padding_mask = (tgt == self.pad_idx)

#         # Decoder causal mask
#         tgt_len = tgt.size(1)
#         causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()

#         return src_key_padding_mask, tgt_key_padding_mask, causal_mask

#     def forward(self, src, tgt):
#         """
#         前向传播

#         Args:
#             src: [B, S] 源序列 (ASR 错误文本)
#             tgt: [B, T] 目标序列 (正确文本，带 <sos> 标记)
#         Returns:
#             logits: [B, T, V] 输出 logits
#         """
#         # Embedding
#         src_emb = self.embedding(src)  # [B, S, D]
#         tgt_emb = self.embedding(tgt)  # [B, T, D]

#         # 位置编码
#         src_emb = self.pos_encoding(src_emb)
#         tgt_emb = self.pos_encoding(tgt_emb)

#         # 创建 mask
#         src_pad_mask, tgt_pad_mask, causal_mask = self.create_mask(src, tgt)

#         # Encoder
#         memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)  # [B, S, D]

#         # Decoder
#         dec_out = self.decoder(
#             tgt_emb,
#             memory,
#             tgt_mask=causal_mask,
#             tgt_key_padding_mask=tgt_pad_mask,
#             memory_key_padding_mask=src_pad_mask,
#         )  # [B, T, D]

#         # 输出投影
#         logits = self.output_proj(dec_out)  # [B, T, V]

#         return logits

#     def generate(self, src, tokenizer, max_len=None, device='cuda'):
#         """
#         推理生成（自回归：逐 token 预测）

#         Args:
#             src: [B, S] 或 [S] 源序列（错误文本）
#             tokenizer: 分词器
#             max_len: 最大生成长度
#             device: 设备
#         Returns:
#             output_ids: [B, T] 生成的 token IDs
#         """
#         if max_len is None:
#             max_len = self.max_seq_len

#         if src.dim() == 1:
#             src = src.unsqueeze(0)

#         B = src.size(0)
#         device = src.device

#         # Encoder（只跑一次）
#         src_emb = self.embedding(src)
#         src_emb = self.pos_encoding(src_emb)
#         src_pad_mask = (src == self.pad_idx)
#         memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

#         # 解码起点
#         sos_id = tokenizer.sos_token_id if hasattr(tokenizer, 'sos_token_id') else 1
#         eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2

#         # === 自回归逐步生成 ===
#         generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)

#         for step in range(max_len - 1):
#             tgt = generated  # [B, current_len]

#             # Embedding + 位置编码
#             tgt_emb = self.embedding(tgt)
#             tgt_emb = self.pos_encoding(tgt_emb)

#             # Causal mask（只看前面的 token）
#             tgt_len = tgt.size(1)
#             causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()

#             # Decoder
#             dec_out = self.decoder(
#                 tgt_emb, memory,
#                 tgt_mask=causal_mask,
#                 memory_key_padding_mask=None,
#             )

#             # 只取最后一个位置的 logits
#             logits = self.output_proj(dec_out[:, -1:, :])  # [B, 1, V]
#             next_token = logits.argmax(dim=-1)  # [B, 1]

#             # 拼接到生成序列
#             generated = torch.cat([generated, next_token], dim=1)

#             # 全部遇到 EOS 就提前结束
#             if (next_token == eos_id).all():
#                 break

#         return generated


# class CharTokenizer:
#     """字符级分词器"""

#     def __init__(self, vocab_file=None, max_vocab=3000):
#         self.pad_token = '<PAD>'
#         self.unk_token = '<UNK>'
#         self.sos_token = '<SOS>'
#         self.eos_token = '<EOS>'

#         self.pad_token_id = 0
#         self.unk_token_id = 1
#         self.sos_token_id = 2
#         self.eos_token_id = 3

#         self.token2id = {
#             self.pad_token: self.pad_token_id,
#             self.unk_token: self.unk_token_id,
#             self.sos_token: self.sos_token_id,
#             self.eos_token: self.eos_token_id,
#         }
#         self.id2token = {v: k for k, v in self.token2id.items()}

#         if vocab_file is not None:
#             self.load_vocab(vocab_file, max_vocab)

#     def load_vocab(self, texts, max_vocab=3000):
#         """从文本构建词表"""
#         char_freq = {}
#         for text in texts:
#             for char in text:
#                 char_freq[char] = char_freq.get(char, 0) + 1

#         # 按频率排序，取 top N
#         sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab - 4]

#         idx = len(self.token2id)
#         for char, _ in sorted_chars:
#             if char not in self.token2id:
#                 self.token2id[char] = idx
#                 self.id2token[idx] = char
#                 idx += 1

#         print(f"[CharTokenizer] 词表大小: {len(self.token2id)}")

#     def encode(self, text, add_sos=False, add_eos=False):
#         """编码文本"""
#         ids = []
#         if add_sos:
#             ids.append(self.sos_token_id)

#         for char in text:
#             ids.append(self.token2id.get(char, self.unk_token_id))

#         if add_eos:
#             ids.append(self.eos_token_id)

#         return ids

#     def decode(self, ids, remove_special=True):
#         """解码 ID 序列"""
#         chars = []
#         for id_ in ids:
#             if remove_special and id_ in [self.pad_token_id, self.unk_token_id, self.sos_token_id, self.eos_token_id]:
#                 continue
#             token = self.id2token.get(id_, self.unk_token)
#             chars.append(token)
#         return ''.join(chars)


# def test_light_rccm():
#     """测试模型"""
#     print("=" * 60)
#     print("LightSACM 测试")
#     print("=" * 60)

#     # 参数配置
#     config = {
#         'vocab_size': 3000,
#         'd_model': 128,
#         'nhead': 4,
#         'num_encoder_layers': 3,
#         'num_decoder_layers': 3,
#         'dim_feedforward': 512,
#         'max_seq_len': 30,
#         'dropout': 0.1,
#     }

#     # 创建模型
#     model = LightSACM(**config)

#     # 测试输入
#     B, S, T = 2, 15, 20
#     src = torch.randint(0, 3000, (B, S))
#     tgt = torch.randint(0, 3000, (B, T))

#     # 前向传播
#     model.eval()
#     with torch.no_grad():
#         logits = model(src, tgt)

#     print(f"\n输入形状:")
#     print(f"  src: {src.shape}")
#     print(f"  tgt: {tgt.shape}")
#     print(f"输出形状:")
#     print(f"  logits: {logits.shape}")

#     assert logits.shape == (B, T, 3000), f"输出形状错误: {logits.shape}"

#     print("\n✅ 测试通过!")


# if __name__ == "__main__":
#     test_light_rccm()






















#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
LightSACM - 轻量级 ASR 纠错模型 (基于论文第四章 SACM 内核)

配置:
- 词表大小: 3000
- d_model: 128
- 注意力头数: 4
- Encoder/Decoder 层数: 3
- 最大序列长度: 30
- 前馈维度: 512
- 总参数量: ~2.0M

用途:
- 非流式整句纠错
- 输入: ASR 识别文本（可能含错误）
- 输出: 纠正后的干净文本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=30, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            x + positional_encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============ 约束解码 Trie ============
class CommandTrie:
    def __init__(self, hotwords):
        self.root = {}
        self.start_chars = set()
        for hw in hotwords:
            hw = hw.strip()
            if not hw:
                continue
            self.start_chars.add(hw[0])
            node = self.root
            for ch in hw:
                node = node.setdefault(ch, {})
            node['__END__'] = True
        print(f"[CommandTrie] {len(hotwords)} 条热词, 首字符={sorted(self.start_chars)}")

    def allowed_next_chars(self, prefix):
        """prefix: 已解码字符串 (不含 <SOS>)"""
        if not prefix:
            return list(sorted(self.start_chars))

        # 向前找最近一个完整热词的结束位置
        segment_start = 0
        node = self.root
        for i, ch in enumerate(prefix):
            if ch in node:
                node = node[ch]
                if '__END__' in node:  # 热词完整结束
                    segment_start = i + 1
            else:
                # 不在 Trie 中，允许从当前位置重新开始
                segment_start = i
                break

        cur_seg = prefix[segment_start:]
        # 走 Trie 找到当前片段所在的节点
        node = self.root
        for ch in cur_seg:
            if ch in node:
                node = node[ch]
            else:
                return list(sorted(self.start_chars))

        # 允许的下一字符：当前节点的子节点
        allowed = [ch for ch in node.keys() if ch != '__END__']
        if not allowed:
            # 当前片段已完整匹配某热词，允许跳到下一热词
            allowed = list(sorted(self.start_chars))
        return allowed

    @classmethod
    def from_texts(cls, texts):
        return cls(texts)

    @classmethod
    def from_hotwords(cls, hotwords):
        return cls(hotwords)


class LightSACM(nn.Module):
    """
    轻量级 SACM 纠错模型

    非自回归 Transformer 编解码器架构:
    - Encoder: 编码 ASR 错误文本
    - Decoder: 生成纠正文本
    """

    def __init__(
        self,
        vocab_size=3000,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        max_seq_len=30,
        dropout=0.1,
        pad_idx=0,
        use_error_detector=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        self.use_error_detector = use_error_detector

        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出层
        self.output_proj = nn.Linear(d_model, vocab_size)

        # === 错误检测器 (可选) ===
        if use_error_detector:
            # 4类: 0=正确(K), 1=替换(S), 2=前插(I), 3=删除(D)
            self.error_detector = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 4),
            )
            self.error_embedding = nn.Embedding(4, d_model)

        # 初始化权重
        self._init_weights()

        # 打印参数信息
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LightSACM] 模型初始化完成:")
        print(f"  - 词表大小: {vocab_size}")
        print(f"  - d_model: {d_model}")
        print(f"  - 注意力头: {nhead}")
        print(f"  - Encoder层数: {num_encoder_layers}")
        print(f"  - Decoder层数: {num_decoder_layers}")
        print(f"  - 最大序列长度: {max_seq_len}")
        if use_error_detector:
            print(f"  - 错误检测器: ✓")
        print(f"  - 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  - 可训练参数: {trainable_params:,}")

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_mask(self, src, tgt):
        """创建 mask"""
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        # Decoder causal mask
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()

        return src_key_padding_mask, tgt_key_padding_mask, causal_mask

    def forward(self, src, tgt):
        """
        前向传播

        Args:
            src: [B, S] 源序列 (ASR 错误文本)
            tgt: [B, T] 目标序列 (正确文本，带 <sos> 标记)
        Returns:
            logits: [B, T, V] 输出 logits
            det_logits: [B, S, 4] 或 None (error_detector 辅助输出)
        """
        # Embedding
        src_emb = self.embedding(src)  # [B, S, D]
        tgt_emb = self.embedding(tgt)  # [B, T, D]

        # 位置编码
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)

        # 创建 mask
        src_pad_mask, tgt_pad_mask, causal_mask = self.create_mask(src, tgt)

        # Encoder
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)  # [B, S, D]
        det_logits = None

        # 错误检测器：仅做辅助 loss，不改变 decoder 看到的 memory
        if self.use_error_detector:
            det_logits = self.error_detector(memory)  # [B, S, 4]

        # Decoder — 始终以原始 memory 为 cross-attention 目标
        dec_out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )  # [B, T, D]

        # 输出投影
        logits = self.output_proj(dec_out)  # [B, T, V]

        return logits, det_logits

    def generate(self, src, tokenizer, max_len=None, device='cuda', trie=None):
        """
        推理生成（自回归：逐 token 预测）

        Args:
            src: [B, S] 或 [S] 源序列（错误文本）
            tokenizer: 分词器
            max_len: 最大生成长度
            device: 设备
            trie: 可选 CommandTrie 用于约束解码
        Returns:
            output_ids: [B, T] 生成的 token IDs
        """
        if max_len is None:
            max_len = self.max_seq_len

        if src.dim() == 1:
            src = src.unsqueeze(0)

        B = src.size(0)
        device = src.device

        # Encoder（只跑一次）
        src_emb = self.embedding(src)
        src_emb = self.pos_encoding(src_emb)
        src_pad_mask = (src == self.pad_idx)
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # 解码起点
        sos_id = tokenizer.sos_token_id if hasattr(tokenizer, 'sos_token_id') else 1
        eos_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2

        # === 自回归逐步生成 ===
        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)

        for step in range(max_len - 1):
            tgt = generated  # [B, current_len]

            # Embedding + 位置编码
            tgt_emb = self.embedding(tgt)
            tgt_emb = self.pos_encoding(tgt_emb)

            # Causal mask（只看前面的 token）
            tgt_len = tgt.size(1)
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()

            # Decoder
            dec_out = self.decoder(
                tgt_emb, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=None,
            )

            # 只取最后一个位置的 logits
            logits = self.output_proj(dec_out[:, -1:, :])  # [B, 1, V]

            # === Trie 约束解码 ===
            if trie is not None:
                # 解码当前已生成的字符串
                current_text = tokenizer.decode(generated[0].tolist())
                next_chars = trie.allowed_next_chars(current_text)
                if next_chars and None not in next_chars:
                    # 找出非法 token 的 mask
                    allowed_ids = set()
                    for ch in next_chars:
                        cid = tokenizer.encode(ch, add_sos=False, add_eos=False)
                        allowed_ids.update(cid)
                    # 始终允许 EOS (eos 也作为合法结束)
                    allowed_ids.add(eos_id)
                    # 将所有不允许的 token 设为 -inf
                    mask = torch.ones_like(logits) * float('-inf')
                    for aid in allowed_ids:
                        mask[0, 0, aid] = 0
                    logits = logits + mask

            next_token = logits.argmax(dim=-1)  # [B, 1]

            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=1)

            # 全部遇到 EOS 就提前结束
            if (next_token == eos_id).all():
                break

        return generated


class CharTokenizer:
    """字符级分词器"""

    def __init__(self, vocab_file=None, max_vocab=3000):
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.sos_token_id = 2
        self.eos_token_id = 3

        self.token2id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.sos_token: self.sos_token_id,
            self.eos_token: self.eos_token_id,
        }
        self.id2token = {v: k for k, v in self.token2id.items()}

        if vocab_file is not None:
            self.load_vocab(vocab_file, max_vocab)

    def load_vocab(self, texts, max_vocab=3000):
        """从文本构建词表"""
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1

        # 按频率排序，取 top N
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab - 4]

        idx = len(self.token2id)
        for char, _ in sorted_chars:
            if char not in self.token2id:
                self.token2id[char] = idx
                self.id2token[idx] = char
                idx += 1

        print(f"[CharTokenizer] 词表大小: {len(self.token2id)}")

    def encode(self, text, add_sos=False, add_eos=False):
        """编码文本"""
        ids = []
        if add_sos:
            ids.append(self.sos_token_id)

        for char in text:
            ids.append(self.token2id.get(char, self.unk_token_id))

        if add_eos:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids, remove_special=True):
        """解码 ID 序列"""
        chars = []
        for id_ in ids:
            if remove_special and id_ in [self.pad_token_id, self.unk_token_id, self.sos_token_id, self.eos_token_id]:
                continue
            token = self.id2token.get(id_, self.unk_token)
            chars.append(token)
        return ''.join(chars)


def test_light_rccm():
    """测试模型"""
    print("=" * 60)
    print("LightSACM 测试")
    print("=" * 60)

    # 参数配置
    config = {
        'vocab_size': 3000,
        'd_model': 128,
        'nhead': 4,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 512,
        'max_seq_len': 30,
        'dropout': 0.1,
    }

    # 创建模型
    model = LightSACM(**config)

    # 测试输入
    B, S, T = 2, 15, 20
    src = torch.randint(0, 3000, (B, S))
    tgt = torch.randint(0, 3000, (B, T))

    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt)

    print(f"\n输入形状:")
    print(f"  src: {src.shape}")
    print(f"  tgt: {tgt.shape}")
    print(f"输出形状:")
    print(f"  logits: {logits.shape}")

    assert logits.shape == (B, T, 3000), f"输出形状错误: {logits.shape}"

    print("\n✅ 测试通过!")


if __name__ == "__main__":
    test_light_rccm()