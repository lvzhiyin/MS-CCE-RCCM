# #!/usr/bin/env python3
# """
# RangeVAD V5 — 频域注意力 + 事件驱动预筛

# 改进:
#   1. 频域注意力 (Frequency Attention): 
#      借鉴陈家琦SAPVAD的谱注意力思想，在BiLSTM之前
#      增加轻量级频域注意力，让模型显式学习"哪些频带
#      对区分枪声/语音更关键"。
#      增加参数量: ~3.2K

#   2. 事件驱动预筛 (Event-Driven Pre-screening):
#      借鉴赵天昊HNANNs的异步特征提取思想，在BiLSTM
#      推理之前用能量/峰值因子做快速预判：
#      - 静音→直接返回(跳过BiLSTM)
#      - 枪声脉冲→直接返回(跳过BiLSTM)  
#      - 其他→进入BiLSTM精细判别
#      预期: 80%的帧跳过BiLSTM，推理加速5倍
# """

# import os, sys, json, random, argparse, logging
# from pathlib import Path
# import numpy as np
# import torch, torch.nn as nn, torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import librosa
# import soundfile as sf

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
# logger = logging.getLogger('RangeVAD-v5')

# # ============ 标签 ============
# CLASS_GUNSHOT = 0
# CLASS_SPEECH  = 1
# CLASS_SILENCE = 2
# NUM_CLASSES   = 3
# SR = 16000
# HOP = 160
# WIN = 400


# # ============ 特征提取 ============
# def extract_logmel(audio, sr=16000, n_mels=80):
#     if len(audio.shape) > 1: audio = audio.mean(axis=1)
#     mel = librosa.feature.melspectrogram(
#         y=audio.astype(np.float32), sr=sr, n_mels=n_mels,
#         n_fft=400, hop_length=160, fmin=80, fmax=7600, power=2
#     )
#     return np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)


# # ============ 改进1: 频域注意力 ============
# class FrequencyAttention(nn.Module):
#     """轻量级频域注意力模块
    
#     思路: 枪声为全频带脉冲(所有mel频带同时高能)，
#           语音为谐波结构(特定频带有共振峰)。
#           该模块学习一个频带权重向量，强化差异化的频带。
    
#     参数量: 80*20 + 20*80 = 3,200 (仅约3.2K)
#     """
#     def __init__(self, mel_dim=80, reduction=4):
#         super().__init__()
#         hidden = max(mel_dim // reduction, 8)
#         self.attn = nn.Sequential(
#             nn.Linear(mel_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, mel_dim),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         """x: [B, T, D]"""
#         # 全局频带统计量 (batch平均+时间平均)
#         global_freq = x.mean(dim=1).mean(dim=0, keepdim=True)  # [1, D]
#         freq_weights = self.attn(global_freq)  # [1, D]
#         return x * freq_weights.unsqueeze(0)  # [B, T, D] 按频带加权


# # ============ 改进2: 事件驱动预筛 ============
# class EventDrivenPreScreener:
#     """事件驱动快速预筛
    
#     三层级联:
#     Layer 1: 能量检测 (O(1)) → 过滤静音
#     Layer 2: 峰值因子检测 (O(1)) → 过滤明显枪声
#     Layer 3: BiLSTM推理 → 精细判别
#     """
#     def __init__(self, energy_threshold=0.01, crest_threshold=8.0):
#         self.energy_threshold = energy_threshold
#         self.crest_threshold = crest_threshold
#         self.stats = {'total': 0, 'l1_silence': 0, 'l2_gunshot': 0, 'l3_bilstm': 0}
    
#     def screen(self, audio_frame):
#         """返回: (label, need_bilstm)
#         label=None 表示需要BiLSTM推理
#         """
#         self.stats['total'] += 1
        
#         # Layer 1: 能量检测
#         rms = np.sqrt(np.mean(audio_frame ** 2))
#         max_rms = max(rms, 1e-10)
        
#         if rms < self.energy_threshold * 0.1:  # 极低能量=静音
#             self.stats['l1_silence'] += 1
#             return CLASS_SILENCE, False
        
#         # Layer 2: 峰值因子检测
#         peak = np.max(np.abs(audio_frame))
#         crest = peak / max_rms
#         if crest > self.crest_threshold:  # 高峰值因子=脉冲(枪声)
#             self.stats['l2_gunshot'] += 1
#             return CLASS_GUNSHOT, False
        
#         # Layer 3: 需要BiLSTM精细推理
#         self.stats['l3_bilstm'] += 1
#         return None, True
    
#     def report(self):
#         t = max(self.stats['total'], 1)
#         logger.info(f"预筛统计: 总计{self.stats['total']}帧, "
#                     f"L1静音={self.stats['l1_silence']}({self.stats['l1_silence']/t*100:.1f}%), "
#                     f"L2枪声={self.stats['l2_gunshot']}({self.stats['l2_gunshot']/t*100:.1f}%), "
#                     f"L3推理={self.stats['l3_bilstm']}({self.stats['l3_bilstm']/t*100:.1f}%)")


# # ============ 模型: BiLSTM + 频域注意力 ============
# class BiLSTMVAD_v5(nn.Module):
#     """RangeVAD V5: BiLSTM + 频域注意力"""
#     def __init__(self, input_dim=80, hidden_dim=64, num_layers=2, dropout=0.2,
#                  use_freq_attn=True):
#         super().__init__()
#         self.use_freq_attn = use_freq_attn
        
#         if use_freq_attn:
#             self.freq_attn = FrequencyAttention(mel_dim=input_dim, reduction=4)
        
#         self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
#                               bidirectional=True, batch_first=True, dropout=dropout)
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(64, NUM_CLASSES),
#         )
#         self._total = sum(p.numel() for p in self.parameters())
#         if use_freq_attn:
#             self._attn_params = sum(p.numel() for p in self.freq_attn.parameters())
#             logger.info(f"频域注意力: {self._attn_params:,} params")
    
#     def forward(self, x):
#         if self.use_freq_attn:
#             x = self.freq_attn(x)
#         x, _ = self.bilstm(x)
#         return self.classifier(x)


# # ============ 数据集 ============
# class VAD4Dataset(Dataset):
#     def __init__(self, manifest_path, max_frames=600):
#         with open(manifest_path) as f:
#             self.manifest = json.load(f)
#         self.data = []
#         for item in self.manifest:
#             audio, _ = sf.read(item['audio'])
#             feat = extract_logmel(audio)
#             labels = np.load(item['label'])
#             T = min(feat.shape[0], len(labels), max_frames)
#             if T < 10: continue
#             self.data.append({
#                 'feat': torch.FloatTensor(feat[:T]),
#                 'label': torch.LongTensor(labels[:T]),
#                 'length': T,
#                 'type': item.get('type', 'unknown'),
#             })
#         logger.info(f"加载 {manifest_path}: {len(self.data)} 样本 "
#                     f"({sum(1 for d in self.data if d['type']=='concat')}拼接, "
#                     f"{sum(1 for d in self.data if d['type']=='overlap')}重叠)")
#     def __len__(self): return len(self.data)
#     def __getitem__(self, i): return self.data[i]


# def collate_fn(batch):
#     max_len = max(b['feat'].shape[0] for b in batch)
#     B, D = len(batch), batch[0]['feat'].shape[1]
#     feat_pad = torch.zeros(B, max_len, D)
#     label_pad = torch.full((B, max_len), -100, dtype=torch.long)
#     mask = torch.zeros(B, max_len, dtype=torch.bool)
#     for i, b in enumerate(batch):
#         L = b['length']
#         feat_pad[i, :L] = b['feat']
#         label_pad[i, :L] = b['label']
#         mask[i, :L] = True
#     return feat_pad, label_pad, mask


# # ============ 训练 ============
# def train(args):
#     device = torch.device(args.device)
#     data_dir = Path(args.data_dir)
    
#     train_ds = VAD4Dataset(str(data_dir / 'train_manifest.json'), max_frames=args.max_frames)
#     val_ds = VAD4Dataset(str(data_dir / 'val_manifest.json'), max_frames=args.max_frames)
    
#     tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
#                            collate_fn=collate_fn, num_workers=0)
#     vl_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
#                            collate_fn=collate_fn, num_workers=0)
    
#     use_freq_attn = not args.no_freq_attn
#     model = BiLSTMVAD_v5(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
#                           dropout=args.dropout, use_freq_attn=use_freq_attn).to(device)
#     logger.info(f"模型参数: {model._total:,} (频域注意力={'ON' if use_freq_attn else 'OFF'})")
    
#     ce = nn.CrossEntropyLoss(ignore_index=-100,
#                              weight=torch.tensor([args.gun_weight, 1.0, 0.5], device=device))
#     opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    
#     od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
#     best_acc = 0
    
#     for ep in range(1, args.epochs + 1):
#         model.train()
#         tl, tok, tcn = 0, 0, 0
#         for feat, label, mask in tr_loader:
#             feat, label, mask = feat.to(device), label.to(device), mask.to(device)
#             logits = model(feat)
#             loss = ce(logits[mask], label[mask])
#             opt.zero_grad(); loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
#             preds = logits.argmax(-1)
#             tok += (preds[mask] == label[mask]).sum().item()
#             tcn += mask.sum().item()
#             tl += loss.item()
#         sch.step()
        
#         model.eval()
#         va, vc = 0, 0
#         with torch.no_grad():
#             for feat, label, mask in vl_loader:
#                 feat, label, mask = feat.to(device), label.to(device), mask.to(device)
#                 logits = model(feat)
#                 preds = logits.argmax(-1)
#                 va += (preds[mask] == label[mask]).sum().item()
#                 vc += mask.sum().item()
#         vac = va / max(vc, 1)
#         logger.info(f"Ep {ep:2d}: loss={tl/len(tr_loader):.4f} val_acc={vac:.4f}")
        
#         if vac > best_acc:
#             best_acc = vac
#             torch.save({'model_state_dict': model.state_dict(),
#                         'config': {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers,
#                                    'dropout': args.dropout, 'input_dim': 80,
#                                    'use_freq_attn': use_freq_attn},
#                        }, od / 'model.pt.best')
    
#     logger.info(f"最佳验证准确率: {best_acc:.4f}")
#     return model


# # ============ 评估 (含事件驱动预筛) ============
# def evaluate(args):
#     device = torch.device(args.device)
#     ck = torch.load(args.model_path or f'{args.output_dir}/model.pt.best',
#                     map_location=device, weights_only=False)
#     cfg = ck['config']
    
#     model = BiLSTMVAD_v5(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
#                           dropout=cfg['dropout'], 
#                           use_freq_attn=cfg.get('use_freq_attn', True)).to(device)
#     model.load_state_dict(ck['model_state_dict'])
#     model.eval()
#     logger.info(f"模型参数: {model._total:,}")
    
#     # 创建预筛器
#     screener = EventDrivenPreScreener(energy_threshold=0.01, crest_threshold=8.0)
    
#     data_dir = Path(args.data_dir)
    
#     for test_name in ['testA', 'testB']:
#         manifest_path = data_dir / f'{test_name}_manifest.json'
#         if not manifest_path.exists(): continue
        
#         with open(manifest_path) as f:
#             manifest = json.load(f)
        
#         print(f"\n{'='*70}")
#         print(f"{test_name} 评估 (新枪型)")
#         print(f"{'='*70}")
        
#         for subset_type in ['concat', 'overlap', 'all']:
#             subset = [m for m in manifest 
#                       if subset_type == 'all' or m['type'] == subset_type]
#             if not subset: continue
            
#             total_gun, gun_correct = 0, 0
#             total_spk, spk_correct = 0, 0
#             total_sil, sil_correct = 0, 0
#             gun2speech = 0
#             events_total = {'l1_sil': 0, 'l2_gun': 0, 'l3': 0}
            
#             for item in subset:
#                 try:
#                     audio, _ = sf.read(item['audio'])
#                 except: continue
#                 feat = extract_logmel(audio)
#                 labels = np.load(item['label'])
#                 T = min(len(feat), len(labels))
#                 if T < 10: continue
                
#                 # Step 1: 全序列BiLSTM推理 (保持时序上下文)
#                 feat_t = torch.FloatTensor(feat[:T]).unsqueeze(0).to(device)
#                 with torch.no_grad():
#                     logits = model(feat_t)
#                 bilstm_preds = logits[0].argmax(-1).cpu().numpy()
                
#                 # Step 2: 事件驱动预筛 → 对高置信度帧进行后置修正
#                 preds = bilstm_preds.copy()
                
#                 for t in range(T):
#                     audio_start = t * HOP
#                     audio_end = min(audio_start + 500, len(audio))
#                     audio_frame = audio[audio_start:audio_end]
                    
#                     quick_label, need_bilstm = screener.screen(audio_frame)
                    
#                     if not need_bilstm:
#                         # 预筛器高置信度 → 覆盖BiLSTM结果
#                         preds[t] = quick_label
#                         if quick_label == CLASS_SILENCE:
#                             events_total['l1_sil'] += 1
#                         else:
#                             events_total['l2_gun'] += 1
#                     else:
#                         events_total['l3'] += 1
                
#                 mask_gun = labels[:T] == CLASS_GUNSHOT
#                 mask_spk = labels[:T] == CLASS_SPEECH
#                 mask_sil = labels[:T] == CLASS_SILENCE
                
#                 total_gun += mask_gun.sum()
#                 total_spk += mask_spk.sum()
#                 total_sil += mask_sil.sum()
#                 gun_correct += (preds[mask_gun] == CLASS_GUNSHOT).sum()
#                 spk_correct += (preds[mask_spk] == CLASS_SPEECH).sum()
#                 sil_correct += (preds[mask_sil] == CLASS_SILENCE).sum()
#                 gun2speech += (preds[mask_gun] == CLASS_SPEECH).sum()
            
#             GFAR = gun2speech / max(total_gun, 1)
#             SDR = spk_correct / max(total_spk, 1)
#             FAR = gun2speech / max(total_gun + total_sil, 1)
            
#             print(f"\n  [{subset_type}] n={len(subset)}")
#             print(f"    GFAR={GFAR*100:.2f}%  SDR={SDR*100:.2f}%  FAR={FAR*100:.2f}%")
            
#             total_events = sum(events_total.values())
#             if total_events > 0:
#                 print(f"    预筛: 静音覆盖{events_total['l1_sil']/total_events*100:.1f}% "
#                       f"枪声覆盖{events_total['l2_gun']/total_events*100:.1f}% "
#                       f"推理{events_total['l3']/total_events*100:.1f}%")


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--mode', default='train', choices=['train', 'eval'])
#     p.add_argument('--data-dir', default='/home/lvzhiyin/桌面/FunASR-main/rccm/vad_data_v4')
#     p.add_argument('--output-dir', default='./vad5_output')
#     p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
#     p.add_argument('--hidden-dim', type=int, default=64)
#     p.add_argument('--num-layers', type=int, default=2)
#     p.add_argument('--dropout', type=float, default=0.2)
#     p.add_argument('--max-frames', type=int, default=600)
#     p.add_argument('--epochs', type=int, default=20)
#     p.add_argument('--batch-size', type=int, default=32)
#     p.add_argument('--lr', type=float, default=0.001)
#     p.add_argument('--gun-weight', type=float, default=3.0)
#     p.add_argument('--model-path', default=None)
#     p.add_argument('--no-freq-attn', action='store_true', 
#                    help='消融: 禁用频域注意力 (训练纯BiLSTM作为对照)')
#     args = p.parse_args()
    
#     if args.mode == 'train':
#         train(args)
#     else:
#         evaluate(args)


# if __name__ == '__main__':
#     main()






#!/usr/bin/env python3
"""
RangeVAD V5 — 频域注意力 + 事件驱动预筛

改进:
  1. 频域注意力 (Frequency Attention): 
     借鉴陈家琦SAPVAD的谱注意力思想，在BiLSTM之前
     增加轻量级频域注意力，让模型显式学习"哪些频带
     对区分枪声/语音更关键"。
     增加参数量: ~3.2K

  2. 事件驱动预筛 (Event-Driven Pre-screening):
     借鉴赵天昊HNANNs的异步特征提取思想，在BiLSTM
     推理之前用能量/峰值因子做快速预判：
     - 静音→直接返回(跳过BiLSTM)
     - 枪声脉冲→直接返回(跳过BiLSTM)  
     - 其他→进入BiLSTM精细判别
     预期: 80%的帧跳过BiLSTM，推理加速5倍
"""

import os, sys, json, random, argparse, logging
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('RangeVAD-v5')

# ============ 标签 ============
CLASS_GUNSHOT      = 0   # 拼接纯枪声 → 切掉
CLASS_SPEECH       = 1   # 语音 → 保留
CLASS_SILENCE      = 2   # 静音 → 切掉
CLASS_OVERLAP_GUN  = 3   # 重叠枪声(语音+枪声混合) → 保留
NUM_CLASSES        = 4
SR = 16000
HOP = 160
WIN = 400


# ============ 特征提取 ============
def extract_logmel(audio, sr=16000, n_mels=80):
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=sr, n_mels=n_mels,
        n_fft=400, hop_length=160, fmin=80, fmax=7600, power=2
    )
    return np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)


# ============ 改进1: 频域注意力 ============
class FrequencyAttention(nn.Module):
    """轻量级频域注意力模块
    
    思路: 枪声为全频带脉冲(所有mel频带同时高能)，
          语音为谐波结构(特定频带有共振峰)。
          该模块学习一个频带权重向量，强化差异化的频带。
    
    参数量: 80*20 + 20*80 = 3,200 (仅约3.2K)
    """
    def __init__(self, mel_dim=80, reduction=4):
        super().__init__()
        hidden = max(mel_dim // reduction, 8)
        self.attn = nn.Sequential(
            nn.Linear(mel_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, mel_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """x: [B, T, D]"""
        # 全局频带统计量 (batch平均+时间平均)
        global_freq = x.mean(dim=1).mean(dim=0, keepdim=True)  # [1, D]
        freq_weights = self.attn(global_freq)  # [1, D]
        return x * freq_weights.unsqueeze(0)  # [B, T, D] 按频带加权


# ============ 改进2: 事件驱动预筛 ============
class EventDrivenPreScreener:
    """事件驱动快速预筛
    
    三层级联:
    Layer 1: 能量检测 (O(1)) → 过滤静音
    Layer 2: 峰值因子检测 (O(1)) → 过滤明显枪声
    Layer 3: BiLSTM推理 → 精细判别
    """
    def __init__(self, energy_threshold=0.01, crest_threshold=8.0):
        self.energy_threshold = energy_threshold
        self.crest_threshold = crest_threshold
        self.stats = {'total': 0, 'l1_silence': 0, 'l2_gunshot': 0, 'l3_bilstm': 0}
    
    def screen(self, audio_frame):
        """返回: (label, need_bilstm)
        label=None 表示需要BiLSTM推理
        """
        self.stats['total'] += 1
        
        # Layer 1: 能量检测
        rms = np.sqrt(np.mean(audio_frame ** 2))
        max_rms = max(rms, 1e-10)
        
        if rms < self.energy_threshold * 0.1:  # 极低能量=静音
            self.stats['l1_silence'] += 1
            return CLASS_SILENCE, False
        
        # Layer 2: 峰值因子检测
        peak = np.max(np.abs(audio_frame))
        crest = peak / max_rms
        if crest > self.crest_threshold:  # 高峰值因子=脉冲(枪声)
            self.stats['l2_gunshot'] += 1
            return CLASS_GUNSHOT, False
        
        # Layer 3: 需要BiLSTM精细推理
        self.stats['l3_bilstm'] += 1
        return None, True
    
    def report(self):
        t = max(self.stats['total'], 1)
        logger.info(f"预筛统计: 总计{self.stats['total']}帧, "
                    f"L1静音={self.stats['l1_silence']}({self.stats['l1_silence']/t*100:.1f}%), "
                    f"L2枪声={self.stats['l2_gunshot']}({self.stats['l2_gunshot']/t*100:.1f}%), "
                    f"L3推理={self.stats['l3_bilstm']}({self.stats['l3_bilstm']/t*100:.1f}%)")


# ============ 模型: BiLSTM + 频域注意力 ============
class BiLSTMVAD_v5(nn.Module):
    """RangeVAD V5: BiLSTM + 频域注意力"""
    def __init__(self, input_dim=80, hidden_dim=64, num_layers=2, dropout=0.2,
                 use_freq_attn=True):
        super().__init__()
        self.use_freq_attn = use_freq_attn
        
        if use_freq_attn:
            self.freq_attn = FrequencyAttention(mel_dim=input_dim, reduction=4)
        
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              bidirectional=True, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )
        self._total = sum(p.numel() for p in self.parameters())
        if use_freq_attn:
            self._attn_params = sum(p.numel() for p in self.freq_attn.parameters())
            logger.info(f"频域注意力: {self._attn_params:,} params")
    
    def forward(self, x):
        if self.use_freq_attn:
            x = self.freq_attn(x)
        x, _ = self.bilstm(x)
        return self.classifier(x)


# ============ 数据集 ============
class VAD4Dataset(Dataset):
    def __init__(self, manifest_path, max_frames=600):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.data = []
        logger.info(f"加载 {manifest_path}: {len(self.manifest)} 条记录, 正在提取特征...")
        from tqdm import tqdm as _tqdm
        for item in _tqdm(self.manifest, desc="加载数据"):
            audio, _ = sf.read(item['audio'])
            feat = extract_logmel(audio)
            labels = np.load(item['label'])
            T = min(feat.shape[0], len(labels), max_frames)
            if T < 10: continue
            self.data.append({
                'feat': torch.FloatTensor(feat[:T]),
                'label': torch.LongTensor(labels[:T]),
                'length': T,
                'type': item.get('type', 'unknown'),
            })
        logger.info(f"加载完成: {len(self.data)} 样本 "
                    f"({sum(1 for d in self.data if d['type']=='concat')}拼接, "
                    f"{sum(1 for d in self.data if d['type']=='overlap')}重叠)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    max_len = max(b['feat'].shape[0] for b in batch)
    B, D = len(batch), batch[0]['feat'].shape[1]
    feat_pad = torch.zeros(B, max_len, D)
    label_pad = torch.full((B, max_len), -100, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b['length']
        feat_pad[i, :L] = b['feat']
        label_pad[i, :L] = b['label']
        mask[i, :L] = True
    return feat_pad, label_pad, mask


def find_manifest(data_dir, split):
    """自动查找manifest，兼容子目录和扁平结构"""
    flat = Path(data_dir) / f'{split}_manifest.json'
    sub = Path(data_dir) / split / 'manifest.json'
    if flat.exists():
        return str(flat)
    if sub.exists():
        return str(sub)
    raise FileNotFoundError(f"找不到 {split} manifest: {flat} 或 {sub}")


# ============ 训练 ============
def train(args):
    device = torch.device(args.device)
    
    train_manifest = find_manifest(args.data_dir, 'train')
    val_manifest = find_manifest(args.data_dir, 'val')
    
    train_ds = VAD4Dataset(train_manifest, max_frames=args.max_frames)
    val_ds = VAD4Dataset(val_manifest, max_frames=args.max_frames)
    
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=0)
    vl_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    
    use_freq_attn = not args.no_freq_attn
    model = BiLSTMVAD_v5(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                          dropout=args.dropout, use_freq_attn=use_freq_attn).to(device)
    logger.info(f"模型参数: {model._total:,} (频域注意力={'ON' if use_freq_attn else 'OFF'})")
    
    ce = nn.CrossEntropyLoss(ignore_index=-100,
                             weight=torch.tensor([args.gun_weight, 1.0, 0.5, 1.5], device=device))
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    best_acc = 0
    
    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tok, tcn = 0, 0, 0
        for feat, label, mask in tr_loader:
            feat, label, mask = feat.to(device), label.to(device), mask.to(device)
            logits = model(feat)
            loss = ce(logits[mask], label[mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
            preds = logits.argmax(-1)
            tok += (preds[mask] == label[mask]).sum().item()
            tcn += mask.sum().item()
            tl += loss.item()
        sch.step()
        
        model.eval()
        va, vc = 0, 0
        with torch.no_grad():
            for feat, label, mask in vl_loader:
                feat, label, mask = feat.to(device), label.to(device), mask.to(device)
                logits = model(feat)
                preds = logits.argmax(-1)
                va += (preds[mask] == label[mask]).sum().item()
                vc += mask.sum().item()
        vac = va / max(vc, 1)
        logger.info(f"Ep {ep:2d}: loss={tl/len(tr_loader):.4f} val_acc={vac:.4f}")
        
        if vac > best_acc:
            best_acc = vac
            torch.save({'model_state_dict': model.state_dict(),
                        'config': {'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers,
                                   'dropout': args.dropout, 'input_dim': 80,
                                   'use_freq_attn': use_freq_attn},
                       }, od / 'model.pt.best')
    
    logger.info(f"最佳验证准确率: {best_acc:.4f}")
    return model


# ============ 评估 (含事件驱动预筛) ============
def evaluate(args):
    device = torch.device(args.device)
    ck = torch.load(args.model_path or f'{args.output_dir}/model.pt.best',
                    map_location=device, weights_only=False)
    cfg = ck['config']
    
    model = BiLSTMVAD_v5(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
                          dropout=cfg['dropout'], 
                          use_freq_attn=cfg.get('use_freq_attn', True)).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    logger.info(f"模型参数: {model._total:,}")
    
    # 创建预筛器
    screener = EventDrivenPreScreener(energy_threshold=0.01, crest_threshold=8.0)
    
    data_dir = Path(args.data_dir)
    
    for test_name in ['testA', 'testB']:
        manifest_path = data_dir / f'{test_name}_manifest.json'
        if not manifest_path.exists(): continue
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"{test_name} 评估 (新枪型)")
        print(f"{'='*70}")
        
        for subset_type in ['concat', 'overlap', 'all']:
            subset = [m for m in manifest 
                      if subset_type == 'all' or m['type'] == subset_type]
            if not subset: continue
            
            total_gun, gun_correct = 0, 0
            total_spk, spk_correct = 0, 0
            total_sil, sil_correct = 0, 0
            gun2speech = 0
            events_total = {'l1_sil': 0, 'l2_gun': 0, 'l3': 0}
            
            for item in subset:
                try:
                    audio, _ = sf.read(item['audio'])
                except: continue
                feat = extract_logmel(audio)
                labels = np.load(item['label'])
                T = min(len(feat), len(labels))
                if T < 10: continue
                
                # Step 1: 全序列BiLSTM推理 (保持时序上下文)
                feat_t = torch.FloatTensor(feat[:T]).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(feat_t)
                bilstm_preds = logits[0].argmax(-1).cpu().numpy()
                
                # Step 2: 事件驱动预筛 → 对高置信度帧进行后置修正
                preds = bilstm_preds.copy()
                
                for t in range(T):
                    audio_start = t * HOP
                    audio_end = min(audio_start + 500, len(audio))
                    audio_frame = audio[audio_start:audio_end]
                    
                    quick_label, need_bilstm = screener.screen(audio_frame)
                    
                    if not need_bilstm:
                        # 预筛器高置信度 → 覆盖BiLSTM结果
                        preds[t] = quick_label
                        if quick_label == CLASS_SILENCE:
                            events_total['l1_sil'] += 1
                        else:
                            events_total['l2_gun'] += 1
                    else:
                        events_total['l3'] += 1
                
                mask_gun = labels[:T] == CLASS_GUNSHOT
                mask_spk = labels[:T] == CLASS_SPEECH
                mask_sil = labels[:T] == CLASS_SILENCE
                
                total_gun += mask_gun.sum()
                total_spk += mask_spk.sum()
                total_sil += mask_sil.sum()
                gun_correct += (preds[mask_gun] == CLASS_GUNSHOT).sum()
                spk_correct += (preds[mask_spk] == CLASS_SPEECH).sum()
                sil_correct += (preds[mask_sil] == CLASS_SILENCE).sum()
                gun2speech += (preds[mask_gun] == CLASS_SPEECH).sum()
            
            GFAR = gun2speech / max(total_gun, 1)
            SDR = spk_correct / max(total_spk, 1)
            FAR = gun2speech / max(total_gun + total_sil, 1)
            
            print(f"\n  [{subset_type}] n={len(subset)}")
            print(f"    GFAR={GFAR*100:.2f}%  SDR={SDR*100:.2f}%  FAR={FAR*100:.2f}%")
            
            total_events = sum(events_total.values())
            if total_events > 0:
                print(f"    预筛: 静音覆盖{events_total['l1_sil']/total_events*100:.1f}% "
                      f"枪声覆盖{events_total['l2_gun']/total_events*100:.1f}% "
                      f"推理{events_total['l3']/total_events*100:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='train', choices=['train', 'eval'])
    p.add_argument('--data-dir', default='/home/lvzhiyin/桌面/FunASR-main/rccm/vad_data_v4')
    p.add_argument('--output-dir', default='./vad5_output')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--max-frames', type=int, default=600)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--gun-weight', type=float, default=3.0)
    p.add_argument('--model-path', default=None)
    p.add_argument('--no-freq-attn', action='store_true', 
                   help='消融: 禁用频域注意力 (训练纯BiLSTM作为对照)')
    args = p.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
