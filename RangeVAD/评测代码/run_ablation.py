#!/usr/bin/env python3
"""
消融实验脚本

每次去掉一个模块 → 重新训练 → 测试，看每个模块的贡献。
RangeVAD 为四阶段架构（频谱编码器 + DFSMN×3 + BiLSTM + 三分类头，无 IRB）。

消融变体:
  full       → RangeVAD (四阶段, 3分类)
  no_lstm    → 去除 BiLSTM
  no_dfsmn   → 去除 DFSMN
  dfsmn_1    → DFSMN ×1
  dfsmn_2    → DFSMN ×2
  dfsmn_5    → DFSMN ×5

用法:
    python3 run_ablation.py \
        --data-dir ../data/vad_train_4cls \
        --output-dir ./ablation_v2 \
        --ablation all \
        --device cuda
"""

import os
import sys
import json
import argparse
import time
import csv
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vad_v5 import extract_logmel, find_manifest
from RangeVAD_Plus import DFSMNBlock

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('ablation')


# ============================================================
# 1. 消融模型变体
# ============================================================

class AblationVAD(nn.Module):
    """可配置模块的消融 VAD 模型"""

    def __init__(self, variant, mel_dim=80, hidden_dim=64, num_classes=3):
        super().__init__()
        self.variant = variant
        self.hidden_dim = hidden_dim

        # Stage 1: Spectral Encoder (所有变体共用)
        self.stem_padding = 4
        self.spectral_encoder = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 2: DFSMN
        if variant == 'dfsmn_1':
            n_blocks = 1
        elif variant == 'dfsmn_2':
            n_blocks = 2
        elif variant == 'dfsmn_5':
            n_blocks = 5
        elif variant == 'no_dfsmn':
            n_blocks = 0
        else:
            n_blocks = 3  # full, no_lstm 默认3块

        self.has_dfsmn = (n_blocks > 0)
        if self.has_dfsmn:
            self.dfsmn_blocks = nn.ModuleList([
                DFSMNBlock(hidden_dim, look_back=20) for _ in range(n_blocks)
            ])

        # Stage 3: BiLSTM
        self.has_lstm = (variant != 'no_lstm')
        if self.has_lstm:
            self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.lstm_dropout = nn.Dropout(0.2)

        # Stage 4: Head
        if self.has_lstm:
            head_in = hidden_dim * 2
        else:
            head_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        n = sum(p.numel() for p in self.parameters())
        logger.info(f"  消融变体 '{variant}': {n:,} 参数 ({n/1000:.1f}K)")

    def forward(self, x):
        B, T, _ = x.shape

        # Stage 1: Encoder
        x = x.transpose(1, 2)
        x = F.pad(x, (self.stem_padding, 0))
        x = self.spectral_encoder(x)
        x = x.transpose(1, 2)

        # Stage 2: DFSMN
        if self.has_dfsmn:
            for dfsmn in self.dfsmn_blocks:
                x = dfsmn(x)

        # Stage 3: BiLSTM
        if self.has_lstm:
            x, _ = self.bilstm(x)
            x = self.lstm_dropout(x)

        # Stage 4: Head
        return self.classifier(x)


# ============================================================
# 2. 数据加载
# ============================================================

class VADDataset(Dataset):
    def __init__(self, manifest_path, max_frames=600):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.data = []
        total = len(self.manifest)
        logger.info(f"加载 {manifest_path}: {total} 条...")
        from tqdm import tqdm
        for item in tqdm(self.manifest, desc="  加载数据", ncols=80):
            audio, _ = sf.read(item['audio'])
            feat = extract_logmel(audio)
            labels = np.load(item['label'])
            T = min(feat.shape[0], len(labels), max_frames)
            if T < 10:
                continue
            self.data.append({
                'feat': torch.FloatTensor(feat[:T]),
                'label': torch.LongTensor(labels[:T]),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['feat'], self.data[idx]['label']


def collate_fn(batch):
    max_len = max(b[0].shape[0] for b in batch)
    B = len(batch)
    D = batch[0][0].shape[1]
    feats = torch.zeros(B, max_len, D)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (f, l) in enumerate(batch):
        feats[i, :len(f)] = f
        labels[i, :len(l)] = l
    return feats, labels


# ============================================================
# 3. 训练函数
# ============================================================

def train_variant(model, variant_name, data_dir, output_dir, device='cuda',
                  epochs=20, batch_size=32, lr=1e-3, patience=5):
    od = Path(output_dir) / variant_name
    od.mkdir(parents=True, exist_ok=True)

    tr_set = VADDataset(find_manifest(data_dir, 'train'))
    va_set = VADDataset(find_manifest(data_dir, 'val'))

    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = model.to(device)
    # 3分类权重: 干净语音=1.0, 带噪语音=1.0, 非语音=0.5
    ce = nn.CrossEntropyLoss(ignore_index=-100,
                             weight=torch.tensor([1.0, 1.0, 0.5], device=device))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    best_acc = 0.0
    no_improve = 0
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for feat, label in tr_loader:
            feat, label = feat.to(device), label.to(device)
            logits = model(feat)
            mask = label != -100
            loss = ce(logits[mask], label[mask])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            tl += loss.item()
        sch.step()

        model.eval()
        va, vc = 0, 0
        with torch.no_grad():
            for feat, label in va_loader:
                feat, label = feat.to(device), label.to(device)
                logits = model(feat)
                mask = label != -100
                preds = logits.argmax(-1)
                va += (preds[mask] == label[mask]).sum().item()
                vc += mask.sum().item()

        vac = va / max(vc, 1)
        logger.info(f"[{variant_name}] Ep {ep+1:2d}: loss={tl/len(tr_loader):.4f}  val_acc={vac:.4f}")

        if vac > best_acc:
            best_acc = vac
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'variant': variant_name,
            }, od / 'model.pt.best')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[{variant_name}] 早停于 epoch {ep+1}")
                break

    logger.info(f"[{variant_name}] 训练完成, best_val_acc={best_acc:.4f}")
    return best_acc


# ============================================================
# 4. 帧级评估
# ============================================================

def evaluate_model(model, data_dir, device='cuda'):
    """3分类帧级评估: GFAR, SDR, F1"""
    manifest = json.load(open(find_manifest(data_dir, 'test')))

    class_names = ['干净语音', '带噪语音', '非语音']
    tp = np.zeros(3, dtype=np.int64)
    fp = np.zeros(3, dtype=np.int64)
    fn = np.zeros(3, dtype=np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    total_frames = 0
    t_start = time.time()

    model = model.to(device)
    model.eval()

    for item in manifest:
        audio, _ = sf.read(item['audio'])
        gt = np.load(item['label'])
        feat = extract_logmel(audio)
        T = min(feat.shape[0], len(gt))
        if T < 10:
            continue
        gt = gt[:T]
        feat_t = torch.FloatTensor(feat[:T]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(feat_t)
            pred = logits[0].argmax(-1).cpu().numpy()

        for c in range(3):
            tp[c] += ((pred == c) & (gt == c)).sum()
            fp[c] += ((pred == c) & (gt != c)).sum()
            fn[c] += ((gt == c) & (pred != c)).sum()

        for c_gt in range(3):
            for c_pr in range(3):
                confusion[c_gt, c_pr] += ((gt == c_gt) & (pred == c_pr)).sum()
        total_frames += T

    elapsed = time.time() - t_start

    GFAR = (confusion[2, 0] + confusion[2, 1]) / max(confusion[2].sum(), 1) * 100
    SDR = (tp[0] + tp[1]) / max(tp[0] + tp[1] + fn[0] + fn[1], 1) * 100

    f1_total = 0
    for c in range(3):
        prec_c = tp[c] / max(tp[c] + fp[c], 1)
        rec_c = tp[c] / max(tp[c] + fn[c], 1)
        f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-10)
        f1_total += f1_c * (tp[c] + fn[c])
    F1 = f1_total / max(total_frames, 1) * 100

    total_audio_s = sum(sf.info(it['audio']).duration for it in manifest)
    RTF = elapsed / max(total_audio_s, 1)

    return {
        'GFAR': GFAR, 'SDR': SDR, 'F1': F1, 'RTF': RTF,
        'elapsed': elapsed, 'total_frames': total_frames,
    }


# ============================================================
# 5. 主函数
# ============================================================

def main():
    p = argparse.ArgumentParser(description='消融实验')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', default='./ablation_v2')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--ablation', nargs='+',
                   default=['full', 'no_lstm', 'no_dfsmn',
                            'dfsmn_1', 'dfsmn_2', 'dfsmn_5'],
                   help='消融变体列表')
    args = p.parse_args()

    if args.ablation == ['all']:
        variants = ['full', 'no_lstm', 'no_dfsmn',
                    'dfsmn_1', 'dfsmn_2', 'dfsmn_5']
    else:
        variants = args.ablation

    logger.info(f"\n{'='*80}")
    logger.info(f"消融实验: {len(variants)} 个变体")
    logger.info(f"{'='*80}")

    results = []
    for variant_name in variants:
        logger.info(f"\n{'='*60}")
        logger.info(f"训练变体: {variant_name}")
        logger.info(f"{'='*60}")

        model = AblationVAD(variant_name, mel_dim=80, hidden_dim=64, num_classes=3)

        best_acc = train_variant(
            model, variant_name, args.data_dir, args.output_dir,
            device=args.device, epochs=args.epochs, batch_size=args.batch_size
        )

        # 加载最佳模型
        ckpt_path = Path(args.output_dir) / variant_name / 'model.pt.best'
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        # 重建模型结构
        eval_model = AblationVAD(variant_name, mel_dim=80, hidden_dim=64, num_classes=3)
        eval_model.load_state_dict(ckpt['model_state_dict'])

        logger.info(f"评估: {variant_name}")
        m = evaluate_model(eval_model, args.data_dir, args.device)
        m['variant'] = variant_name
        m['params'] = sum(p.numel() for p in eval_model.parameters())
        m['val_acc'] = best_acc
        results.append(m)
        logger.info(f"  GFAR={m['GFAR']:.1f}% SDR={m['SDR']:.1f}% F1={m['F1']:.1f}% RTF={m['RTF']:.4f}")

        del model, eval_model
        torch.cuda.empty_cache()

    # ===== 输出汇总 =====
    logger.info(f"\n{'='*90}")
    logger.info("消融实验结果汇总")
    logger.info(f"{'='*90}")
    header = f"{'变体':<15} {'参数':>7} {'GFAR(%)':>8} {'SDR(%)':>8} {'F1(%)':>8} {'RTF':>8}"
    print(f"\n{header}")
    print("-" * 65)
    for m in results:
        params_k = m['params'] / 1000
        print(f"{m['variant']:<15} {params_k:>6.1f}K {m['GFAR']:>8.2f} {m['SDR']:>8.2f} {m['F1']:>8.2f} {m['RTF']:>8.4f}")

    # 保存 CSV
    csv_path = Path(args.output_dir) / 'ablation_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['variant', 'params', 'GFAR', 'SDR', 'F1', 'RTF', 'val_acc']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in results:
            row = {k: m[k] for k in fieldnames if k in m}
            w.writerow(row)
    logger.info(f"结果已保存: {csv_path}")


if __name__ == '__main__':
    main()
