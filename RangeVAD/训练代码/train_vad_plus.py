#!/usr/bin/env python3
"""
RangeVAD-Plus 训练脚本
用法:
    python3 train_vad_plus.py \
        --mode train \
        --data-dir ../data/vad_train_4cls \
        --output-dir ./vad_plus_output \
        --epochs 30 --batch-size 32 --device cuda
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vad_v5 import extract_logmel, find_manifest  # 复用特征提取
from RangeVAD_Plus import RangeVADPlus

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('RangeVAD-Plus')


class VADPlusDataset(Dataset):
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
            if T < 10:
                continue
            self.data.append({
                'feat': torch.FloatTensor(feat[:T]),
                'label': torch.LongTensor(labels[:T]),
                'length': T,
                'type': item.get('type', 'unknown'),
            })
        logger.info(f"加载完成: {len(self.data)} 样本 "
                    f"({sum(1 for d in self.data if d['type'] == 'concat')}拼接, "
                    f"{sum(1 for d in self.data if d['type'] == 'overlap')}重叠)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


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


def train(args):
    device = torch.device(args.device)

    train_manifest = find_manifest(args.data_dir, 'train')
    val_manifest = find_manifest(args.data_dir, 'val')

    train_ds = VADPlusDataset(train_manifest, max_frames=args.max_frames)
    val_ds = VADPlusDataset(val_manifest, max_frames=args.max_frames)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=0)
    vl_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)

    model = RangeVADPlus(mel_dim=80, hidden_dim=args.hidden_dim, num_classes=3,
                         dfsmn_blocks=args.dfsmn_blocks, look_back=args.look_back)
    model = model.to(device)
    logger.info(f"隐藏维度: {args.hidden_dim}, DFSMN块: {args.dfsmn_blocks}, look_back: {args.look_back}")

    ce = nn.CrossEntropyLoss(ignore_index=-100,
                             weight=torch.tensor([1.0, 1.0, 0.5], device=device))
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    best_acc = 0
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tl, tok, tcn = 0, 0, 0
        for feat, label, mask in tr_loader:
            feat, label, mask = feat.to(device), label.to(device), mask.to(device)
            logits = model(feat)
            loss = ce(logits[mask], label[mask])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
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
        logger.info(f"Ep {ep:2d}: loss={tl / len(tr_loader):.4f} val_acc={vac:.4f}")

        if vac > best_acc:
            best_acc = vac
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'hidden_dim': args.hidden_dim,
                    'dfsmn_blocks': args.dfsmn_blocks,
                    'look_back': args.look_back,
                },
            }, od / 'model.pt.best')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"早停: 连续 {args.patience} 个 epoch 验证准确率无提升")
                break

    logger.info(f"最佳验证准确率: {best_acc:.4f}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='train', choices=['train'])
    p.add_argument('--data-dir', default='/home/FunASR/FunASR-main/data/vad_train_4cls')
    p.add_argument('--output-dir', default='./vad_plus_output')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--dfsmn-blocks', type=int, default=3)
    p.add_argument('--look-back', type=int, default=20)
    p.add_argument('--max-frames', type=int, default=600)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=5)
    args = p.parse_args()

    train(args)


if __name__ == '__main__':
    main()
