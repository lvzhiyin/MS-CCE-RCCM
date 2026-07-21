#!/usr/bin/env python3
"""
RangeVAD-Plus 2分类消融训练
与 train_vad_plus.py 架构完全一致 (无IR), 仅输出头从3类改为2类

标签映射: {干净语音, 带噪/重叠语音}→语音, {纯枪声, 静音}→非语音

用法:
    python3 train_vad_plus_2class.py \
        --mode train \
        --data-dir ../data/vad_train_4cls \
        --output-dir ./vad_plus_2class_output \
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
from train_vad_v5 import extract_logmel, find_manifest
from RangeVAD_Plus_2class import RangeVADPlus2Class

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('VAD-Plus-2Class')


class VAD2ClassDataset(Dataset):
    """加载4分类数据, 映射为2分类: 语音(1)和重叠(3)→1, 枪声(0)和静音(2)→0"""

    def __init__(self, manifest_path, max_frames=600):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.data = []
        logger.info(f"加载 {manifest_path}: {len(self.manifest)} 条, 提取特征...")
        from tqdm import tqdm
        for item in tqdm(self.manifest, desc="加载数据"):
            audio, _ = sf.read(item['audio'])
            feat = extract_logmel(audio)
            labels_4c = np.load(item['label'])
            T = min(feat.shape[0], len(labels_4c), max_frames)
            if T < 10:
                continue
            labels_2c = np.where((labels_4c[:T] == 1) | (labels_4c[:T] == 3), 1, 0)
            self.data.append({
                'feat': torch.FloatTensor(feat[:T]),
                'label': torch.LongTensor(labels_2c),
                'length': T,
                'type': item.get('type', 'unknown'),
            })
        logger.info(f"加载完成: {len(self.data)} 样本")

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

    train_ds = VAD2ClassDataset(train_manifest, max_frames=args.max_frames)
    val_ds = VAD2ClassDataset(val_manifest, max_frames=args.max_frames)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=0)
    vl_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)

    # 同架构, 无IR, 2分类输出头
    model = RangeVADPlus(mel_dim=80, hidden_dim=args.hidden_dim, num_classes=2,
                         dfsmn_blocks=args.dfsmn_blocks, look_back=args.look_back,
                         use_ir=False)
    model = model.to(device)
    logger.info(f"2分类 RangeVAD-Plus: hidden_dim={args.hidden_dim}, "
                f"DFSMN={args.dfsmn_blocks}, look_back={args.look_back}, 无IR")

    # 非语音权重略高(减少枪声漏检)
    ce = nn.CrossEntropyLoss(ignore_index=-100,
                             weight=torch.tensor([args.gun_weight, 1.0], device=device))
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    best_acc = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0
        for feat, label, mask in tr_loader:
            feat, label, mask = feat.to(device), label.to(device), mask.to(device)
            logits = model(feat)
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'hidden_dim': args.hidden_dim,
                    'dfsmn_blocks': args.dfsmn_blocks,
                    'look_back': args.look_back,
                    'num_classes': 2,
                },
            }, od / 'model.pt.best')

    logger.info(f"最佳验证准确率: {best_acc:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='train', choices=['train'])
    p.add_argument('--data-dir', default='/home/FunASR/FunASR-main/data/vad_train_4cls')
    p.add_argument('--output-dir', default='./vad_plus_2class_output')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--dfsmn-blocks', type=int, default=3)
    p.add_argument('--look-back', type=int, default=20)
    p.add_argument('--max-frames', type=int, default=600)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--gun-weight', type=float, default=3.0)
    args = p.parse_args()

    train(args)


if __name__ == '__main__':
    main()
