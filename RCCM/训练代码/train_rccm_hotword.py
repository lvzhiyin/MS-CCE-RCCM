# #!/usr/bin/env python3
# # -*- encoding: utf-8 -*-
# """
# SACM 预训练 (热词感知损失版) — 基于 train_rccm_pretrain.py，增加 HotwordAwareLoss
# 用法:
#     python3 rccm/train_rccm_hotword.py \
#         --pairs rccm/rccm_train_0103_nopunct.jsonl \
#         --corpus rccm/corpus_0103_nopunct.txt \
#         --output-dir outputs_rccm_hotword \
#         --epochs 30 --batch-size 64 --lr 1e-3 --max-len 30
# """
# import os, sys, json, time, math, argparse, random, logging
# from datetime import datetime
# from pathlib import Path

# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.cuda.amp import autocast, GradScaler

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from light_rccm import LightSACM, CharTokenizer

# HOTWORD_LIST = [
#     '三号', '七号', '四号', '站立', '旋转', '前进', '暂停',
#     '匍匐', '九号', '停止', '向右', '倒靶', '后退', '向左',
#     '卧倒', '右转', '开始', '立靶', '六号', '射击', '放靶',
#     '二号', '冲击', '十号', '八号', '五号', '一号', '跃进', '左转',
# ]


# class HotwordAwareLoss(nn.Module):
#     def __init__(self, pad_idx=0, hotword_weight=10.0):
#         super().__init__()
#         self.pad_idx = pad_idx
#         self.hotword_weight = hotword_weight
#         self.base_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')

#     def forward(self, logits, target, hotword_mask):
#         B, T, V = logits.shape
#         base_loss = self.base_criterion(logits.reshape(-1, V), target.reshape(-1))
#         base_loss = base_loss.view(B, T)
#         hotword_mask = hotword_mask.to(base_loss.device)
#         weights = torch.ones_like(target, dtype=torch.float, device=base_loss.device)
#         weights = weights + hotword_mask * (self.hotword_weight - 1.0)
#         pad_mask = (target != self.pad_idx).float()
#         weights = weights * pad_mask
#         total_weight = weights.sum()
#         return (base_loss * weights).sum() / total_weight if total_weight > 0 else (base_loss * weights).sum()


# def create_hotword_mask(text, tokenizer, max_len=30):
#     mask = [0] * max_len
#     encoded = tokenizer.encode(text)
#     for hw in HOTWORD_LIST:
#         hw_ids = tokenizer.encode(hw)
#         hw_len = len(hw_ids)
#         if hw_len == 0: continue
#         for i in range(len(encoded) - hw_len + 1):
#             if i + hw_len > max_len: break
#             if encoded[i:i + hw_len] == hw_ids:
#                 for j in range(i, min(i + hw_len, max_len)):
#                     mask[j] = 1
#     return torch.tensor(mask, dtype=torch.float)


# # ======== 日志 (同原版) ========
# def setup_logging(log_file=None, log_level=logging.INFO):
#     logger = logging.getLogger('SACM_Hotword')
#     logger.setLevel(log_level)
#     logger.handlers = []
#     fmt = logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#     ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch.setFormatter(fmt); logger.addHandler(ch)
#     if log_file:
#         Path(log_file).parent.mkdir(parents=True, exist_ok=True)
#         fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
#         fh.setLevel(log_level); fh.setFormatter(fmt); logger.addHandler(fh)
#         logger.info(f"日志文件: {Path(log_file).absolute()}")
#     return logger

# class LoggerWriter:
#     def __init__(self, logger, level=logging.INFO):
#         self.logger = logger; self.level = level; self._buffer = ''
#     def write(self, msg):
#         if msg and not msg.isspace(): self.logger.log(self.level, msg.strip())
#     def flush(self): pass

# def setup_system_output(log_file=None):
#     logger = setup_logging(log_file=log_file)
#     sys.stdout = LoggerWriter(logger, logging.INFO)
#     sys.stderr = LoggerWriter(logger, logging.ERROR)
#     return logger


# # ======== 数据集（增加 hotword_mask） ========
# class CorrectionDataset(Dataset):
#     def __init__(self, pairs_jsonl, tokenizer, max_len=30):
#         self.pairs = []
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         print(f"加载数据: {pairs_jsonl}")
#         with open(pairs_jsonl, 'r', encoding='utf-8') as f:
#             for line_num, line in enumerate(f, 1):
#                 try:
#                     item = json.loads(line.strip())
#                     if 'src' in item and 'tgt' in item:
#                         src, tgt = item['src'], item['tgt']
#                     elif 'source' in item and 'target' in item:
#                         src, tgt = item['source'], item['target']
#                     elif 'noisy' in item and 'clean' in item:
#                         src, tgt = item['noisy'], item['clean']
#                     else:
#                         keys = list(item.keys())
#                         if len(keys) >= 2:
#                             src, tgt = str(item[keys[0]]), str(item[keys[1]])
#                         else:
#                             continue
#                     self.pairs.append((src, tgt))
#                 except json.JSONDecodeError:
#                     continue
#         print(f"  ✓ 加载 {len(self.pairs)} 条训练对")

#     def __len__(self): return len(self.pairs)

#     def __getitem__(self, idx):
#         src_text, tgt_text = self.pairs[idx]
#         src_ids = self.tokenizer.encode(src_text)
#         tgt_ids = self.tokenizer.encode(tgt_text, add_sos=True, add_eos=True)
#         src_ids = src_ids[:self.max_len]
#         tgt_ids = tgt_ids[:self.max_len]
#         src_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(src_ids))
#         tgt_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(tgt_ids))
#         # 热词 mask
#         hw_mask = create_hotword_mask(tgt_text, self.tokenizer, max_len=self.max_len)
#         return {
#             'src': torch.tensor(src_ids, dtype=torch.long),
#             'tgt': torch.tensor(tgt_ids, dtype=torch.long),
#             'hotword_mask': hw_mask,
#             'src_text': src_text,
#             'tgt_text': tgt_text,
#         }


# def collate_fn(batch):
#     return {
#         'src': torch.stack([b['src'] for b in batch]),
#         'tgt': torch.stack([b['tgt'] for b in batch]),
#         'hotword_mask': torch.stack([b['hotword_mask'] for b in batch]),
#         'src_texts': [b['src_text'] for b in batch],
#         'tgt_texts': [b['tgt_text'] for b in batch],
#     }


# # ======== 学习率调度（同原版） ========
# class WarmupCosineScheduler:
#     def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
#         self.optimizer = optimizer; self.warmup_steps = warmup_steps
#         self.total_steps = total_steps; self.min_lr = min_lr; self.current_step = 0
#     def step(self):
#         self.current_step += 1
#         if self.current_step <= self.warmup_steps:
#             lr = self.min_lr + (self.base_lr - self.min_lr) * (self.current_step / self.warmup_steps)
#         else:
#             progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
#             lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
#         for pg in self.optimizer.param_groups: pg['lr'] = lr
#         return lr
#     def get_lr(self): return self.optimizer.param_groups[0]['lr']


# # ======== 训练/评估（改用 HotwordAwareLoss） ========
# def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, epoch, hotword_loss_fn, log_interval=100):
#     model.train()
#     total_loss = num_batches = 0
#     for bi, batch in enumerate(dataloader):
#         src = batch['src'].to(device)
#         tgt = batch['tgt'].to(device)
#         hotword_mask = batch['hotword_mask'].to(device)
#         optimizer.zero_grad()
#         with autocast():
#             logits = model(src, tgt[:, :-1])
#             loss = hotword_loss_fn(logits, tgt[:, 1:], hotword_mask[:, 1:])
#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()
#         scheduler.step()
#         total_loss += loss.item(); num_batches += 1
#         if (bi + 1) % log_interval == 0:
#             print(f"  Epoch [{epoch}] Batch [{bi+1}/{len(dataloader)}] "
#                   f"Loss: {total_loss/num_batches:.4f} LR: {scheduler.get_lr():.6f}")
#     return total_loss / num_batches


# @torch.no_grad()
# def evaluate(model, dataloader, device, hotword_loss_fn):
#     model.eval()
#     total_loss = total_correct = total_tokens = num_batches = 0
#     for batch in dataloader:
#         src = batch['src'].to(device)
#         tgt = batch['tgt'].to(device)
#         hotword_mask = batch['hotword_mask'].to(device)
#         with autocast():
#             logits = model(src, tgt[:, :-1])
#             loss = hotword_loss_fn(logits, tgt[:, 1:], hotword_mask[:, 1:])
#         total_loss += loss.item(); num_batches += 1
#         preds = logits.argmax(dim=-1)
#         targets = tgt[:, 1:]
#         mask = (targets != model.tokenizer.pad_token_id)
#         total_correct += (preds[mask] == targets[mask]).sum().item()
#         total_tokens += mask.sum().item()
#     return total_loss / num_batches, total_correct / total_tokens if total_tokens > 0 else 0


# def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir, is_best=False):
#     ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': {'current_step': scheduler.current_step, 'base_lr': scheduler.base_lr},
#             'loss': loss, 'tokenizer': getattr(model, 'tokenizer', None),
#             'token2id': getattr(model, 'tokenizer', None).token2id if getattr(model, 'tokenizer', None) else None}
#     torch.save(ckpt, os.path.join(output_dir, 'checkpoint_latest.pt'))
#     if is_best:
#         torch.save(ckpt, os.path.join(output_dir, 'model.pt.best'))
#         print(f"  ✅ 保存最佳模型: {os.path.join(output_dir, 'model.pt.best')}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pairs', required=True); parser.add_argument('--corpus', default=None)
#     parser.add_argument('--output-dir', default='./outputs/rccm_hotword')
#     parser.add_argument('--epochs', type=int, default=30); parser.add_argument('--batch-size', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=1e-3); parser.add_argument('--warmup-steps', type=int, default=500)
#     parser.add_argument('--max-len', type=int, default=30); parser.add_argument('--log-interval', type=int, default=100)
#     parser.add_argument('--log-file', type=str, default=None); parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--hotword-weight', type=float, default=10.0, help='热词损失权重 (默认 10x)')
#     parser.add_argument('--d-model', type=int, default=256); parser.add_argument('--nhead', type=int, default=8)
#     parser.add_argument('--num-layers', type=int, default=4); parser.add_argument('--dim-ff', type=int, default=1024)
#     args = parser.parse_args()

#     setup_system_output(log_file=args.log_file)
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     os.makedirs(args.output_dir, exist_ok=True)

#     print("=" * 70)
#     print("SACM 预训练 (HotwordAwareLoss)")
#     print(f"损失: HotwordAwareLoss (热词权重 {args.hotword_weight}x)")
#     print(f"输出: {args.output_dir}, Epochs: {args.epochs}, Batch: {args.batch_size}")
#     print("=" * 70)

#     # 1. 词表
#     print("\n[1/4] 构建词表...")
#     corpus_texts = []
#     if args.corpus and os.path.exists(args.corpus):
#         with open(args.corpus, 'r', encoding='utf-8') as f:
#             corpus_texts = [l.strip() for l in f if l.strip()]
#     else:
#         seen = set()
#         with open(args.pairs, 'r', encoding='utf-8') as f:
#             for line in f:
#                 item = json.loads(line.strip())
#                 for k in ('tgt', 'target', 'clean'):
#                     t = item.get(k, '')
#                     if t and t not in seen:
#                         corpus_texts.append(t); seen.add(t)
#                         break
#     if not corpus_texts:
#         raise FileNotFoundError("无法构建词表！请提供 --corpus 或确保 --pairs 有效")
#     tokenizer = CharTokenizer(max_vocab=3000)
#     tokenizer.load_vocab(corpus_texts, max_vocab=2996)
#     vocab_size = len(tokenizer.token2id)
#     print(f"  词表大小: {vocab_size}")

#     # 2. 数据
#     print("\n[2/4] 创建数据集...")
#     ds = CorrectionDataset(args.pairs, tokenizer=tokenizer, max_len=args.max_len)
#     dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
#                      collate_fn=collate_fn, pin_memory=(device.type == 'cuda'))
#     print(f"  {len(ds)} 条, {len(dl)} batches")

#     # 3. 模型
#     print("\n[3/4] 创建模型...")
#     model = LightSACM(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
#                       num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
#                       dim_feedforward=args.dim_ff, max_seq_len=args.max_len, dropout=0.1)
#     model.tokenizer = tokenizer
#     model.to(device)
#     print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")

#     # 4. 优化器 + 热词损失
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
#     total_steps = args.epochs * len(dl)
#     scheduler = WarmupCosineScheduler(optimizer, warmup_steps=args.warmup_steps,
#                                        total_steps=total_steps, min_lr=args.lr * 0.01)
#     scheduler.base_lr = args.lr
#     scaler = GradScaler()
#     hotword_loss_fn = HotwordAwareLoss(pad_idx=tokenizer.pad_token_id,
#                                         hotword_weight=args.hotword_weight)

#     # 5. 训练
#     print("\n[4/4] 开始训练...")
#     best_loss = float('inf')
#     for epoch in range(1, args.epochs + 1):
#         print(f"\n{'='*60}\nEpoch [{epoch}/{args.epochs}]\n{'='*60}")
#         t0 = time.time()
#         train_loss = train_epoch(model, dl, optimizer, scheduler, device, scaler,
#                                  epoch, hotword_loss_fn, args.log_interval)
#         val_loss, val_acc = evaluate(model, dl, device, hotword_loss_fn)
#         print(f"\nEpoch [{epoch}] 完成: 训练损失={train_loss:.4f}  验证损失={val_loss:.4f}  "
#               f"准确率={val_acc*100:.2f}%  LR={scheduler.get_lr():.6f}  耗时={time.time()-t0:.1f}s")
#         is_best = val_loss < best_loss
#         if is_best: best_loss = val_loss
#         save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args.output_dir, is_best)

#     # 保存 final
#     final_path = os.path.join(args.output_dir, 'model.pt.final')
#     torch.save({
#         'model_state_dict': model.state_dict(), 'tokenizer': tokenizer,
#         'token2id': tokenizer.token2id, 'id2token': tokenizer.id2token,
#         'vocab_size': vocab_size,
#         'config': {'vocab_size': vocab_size, 'd_model': args.d_model, 'nhead': args.nhead,
#                    'num_encoder_layers': args.num_layers, 'num_decoder_layers': args.num_layers,
#                    'dim_feedforward': args.dim_ff, 'max_seq_len': args.max_len},
#     }, final_path)
#     print(f"\n{'='*60}\n训练完成！最佳损失={best_loss:.4f}\n最终模型={final_path}\n{'='*60}")


# if __name__ == '__main__':
#     main()




















































#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
SACM 预训练 (热词感知损失版) — 基于 train_rccm_pretrain.py，增加 HotwordAwareLoss
用法:
    python3 rccm/train_rccm_hotword.py \
        --pairs rccm/rccm_train_0103_nopunct.jsonl \
        --corpus rccm/corpus_0103_nopunct.txt \
        --output-dir outputs_rccm_hotword \
        --epochs 30 --batch-size 64 --lr 1e-3 --max-len 30
"""
import os, sys, json, time, math, argparse, random, logging
from datetime import datetime
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from light_rccm import LightSACM, CharTokenizer

HOTWORD_LIST = [
    '三号', '七号', '四号', '站立', '旋转', '前进', '暂停',
    '匍匐', '九号', '停止', '向右', '倒靶', '后退', '向左',
    '卧倒', '右转', '开始', '立靶', '六号', '射击', '放靶',
    '二号', '冲击', '十号', '八号', '五号', '一号', '跃进', '左转',
]


class HotwordAwareLoss(nn.Module):
    def __init__(self, pad_idx=0, hotword_weight=10.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.hotword_weight = hotword_weight
        self.base_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none')

    def forward(self, logits, target, hotword_mask):
        B, T, V = logits.shape
        base_loss = self.base_criterion(logits.reshape(-1, V), target.reshape(-1))
        base_loss = base_loss.view(B, T)
        hotword_mask = hotword_mask.to(base_loss.device)
        weights = torch.ones_like(target, dtype=torch.float, device=base_loss.device)
        weights = weights + hotword_mask * (self.hotword_weight - 1.0)
        pad_mask = (target != self.pad_idx).float()
        weights = weights * pad_mask
        total_weight = weights.sum()
        return (base_loss * weights).sum() / total_weight if total_weight > 0 else (base_loss * weights).sum()


def create_hotword_mask(text, tokenizer, max_len=30):
    mask = [0] * max_len
    encoded = tokenizer.encode(text)
    for hw in HOTWORD_LIST:
        hw_ids = tokenizer.encode(hw)
        hw_len = len(hw_ids)
        if hw_len == 0: continue
        for i in range(len(encoded) - hw_len + 1):
            if i + hw_len > max_len: break
            if encoded[i:i + hw_len] == hw_ids:
                for j in range(i, min(i + hw_len, max_len)):
                    mask[j] = 1
    return torch.tensor(mask, dtype=torch.float)


# ======== 日志 (同原版) ========
def setup_logging(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('SACM_Hotword')
    logger.setLevel(log_level)
    logger.handlers = []
    fmt = logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(log_level); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(log_level); fh.setFormatter(fmt); logger.addHandler(fh)
        logger.info(f"日志文件: {Path(log_file).absolute()}")
    return logger

class LoggerWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger; self.level = level; self._buffer = ''
    def write(self, msg):
        if msg and not msg.isspace(): self.logger.log(self.level, msg.strip())
    def flush(self): pass

def setup_system_output(log_file=None):
    logger = setup_logging(log_file=log_file)
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    return logger


# ======== 数据集（增加 hotword_mask） ========
class CorrectionDataset(Dataset):
    def __init__(self, pairs_jsonl, tokenizer, max_len=30):
        self.pairs = []
        self.error_labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"加载数据: {pairs_jsonl}")
        with open(pairs_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if 'src' in item and 'tgt' in item:
                        src, tgt = item['src'], item['tgt']
                    elif 'source' in item and 'target' in item:
                        src, tgt = item['source'], item['target']
                    elif 'noisy' in item and 'clean' in item:
                        src, tgt = item['noisy'], item['clean']
                    else:
                        keys = list(item.keys())
                        if len(keys) >= 2:
                            src, tgt = str(item[keys[0]]), str(item[keys[1]])
                        else:
                            continue
                    self.pairs.append((src, tgt))
                    self.error_labels.append(item.get('error_labels', []))
                except json.JSONDecodeError:
                    continue
        print(f"  ✓ 加载 {len(self.pairs)} 条训练对")
        has_labels = sum(1 for e in self.error_labels if e)
        print(f"  ✓ 含错误标签: {has_labels} 条")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text, add_sos=True, add_eos=True)
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        src_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(src_ids))
        tgt_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(tgt_ids))
        # 热词 mask
        hw_mask = create_hotword_mask(tgt_text, self.tokenizer, max_len=self.max_len)
        # 错误标签 (对齐到 max_len, pad 位置填 0)
        elabels = self.error_labels[idx] if self.error_labels else []
        elabels = elabels[:self.max_len] + [-100] * (self.max_len - len(elabels))
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'hotword_mask': hw_mask,
            'error_labels': torch.tensor(elabels, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text,
        }


def collate_fn(batch):
    return {
        'src': torch.stack([b['src'] for b in batch]),
        'tgt': torch.stack([b['tgt'] for b in batch]),
        'hotword_mask': torch.stack([b['hotword_mask'] for b in batch]),
        'error_labels': torch.stack([b['error_labels'] for b in batch]),
        'src_texts': [b['src_text'] for b in batch],
        'tgt_texts': [b['tgt_text'] for b in batch],
    }


# ======== 学习率调度（同原版） ========
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer; self.warmup_steps = warmup_steps
        self.total_steps = total_steps; self.min_lr = min_lr; self.current_step = 0
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups: pg['lr'] = lr
        return lr
    def get_lr(self): return self.optimizer.param_groups[0]['lr']


# ======== 训练/评估（改用 HotwordAwareLoss） ========
def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, epoch, hotword_loss_fn,
                 det_weight=0.3, log_interval=100):
    model.train()
    total_loss = total_corr = total_det = num_batches = 0
    det_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    for bi, batch in enumerate(dataloader):
        src = batch['src'].to(device, non_blocking=True)
        tgt = batch['tgt'].to(device, non_blocking=True)
        hotword_mask = batch['hotword_mask'].to(device, non_blocking=True)
        error_labels = batch['error_labels'].to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            logits, det_logits = model(src, tgt[:, :-1])
            corr_loss = hotword_loss_fn(logits, tgt[:, 1:], hotword_mask[:, 1:])
            loss = corr_loss
            if det_logits is not None:
                det_loss = det_criterion(det_logits.permute(0, 2, 1), error_labels)
                loss = corr_loss + det_weight * det_loss
                total_det += det_loss.item()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item(); total_corr += corr_loss.item(); num_batches += 1
        if bi == 0:
            print(f"  Epoch [{epoch}] Batch [1/{len(dataloader)}] — first batch OK")
        if (bi + 1) % log_interval == 0:
            lr = scheduler.get_lr()
            if det_logits is not None:
                print(f"  Epoch [{epoch}] Batch [{bi+1}/{len(dataloader)}] "
                      f"Loss: {total_loss/num_batches:.4f} (C:{total_corr/num_batches:.4f} "
                      f"D:{total_det/num_batches:.4f}) LR: {lr:.6f}")
            else:
                print(f"  Epoch [{epoch}] Batch [{bi+1}/{len(dataloader)}] "
                      f"Loss: {total_loss/num_batches:.4f} LR: {lr:.6f}")
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device, hotword_loss_fn):
    model.eval()
    total_loss = total_correct = total_tokens = num_batches = 0
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        hotword_mask = batch['hotword_mask'].to(device)
        error_labels = batch['error_labels'].to(device) if model.use_error_detector else None
        with autocast():
            logits, det_logits = model(src, tgt[:, :-1])
            loss = hotword_loss_fn(logits, tgt[:, 1:], hotword_mask[:, 1:])
        total_loss += loss.item(); num_batches += 1
        preds = logits.argmax(dim=-1)
        targets = tgt[:, 1:]
        mask = (targets != model.tokenizer.pad_token_id)
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_tokens += mask.sum().item()
    return total_loss / num_batches, total_correct / total_tokens if total_tokens > 0 else 0


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir, is_best=False):
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': {'current_step': scheduler.current_step, 'base_lr': scheduler.base_lr},
            'loss': loss, 'tokenizer': getattr(model, 'tokenizer', None),
            'token2id': getattr(model, 'tokenizer', None).token2id if getattr(model, 'tokenizer', None) else None}
    torch.save(ckpt, os.path.join(output_dir, 'checkpoint_latest.pt'))
    if is_best:
        torch.save(ckpt, os.path.join(output_dir, 'model.pt.best'))
        print(f"  ✅ 保存最佳模型: {os.path.join(output_dir, 'model.pt.best')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True); parser.add_argument('--corpus', default=None)
    parser.add_argument('--output-dir', default='./outputs/rccm_hotword')
    parser.add_argument('--epochs', type=int, default=30); parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3); parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--max-len', type=int, default=30); parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--log-file', type=str, default=None); parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hotword-weight', type=float, default=10.0, help='热词损失权重 (默认 10x)')
    parser.add_argument('--d-model', type=int, default=256); parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4); parser.add_argument('--dim-ff', type=int, default=1024)
    parser.add_argument('--use-error-detector', action='store_true', help='启用错误检测器 (Detector-Corrector)')
    parser.add_argument('--det-weight', type=float, default=0.3, help='检测 loss 权重')
    args = parser.parse_args()

    setup_system_output(log_file=args.log_file)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SACM 预训练 (HotwordAwareLoss)")
    print(f"损失: HotwordAwareLoss (热词权重 {args.hotword_weight}x)")
    if args.use_error_detector:
        print(f"  错误检测器: ✓ (检测权重 {args.det_weight}x)")
    print(f"输出: {args.output_dir}, Epochs: {args.epochs}, Batch: {args.batch_size}")
    print("=" * 70)

    # 1. 词表
    print("\n[1/4] 构建词表...")
    corpus_texts = []
    if args.corpus and os.path.exists(args.corpus):
        with open(args.corpus, 'r', encoding='utf-8') as f:
            corpus_texts = [l.strip() for l in f if l.strip()]
    else:
        seen = set()
        with open(args.pairs, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                for k in ('tgt', 'target', 'clean'):
                    t = item.get(k, '')
                    if t and t not in seen:
                        corpus_texts.append(t); seen.add(t)
                        break
    if not corpus_texts:
        raise FileNotFoundError("无法构建词表！请提供 --corpus 或确保 --pairs 有效")
    tokenizer = CharTokenizer(max_vocab=3000)
    tokenizer.load_vocab(corpus_texts, max_vocab=2996)
    vocab_size = len(tokenizer.token2id)
    print(f"  词表大小: {vocab_size}")

    # 2. 数据
    print("\n[2/4] 创建数据集...")
    ds = CorrectionDataset(args.pairs, tokenizer=tokenizer, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                     collate_fn=collate_fn, pin_memory=(device.type == 'cuda'))
    print(f"  {len(ds)} 条, {len(dl)} batches")

    # 3. 模型
    print("\n[3/4] 创建模型...")
    model = LightSACM(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
                      num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
                      dim_feedforward=args.dim_ff, max_seq_len=args.max_len, dropout=0.1,
                      use_error_detector=args.use_error_detector)
    model.tokenizer = tokenizer
    model.to(device)
    print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 优化器 + 热词损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(dl)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=args.warmup_steps,
                                       total_steps=total_steps, min_lr=args.lr * 0.01)
    scheduler.base_lr = args.lr
    scaler = GradScaler()
    hotword_loss_fn = HotwordAwareLoss(pad_idx=tokenizer.pad_token_id,
                                        hotword_weight=args.hotword_weight)

    # 5. 训练
    print("\n[4/4] 开始训练...")
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch [{epoch}/{args.epochs}]\n{'='*60}")
        t0 = time.time()
        train_loss = train_epoch(model, dl, optimizer, scheduler, device, scaler,
                                 epoch, hotword_loss_fn, det_weight=args.det_weight,
                                 log_interval=args.log_interval)
        val_loss, val_acc = evaluate(model, dl, device, hotword_loss_fn)
        print(f"\nEpoch [{epoch}] 完成: 训练损失={train_loss:.4f}  验证损失={val_loss:.4f}  "
              f"准确率={val_acc*100:.2f}%  LR={scheduler.get_lr():.6f}  耗时={time.time()-t0:.1f}s")
        is_best = val_loss < best_loss
        if is_best: best_loss = val_loss
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args.output_dir, is_best)

    # 保存 final
    final_path = os.path.join(args.output_dir, 'model.pt.final')
    torch.save({
        'model_state_dict': model.state_dict(), 'tokenizer': tokenizer,
        'token2id': tokenizer.token2id, 'id2token': tokenizer.id2token,
        'vocab_size': vocab_size,
        'config': {'vocab_size': vocab_size, 'd_model': args.d_model, 'nhead': args.nhead,
                   'num_encoder_layers': args.num_layers, 'num_decoder_layers': args.num_layers,
                   'dim_feedforward': args.dim_ff, 'max_seq_len': args.max_len,
                   'use_error_detector': args.use_error_detector},
    }, final_path)
    print(f"\n{'='*60}\n训练完成！最佳损失={best_loss:.4f}\n最终模型={final_path}\n{'='*60}")


if __name__ == '__main__':
    main()