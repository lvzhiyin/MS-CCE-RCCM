#!/usr/bin/env python3
"""
VAD 帧级评估脚本
计算 GFAR (枪声误报率), SDR (语音检测率), ODR (带噪语音检测率), F1, RTF

三分类: 0=干净语音, 1=带噪语音, 2=非语音

用法:
    CUDA_VISIBLE_DEVICES=0 python3 eval_vad_frame.py \
        --data-dir /home/FunASR/FunASR-main/data/vad_train_4cls \
        --model-path rccm/vad_v8_4cls/model.pt.best \
        --split test \
        --max-samples 500
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_manifest(data_dir, split):
    flat = Path(data_dir) / f'{split}_manifest.json'
    sub = Path(data_dir) / split / 'manifest.json'
    if flat.exists():
        return str(flat)
    if sub.exists():
        return str(sub)
    raise FileNotFoundError(f"找不到 {split} manifest: {flat} 或 {sub}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-path', required=True, help='VAD权重路径 (.pt)')
    parser.add_argument('--vad-variant', default=None,
                        help='消融变体名称 (no_ir/no_lstm/no_dfsmn等)，checkpoint不含config时需指定')
    parser.add_argument('--split', default='test', help='test 或 val')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    # SNR 后处理参数
    parser.add_argument('--snr-model', default=None,
                        help='SNR估计器模型路径 (如 ./snr_estimator/model.pt)，不指定则不做后处理')
    parser.add_argument('--snr-threshold', type=float, default=-5.0,
                        help='SNR阈值(dB)，低于此值的label=1帧降级为非语音 (默认 -5dB)')
    args = parser.parse_args()

    device = torch.device(args.device)

    # 加载数据
    manifest_path = find_manifest(args.data_dir, args.split)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if args.max_samples > 0:
        manifest = manifest[:args.max_samples]

    # 加载模型
    from train_vad_v5 import extract_logmel
    ck = torch.load(args.model_path, map_location=device, weights_only=False)
    if args.vad_variant:
        from run_ablation import AblationVAD
        model = AblationVAD(args.vad_variant, mel_dim=80, hidden_dim=64, num_classes=3).to(device)
    elif 'config' in ck:
        cfg = ck['config']
        ncls = cfg.get('num_classes', 3)
        if 'dfsmn_blocks' in cfg:
            if ncls == 2:
                from RangeVAD_Plus_2class import RangeVADPlus2Class
                model = RangeVADPlus2Class(
                    mel_dim=80, hidden_dim=cfg['hidden_dim'],
                    dfsmn_blocks=cfg['dfsmn_blocks'],
                    look_back=cfg.get('look_back', 20)).to(device)
            else:
                from RangeVAD_Plus import RangeVADPlus
                model = RangeVADPlus(mel_dim=80, hidden_dim=cfg['hidden_dim'], num_classes=ncls,
                                     dfsmn_blocks=cfg['dfsmn_blocks'],
                                     look_back=cfg.get('look_back', 20)).to(device)
        else:
            from train_vad_v5 import BiLSTMVAD_v5
            model = BiLSTMVAD_v5(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
                                  dropout=cfg['dropout'],
                                  use_freq_attn=cfg.get('use_freq_attn', True)).to(device)
    else:
        raise RuntimeError("Checkpoint不含config，请指定 --vad-variant (如 no_ir, no_lstm)")
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    # 加载 SNR 后处理模型 (可选)
    snr_estimator = None
    if args.snr_model:
        print(f"\n加载 SNR 估计器: {args.snr_model}")
        from train_snr_estimator import SNREstimator
        snr_estimator = SNREstimator(mel_dim=80, hidden_dim=64, num_layers=2).to(device)
        snr_estimator.load_state_dict(torch.load(args.snr_model, map_location=device))
        snr_estimator.eval()
        print(f"  SNR 阈值 = {args.snr_threshold} dB")
        print(f"  label=1 帧中 SNR < {args.snr_threshold}dB → 降级为非语音(丢弃)")

    # 统计
    # 多分类: 自动检测类别数
    #   2分类: 0=非语音, 1=语音
    #   3分类: 0=干净语音, 1=带噪语音, 2=非语音
    #   4分类: 0=纯枪声, 1=干净语音, 2=静音, 3=重叠枪声
    ncls = cfg.get('num_classes', 3) if 'config' in ck else 3
    total_frames = 0
    total_time = 0

    # 动态类别名
    if ncls == 2:
        class_names = ['非语音', '语音']
        # 2分类GFAR: 非语音→语音 误报率
        # 2分类SDR:  语音检测率
    elif ncls == 4:
        class_names = ['纯枪声', '干净语音', '静音', '重叠枪声']
    else:
        class_names = ['干净语音', '带噪语音', '非语音']

    tp = np.zeros(ncls, dtype=np.int64)
    fp = np.zeros(ncls, dtype=np.int64)
    fn = np.zeros(ncls, dtype=np.int64)
    confusion = np.zeros((ncls, ncls), dtype=np.int64)
    total_post_dropped = 0

    for item in manifest:
        audio, sr = sf.read(item['audio'])
        feat = extract_logmel(audio)
        gt = np.load(item['label'])
        T = min(len(feat), len(gt))
        if T < 10:
            continue
        feat_t = torch.FloatTensor(feat[:T]).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            logits = model(feat_t)
        elapsed = time.time() - t0
        total_frames += T
        total_time += elapsed

        pred = logits[0].argmax(-1).cpu().numpy()
        gt = gt[:T]

        # 2分类数据: 原始4类GT映射为2类 (1语音+3重叠→1, 0枪声+2静音→0)
        if ncls == 2:
            gt = np.where((gt == 1) | (gt == 3), 1, 0)

        # SNR 后处理 (仅3分类以上)
        if ncls >= 3 and snr_estimator is not None:
            feat_t_snr = torch.FloatTensor(feat[:T]).unsqueeze(0).to(device)
            with torch.no_grad():
                snr_pred = snr_estimator(feat_t_snr).squeeze(0).cpu().numpy()

            post_pred = pred.copy()
            mask_label1 = (post_pred == 1)
            mask_low_snr = (snr_pred < args.snr_threshold)
            post_pred[mask_label1 & mask_low_snr] = 2
            total_post_dropped += int((mask_label1 & mask_low_snr).sum())
            pred = post_pred

        for c in range(ncls):
            tp[c] += ((pred == c) & (gt == c)).sum()
            fp[c] += ((pred == c) & (gt != c)).sum()
            fn[c] += ((gt == c) & (pred != c)).sum()

        for c_gt in range(ncls):
            for c_pred in range(ncls):
                confusion[c_gt, c_pred] += ((gt == c_gt) & (pred == c_pred)).sum()

    # ========== 计算指标 ==========
    print(f"\n{'='*60}")
    ncls_tag = f"{ncls}分类" if ncls != 3 else "三分类"
    print(f"VAD 帧级评估 ({args.split} 集, {ncls_tag}, {len(manifest)} 条, {total_frames} 帧)")
    print(f"{'='*60}")

    # 各类精度
    for c in range(ncls):
        prec = tp[c] / max(tp[c] + fp[c], 1) * 100
        rec = tp[c] / max(tp[c] + fn[c], 1) * 100
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        print(f"  {class_names[c]:>6}: Precision={prec:.1f}%  Recall={rec:.1f}%  F1={f1:.1f}%  (TP={tp[c]}, FP={fp[c]}, FN={fn[c]})")

    # GFAR / SDR — 按类别数自适应计算
    if ncls == 2:
        # 2分类: 非语音(0)→语音(1) 误报 = GFAR
        gf_total = confusion[0].sum()
        gf_fp = confusion[0, 1]
        GFAR = gf_fp / max(gf_total, 1) * 100
        # SDR = 语音检测率
        sdr_correct = tp[1]
        sdr_total = tp[1] + fn[1]
    elif ncls == 4:
        # 4分类: 非语音 = 纯枪声(0) + 静音(2)
        gf_total = confusion[0].sum() + confusion[2].sum()
        gf_fp = (confusion[0, 1] + confusion[0, 3] +
                 confusion[2, 1] + confusion[2, 3])
        GFAR = gf_fp / max(gf_total, 1) * 100
        # SDR = 干净语音(1) + 重叠(3) 检测率
        sdr_correct = tp[1] + tp[3]
        sdr_total = tp[1] + tp[3] + fn[1] + fn[3]
    else:
        # 3分类: 非语音(2)→保留(0或1) 误报
        gf_total = confusion[2].sum()
        gf_fp = confusion[2, 0] + confusion[2, 1]
        GFAR = gf_fp / max(gf_total, 1) * 100
        # SDR = 干净语音(0) + 带噪语音(1) 检测率
        sdr_correct = tp[0] + tp[1]
        sdr_total = tp[0] + tp[1] + fn[0] + fn[1]
    SDR = sdr_correct / max(sdr_total, 1) * 100

    # 整体准确率
    accuracy = tp.sum() / max(total_frames, 1) * 100

    # 加权F1
    f1_total = 0
    for c in range(ncls):
        prec_c = tp[c] / max(tp[c] + fp[c], 1)
        rec_c = tp[c] / max(tp[c] + fn[c], 1)
        f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-10)
        f1_total += f1_c * (tp[c] + fn[c])
    F1 = f1_total / max(total_frames, 1) * 100

    # RTF
    total_audio_s = sum([
        sf.info(item['audio']).duration for item in manifest
    ]) if manifest else 1
    RTF = total_time / max(total_audio_s, 1)

    print(f"\n  整体准确率: {accuracy:.2f}%")
    print(f"  GFAR (非语音误报为语音): {GFAR:.2f}%")
    print(f"  SDR  (语音检测率):       {SDR:.2f}%")
    print(f"  F1   (加权F1):           {F1:.2f}%")
    print(f"  RTF  (实时因子):         {RTF:.4f}")
    print(f"  总推理时间:              {total_time:.2f}s")
    if ncls >= 3 and snr_estimator is not None:
        pct = total_post_dropped / max(total_frames, 1) * 100
        print(f"  SNR后处理丢弃帧:         {total_post_dropped} ({pct:.2f}%)")

    # 表格格式输出
    print(f"\n{'| 指标 | 值 |':-^40}")
    print(f"{'|':-^40}")
    print(f"{'| GFAR':<12} {'|':<4} {GFAR:>6.2f}% {'|':<4}")
    print(f"{'| SDR':<12} {'|':<4} {SDR:>6.2f}% {'|':<4}")
    print(f"{'| F1':<12} {'|':<4} {F1:>6.2f}% {'|':<4}")
    print(f"{'| RTF':<12} {'|':<4} {RTF:>6.4f} {'|':<4}")
    print(f"{'':-^40}")

    # 混淆矩阵
    print(f"\n{'='*60}")
    print("混淆矩阵 (行=真实, 列=预测)")
    print(f"{'='*60}")
    header = f"{'':>10}" + ''.join(f"{n:>8}" for n in class_names)
    print(header)
    for c_gt in range(ncls):
        row = f"{class_names[c_gt]:>10}"
        for c_pred in range(ncls):
            row += f"{confusion[c_gt, c_pred]:>8}"
        print(row)

    print(f"\n归一化混淆矩阵 (每行归一化, %)")
    print(header)
    for c_gt in range(ncls):
        row_total = confusion[c_gt].sum()
        row = f"{class_names[c_gt]:>10}"
        for c_pred in range(ncls):
            pct = confusion[c_gt, c_pred] / max(row_total, 1) * 100
            row += f"{pct:>7.1f}%"
        print(row)


if __name__ == '__main__':
    main()
