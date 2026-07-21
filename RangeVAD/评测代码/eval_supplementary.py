#!/usr/bin/env python3
"""
补充实验脚本: 通用噪声泛化实验 + SNR 敏感性实验

=== Experiment 1: 通用噪声泛化实验 ===
  目的: 验证模型不是只记住枪声，而是真正学会区分语音与噪声
  方案: 取原始干净语音 → 混合 white/babble/factory 噪声 @ 0/5/10/15 dB SNR
        → 测试模型的三分类行为
  指标: SDR (语音帧保留率), 噪声帧分类分布

=== Experiment 2: SNR 敏感性实验 ===
  目的: 分析不同枪声 SNR 下模型的性能稳定性
  方案: 对测试集中重叠类样本按枪声 SNR 分桶统计 GFAR/SDR/F1
  分桶: strong[-15dB], core[-10~0], moderate[0~5], light[5~10], quiet[10~20]

用法:
    # 只跑通用噪声泛化
    python3 eval_supplementary.py \
        --data-dir ../data/vad_train_4cls \
        --model-path ./vad_plus_v2_output/model.pt.best \
        --device cuda \
        --exp noise

    # 只跑 SNR 敏感性
    python3 eval_supplementary.py \
        --data-dir ../data/vad_train_4cls \
        --model-path ./vad_plus_v2_output/model.pt.best \
        --device cuda \
        --exp snr

    # 两个都跑
    python3 eval_supplementary.py \
        --data-dir ../data/vad_train_4cls \
        --model-path ./vad_plus_v2_output/model.pt.best \
        --device cuda \
        --exp all
"""

import os
import sys
import json
import glob
import time
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vad_v5 import extract_logmel, find_manifest

SR = 16000
HOP_MS = 10.0
HOP = int(SR * HOP_MS / 1000)  # 160 samples


# ============================================================
# 工具函数
# ============================================================

def load_model(model_path, device, vad_variant=None):
    """加载 VAD 模型（支持标准模型和消融变体）"""
    ck = torch.load(model_path, map_location=device, weights_only=False)
    if vad_variant:
        from run_ablation import AblationVAD
        model = AblationVAD(vad_variant, mel_dim=80, hidden_dim=64, num_classes=3).to(device)
    elif 'config' in ck:
        cfg = ck['config']
        from RangeVAD_Plus import RangeVADPlus
        model = RangeVADPlus(mel_dim=80, hidden_dim=cfg['hidden_dim'], num_classes=3,
                             dfsmn_blocks=cfg['dfsmn_blocks'],
                             look_back=cfg.get('look_back', 20)).to(device)
    else:
        raise RuntimeError("Checkpoint不含config，请指定 --vad-variant (如 no_ir, no_lstm)")
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model


def compute_metrics_3class(pred, gt, confusion, tp, fp, fn):
    """累积3分类指标"""
    for c in range(3):
        tp[c] += ((pred == c) & (gt == c)).sum()
        fp[c] += ((pred == c) & (gt != c)).sum()
        fn[c] += ((gt == c) & (pred != c)).sum()
    for cg in range(3):
        for cp in range(3):
            confusion[cg, cp] += ((gt == cg) & (pred == cp)).sum()


def report_metrics(tp, fp, fn, confusion, total_frames, prefix=''):
    """计算并打印 GFAR/SDR/F1"""
    GFAR = (confusion[2, 0] + confusion[2, 1]) / max(confusion[2].sum(), 1) * 100
    SDR = (tp[0] + tp[1]) / max(tp[0] + tp[1] + fn[0] + fn[1], 1) * 100
    f1_total = 0
    for c in range(3):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1_c = 2 * prec * rec / max(prec + rec, 1e-10)
        f1_total += f1_c * (tp[c] + fn[c])
    F1 = f1_total / max(total_frames, 1) * 100
    acc = (confusion[0, 0] + confusion[1, 1] + confusion[2, 2]) / max(total_frames, 1) * 100
    if prefix:
        print(f"\n{prefix}")
    print(f"  GFAR={GFAR:.2f}%  SDR={SDR:.2f}%  F1={F1:.2f}%  Acc={acc:.2f}%  Frames={total_frames}")
    return {'GFAR': GFAR, 'SDR': SDR, 'F1': F1, 'accuracy': acc}


# ============================================================
# Experiment 1: 通用噪声泛化
# ============================================================

def generate_noise(noise_type, length_samples):
    """程序化生成通用噪声"""
    if noise_type == 'white':
        return np.random.randn(length_samples).astype(np.float32) * 0.3

    elif noise_type == 'babble':
        num_sources = 10
        noise = np.zeros(length_samples, dtype=np.float32)
        freqs = np.random.uniform(80, 4000, num_sources)
        amps = np.random.uniform(0.03, 0.15, num_sources) / np.sqrt(num_sources)
        t = np.arange(length_samples) / SR
        for f, a in zip(freqs, amps):
            noise += a * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
        return noise

    elif noise_type == 'factory':
        noise = np.zeros(length_samples, dtype=np.float32)
        t = np.arange(length_samples) / SR
        for f in [50, 100, 200, 400, 800]:
            noise += np.sin(2 * np.pi * f * t) * (200.0 / f)
        noise *= 0.3 / max(np.std(noise), 1e-10)
        return noise.astype(np.float32)
    else:
        raise ValueError(f"未知噪声: {noise_type}")


def mix_at_snr(signal, noise, snr_db):
    sig_pow = np.mean(signal ** 2) + 1e-12
    noise_pow = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(sig_pow / noise_pow * (10 ** (-snr_db / 10)))
    return signal + noise * scale


def run_noise_generalization(args):
    """通用噪声泛化实验"""
    print(f"\n{'='*80}")
    print("实验1: 通用噪声泛化实验")
    print(f"{'='*80}")

    device = torch.device(args.device)
    model = load_model(args.model_path, device, args.vad_variant)

    noise_types = args.noise_types.split(',')
    snr_list = [int(s) for s in args.snrs.split(',')]

    # 加载干净语音（从 raw data 或 test manifest 提取纯语音段）
    if args.clean_dir:
        clean_files = sorted(glob.glob(os.path.join(args.clean_dir, '*.wav')))
        print(f"从 {args.clean_dir} 加载 {len(clean_files)} 个干净语音文件")
        clean_audios = []
        for f in clean_files:
            audio, sr = sf.read(f)
            if sr != SR:
                from scipy import signal as sp_sig
                audio = sp_sig.resample(audio, int(len(audio) * SR / sr))
            if len(audio) >= SR * 0.5:
                clean_audios.append(audio.astype(np.float32))
    else:
        # 从测试集提取干净语音帧
        print("从测试集中提取纯语音段...")
        manifest = json.load(open(find_manifest(args.data_dir, 'test')))
        clean_audios = []
        for item in manifest[:args.max_clean]:
            if item.get('type') != 'overlap':
                continue
            audio, _ = sf.read(item['audio'])
            gt = np.load(item['label'])
            T = min(extract_logmel(audio).shape[0], len(gt))
            if T < 10:
                continue
            # 提取 label=0 (干净语音) 的帧, 且长度 >= 0.5s
            gt = gt[:T]
            clean_mask = (gt == 0)
            # 找连续段
            segments = []
            in_seg = False
            seg_start = 0
            for i, v in enumerate(clean_mask):
                if v and not in_seg:
                    seg_start = i
                    in_seg = True
                elif not v and in_seg:
                    seg_len = (i - seg_start) * HOP
                    if seg_len >= SR * 0.5:
                        segments.append((seg_start * HOP, i * HOP))
                    in_seg = False
            if in_seg:
                seg_len = (len(clean_mask) - seg_start) * HOP
                if seg_len >= SR * 0.5:
                    segments.append((seg_start * HOP, len(clean_mask) * HOP))
            for s, e in segments:
                clean_audios.append(audio[s:e].astype(np.float32))

        print(f"提取到 {len(clean_audios)} 个纯语音段")
        if args.max_clean > 0:
            clean_audios = clean_audios[:args.max_clean]

    print(f"有效语音段: {len(clean_audios)}, 噪声类型: {noise_types}, SNR: {snr_list}")

    results = []

    for noise_type in noise_types:
        for snr_db in snr_list:
            print(f"\n--- {noise_type} @ SNR={snr_db}dB ---")

            tp = np.zeros(3, dtype=np.int64)
            fp = np.zeros(3, dtype=np.int64)
            fn = np.zeros(3, dtype=np.int64)
            confusion = np.zeros((3, 3), dtype=np.int64)
            total_frames = 0

            for idx, clean in enumerate(clean_audios):
                # 构建测试样本: [静音0.2s] [干净语音+噪声] [静音0.2s]
                pad_s = int(0.2 * SR)
                noise_full = generate_noise(noise_type, len(clean) + 2 * pad_s)
                noisy_speech = mix_at_snr(clean, noise_full[pad_s:pad_s+len(clean)], snr_db)
                audio = np.concatenate([
                    noise_full[:pad_s] * 0.5,
                    noisy_speech,
                    noise_full[pad_s+len(clean):] * 0.5
                ])

                # 生成 ground truth 标签: 0=语音, 2=非语音
                feat = extract_logmel(audio)
                nf = feat.shape[0]
                sp_start = int(0.2 * 1000 / HOP_MS)
                sp_end = sp_start + int(len(clean) / SR * 1000 / HOP_MS)
                sp_end = min(sp_end, nf)

                gt = np.full(nf, 2, dtype=np.int64)  # 默认非语音
                gt[sp_start:sp_end] = 0  # 干净语音 (原始无枪声, 混合噪声后应识别为...)

                feat_t = torch.FloatTensor(feat[:nf]).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(feat_t)
                    pred = logits[0].argmax(-1).cpu().numpy()[:nf]

                compute_metrics_3class(pred, gt, confusion, tp, fp, fn)
                total_frames += nf

            report_metrics(tp, fp, fn, confusion, total_frames,
                          prefix=f"  {noise_type} SNR={snr_db}dB")

            # 分类分布
            total_predictions = confusion.sum()
            dist = {}
            for c, name in enumerate(['干净语音', '带噪语音', '非语音']):
                pct = confusion[:, c].sum() / max(total_predictions, 1) * 100
                dist[f'{name}_pct'] = pct
                if c == 0:
                    # 噪声帧中被分为干净语音的比例
                    n_ns = confusion[2].sum()
                    ns_as_clean = confusion[2, 0] / max(n_ns, 1) * 100 if n_ns > 0 else 0
                    dist['noise_as_clean_pct'] = ns_as_clean
                elif c == 1:
                    n_ns = confusion[2].sum()
                    ns_as_noisy = confusion[2, 1] / max(n_ns, 1) * 100 if n_ns > 0 else 0
                    dist['noise_as_noisy_pct'] = ns_as_noisy

            print(f"  分类分布: 干净语音={dist['干净语音_pct']:.1f}% "
                  f"带噪语音={dist['带噪语音_pct']:.1f}% 非语音={dist['非语音_pct']:.1f}%")
            print(f"  噪声帧→干净语音: {dist['noise_as_clean_pct']:.1f}% "
                  f"噪声帧→带噪语音: {dist['noise_as_noisy_pct']:.1f}%")

            gf = (confusion[2, 0] + confusion[2, 1]) / max(confusion[2].sum(), 1) * 100
            sd = (tp[0] + tp[1]) / max(tp[0] + tp[1] + fn[0] + fn[1], 1) * 100
            f1t = 0
            for c in range(3):
                pc = tp[c] / max(tp[c] + fp[c], 1)
                rc = tp[c] / max(tp[c] + fn[c], 1)
                f1c = 2 * pc * rc / max(pc + rc, 1e-10)
                f1t += f1c * (tp[c] + fn[c])
            f1v = f1t / max(total_frames, 1) * 100

            results.append({
                'exp': 'noise', 'noise': noise_type, 'snr_dB': snr_db,
                'GFAR': gf, 'SDR': sd, 'F1': f1v, 'total_frames': total_frames,
                **dist,
            })

    # 打印汇总
    print(f"\n{'='*90}")
    print("通用噪声泛化实验汇总")
    print(f"{'='*90}")
    hdr = f"{'噪声':<8} {'SNR':>5} {'GFAR%':>7} {'SDR%':>7} {'F1%':>7} {'→干净%':>7} {'→带噪%':>7} {'→非语音%':>7}"
    print(hdr)
    print("-" * 80)
    for r in results:
        print(f"{r['noise']:<8} {r['snr_dB']:>4}dB "
              f"{r['GFAR']:>6.2f} {r['SDR']:>6.2f} {r['F1']:>6.2f} "
              f"{r['noise_as_clean_pct']:>6.1f} {r['noise_as_noisy_pct']:>6.1f} "
              f"{r['非语音_pct']:>6.1f}")

    return results


# ============================================================
# Experiment 2: SNR 敏感性实验
# ============================================================

def estimate_gunshot_snr(audio, gt, hop=HOP):
    """从重叠样本中估计枪声段 SNR"""
    # 找 label=1 (带噪语音/枪声重叠) 的连续段
    changes = np.diff(np.concatenate([[0], (gt == 1).astype(int), [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    snr_values = []
    for s, e in zip(starts, ends):
        if e - s < 3:  # 跳过太短的段
            continue
        s_samp = s * hop
        e_samp = min(e * hop, len(audio))
        if e_samp <= s_samp:
            continue
        gun_seg = np.abs(audio[s_samp:e_samp])

        # 找相邻的干净语音帧 (label=0) 来估计语音功率
        before_clean = None
        for i in range(s-1, max(0, s-50), -1):
            if gt[i] == 0:
                bs = i * hop
                be = min((i+1) * hop, len(audio))
                before_clean = np.abs(audio[bs:be])
                break
        after_clean = None
        for i in range(e, min(len(gt), e+50)):
            if gt[i] == 0:
                a_s = i * hop
                a_e = min((i+1) * hop, len(audio))
                after_clean = np.abs(audio[a_s:a_e])
                break

        ref_rms = None
        if before_clean is not None:
            ref_rms = np.sqrt(np.mean(before_clean ** 2))
        if after_clean is not None:
            r = np.sqrt(np.mean(after_clean ** 2))
            if ref_rms is None or r > ref_rms:
                ref_rms = r
        if ref_rms is None or ref_rms < 1e-10:
            continue

        gun_rms = np.sqrt(np.mean(gun_seg ** 2))
        # 假设枪声功率远大于语音, 语音功率 ≈ ref_rms^2
        speech_pow = ref_rms ** 2
        # gun_overlap = speech + gunshot, so gunshot_pow ≈ total_pow - speech_pow
        gunshot_pow = max(0, gun_rms ** 2 - speech_pow)
        if gunshot_pow > 0:
            snr_tmp = 10 * np.log10(speech_pow / gunshot_pow)
            snr_values.append(snr_tmp)

    if len(snr_values) == 0:
        return None
    return np.mean(snr_values)


def frame_mask_to_samples(mask, hop, max_len):
    """将帧级布尔掩码展开为采样级掩码"""
    sample_mask = np.zeros(max_len, dtype=bool)
    for i, v in enumerate(mask):
        if v:
            s, e = i * hop, min((i + 1) * hop, max_len)
            sample_mask[s:e] = True
    return sample_mask


def estimate_sample_snr(audio, gt):
    """估计单个样本中枪声段的平均 SNR。
    拼接样本(label=2): 高能量段=枪声, SNR = speech_pow / gunshot_pow
    重叠样本(label=1): gunshot_pow = total_pow - speech_pow
    返回平均 SNR (dB)，若无法估计返回 None。"""
    L = len(audio)
    T = len(gt)

    # 干净语音段 RMS (样本级掩码)
    clean_mask = (gt == 0)
    if clean_mask.sum() < 5:
        return None
    clean_samp = frame_mask_to_samples(clean_mask, HOP, L)
    if clean_samp.sum() < HOP:
        return None
    clean_rms = np.sqrt(np.mean(audio[clean_samp] ** 2) + 1e-12)

    all_snr = []

    # === 重叠样本 (label=1) ===
    gun_mask = (gt == 1)
    changes = np.diff(np.concatenate([[0], gun_mask.astype(int), [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        s_samp, e_samp = s * HOP, min(e * HOP, L)
        seg_rms = np.sqrt(np.mean(audio[s_samp:e_samp] ** 2))
        gunshot_pow = max(0, seg_rms**2 - clean_rms**2)
        if gunshot_pow > 1e-12:
            all_snr.append(10 * np.log10(clean_rms**2 / gunshot_pow))

    # === 拼接样本 (label=2): 只取高能量段（枪声），跳过静音段 ===
    gun_mask2 = (gt == 2)
    changes = np.diff(np.concatenate([[0], gun_mask2.astype(int), [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        s_samp, e_samp = s * HOP, min(e * HOP, L)
        seg_rms = np.sqrt(np.mean(audio[s_samp:e_samp] ** 2))
        # 静音段 RMS 很低 (< 0.02), 枪声段 RMS > 0.05
        if seg_rms < 0.03:
            continue  # 跳过静音
        all_snr.append(10 * np.log10(clean_rms**2 / (seg_rms**2 + 1e-12)))

    if len(all_snr) == 0:
        return None
    return np.mean(all_snr)


def run_snr_sensitivity(args):
    """SNR 敏感性实验 — 使用全部测试集"""
    print(f"\n{'='*80}")
    print("实验2: SNR 敏感性实验（全部测试集）")
    print(f"{'='*80}")

    device = torch.device(args.device)
    model = load_model(args.model_path, device, args.vad_variant)

    manifest = json.load(open(find_manifest(args.data_dir, 'test')))
    if args.max_samples > 0:
        manifest = manifest[:args.max_samples]

    n_overlap = sum(1 for it in manifest if it.get('type') == 'overlap')
    n_concat = sum(1 for it in manifest if it.get('type') == 'concat')
    print(f"测试集: {len(manifest)} 条 (重叠 {n_overlap} + 拼接 {n_concat})")

    # SNR 分桶 — 注意拼接样本的枪声 SNR 较高（信号 vs 静音中的枪声），因而区间放宽
    buckets = {
        'very_low (<--10dB)': {'min': -50, 'max': -10,
                               'tp': np.zeros(3, dtype=np.int64), 'fp': np.zeros(3, dtype=np.int64),
                               'fn': np.zeros(3, dtype=np.int64), 'conf': np.zeros((3,3), dtype=np.int64),
                               'n': 0, 'frames': 0, 'snr_list': []},
        'low [-10,0]':        {'min': -10, 'max': 0,
                               'tp': np.zeros(3, dtype=np.int64), 'fp': np.zeros(3, dtype=np.int64),
                               'fn': np.zeros(3, dtype=np.int64), 'conf': np.zeros((3,3), dtype=np.int64),
                               'n': 0, 'frames': 0, 'snr_list': []},
        'mid [0,5]':          {'min': 0, 'max': 5,
                               'tp': np.zeros(3, dtype=np.int64), 'fp': np.zeros(3, dtype=np.int64),
                               'fn': np.zeros(3, dtype=np.int64), 'conf': np.zeros((3,3), dtype=np.int64),
                               'n': 0, 'frames': 0, 'snr_list': []},
        'high [5,15]':        {'min': 5, 'max': 15,
                               'tp': np.zeros(3, dtype=np.int64), 'fp': np.zeros(3, dtype=np.int64),
                               'fn': np.zeros(3, dtype=np.int64), 'conf': np.zeros((3,3), dtype=np.int64),
                               'n': 0, 'frames': 0, 'snr_list': []},
        'quiet (>15dB)':      {'min': 15, 'max': 200,
                               'tp': np.zeros(3, dtype=np.int64), 'fp': np.zeros(3, dtype=np.int64),
                               'fn': np.zeros(3, dtype=np.int64), 'conf': np.zeros((3,3), dtype=np.int64),
                               'n': 0, 'frames': 0, 'snr_list': []},
    }

    tp_all, fp_all, fn_all = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    conf_all = np.zeros((3, 3), dtype=np.int64)
    frames_all = 0
    skipped_no_snr = 0

    for idx, item in enumerate(manifest):
        if idx % 1000 == 0:
            print(f"  处理: {idx}/{len(manifest)} (已跳 {skipped_no_snr})")

        audio, _ = sf.read(item['audio'])
        gt = np.load(item['label'])
        feat = extract_logmel(audio)
        T = min(feat.shape[0], len(gt))
        if T < 10:
            continue
        gt = gt[:T]

        snr_est = estimate_sample_snr(audio, gt)
        if snr_est is None:
            skipped_no_snr += 1
            continue

        # 分配桶
        assigned = None
        for bname, bucket in buckets.items():
            if bucket['min'] <= snr_est < bucket['max']:
                assigned = bucket
                break
        if assigned is None:
            assigned = buckets['quiet (>15dB)']

        assigned['snr_list'].append(snr_est)
        assigned['n'] += 1

        feat_t = torch.FloatTensor(feat[:T]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feat_t)
            pred = logits[0].argmax(-1).cpu().numpy()[:T]

        compute_metrics_3class(pred, gt, assigned['conf'], assigned['tp'], assigned['fp'], assigned['fn'])
        assigned['frames'] += T

        compute_metrics_3class(pred, gt, conf_all, tp_all, fp_all, fn_all)
        frames_all += T

    print(f"\n全部测试集 (分配 {frames_all} 帧, 跳过 {skipped_no_snr} 条)")
    report_metrics(tp_all, fp_all, fn_all, conf_all, frames_all, prefix="  全部:")

    # 打印分桶报告
    print(f"\n{'='*90}")
    print("SNR 敏感性实验汇总（全部测试集，含拼接与重叠样本）")
    print(f"{'='*90}")
    print(f"{'SNR 区间':<20} {'样本数':>6} {'帧数':>10} {'SNR均值':>7} {'GFAR%':>7} {'SDR%':>7} {'F1%':>7}")
    print("-" * 90)

    snr_results = []
    for bname, bucket in buckets.items():
        if bucket['n'] == 0:
            continue
        avg_snr = np.mean(bucket['snr_list'])
        m = report_metrics(bucket['tp'], bucket['fp'], bucket['fn'], bucket['conf'], bucket['frames'],
                          prefix=f"  {bname} (n={bucket['n']}, avg SNR={avg_snr:.1f}dB)")
        m['snr_bucket'] = bname
        m['n_samples'] = bucket['n']
        m['avg_snr'] = avg_snr
        m['frames'] = bucket['frames']
        snr_results.append(m)

        print(f"{bname:<20} {bucket['n']:>6} {bucket['frames']:>10} "
              f"{avg_snr:>6.1f}dB {m['GFAR']:>6.2f} {m['SDR']:>6.2f} {m['F1']:>6.2f}")

    return snr_results


# ============================================================
# 主函数
# ============================================================

def main():
    p = argparse.ArgumentParser(description='补充实验: 通用噪声泛化 + SNR敏感性')
    p.add_argument('--data-dir', required=True, help='测试数据目录')
    p.add_argument('--model-path', required=True, help='RangeVAD-Plus 模型路径')
    p.add_argument('--device', default='cuda')
    p.add_argument('--exp', default='all', choices=['noise', 'snr', 'all'])
    p.add_argument('--clean-dir', default=None,
                   help='干净语音 wav 目录 (不指定则从测试集提取)')
    p.add_argument('--noise-types', default='white,babble,factory',
                   help='噪声类型, 逗号分隔')
    p.add_argument('--snrs', default='0,5,10,15',
                   help='SNR列表(dB), 逗号分隔')
    p.add_argument('--max-clean', type=int, default=200,
                   help='最多使用的干净语音段数')
    p.add_argument('--max-samples', type=int, default=0,
                   help='最多处理的测试样本数')
    p.add_argument('--output', default=None, help='CSV输出前缀')
    p.add_argument('--vad-variant', default=None,
                   help='消融变体名称 (no_ir/no_lstm/no_dfsmn等)，checkpoint不含config时需指定')
    args = p.parse_args()

    all_results = []

    if args.exp in ('noise', 'all'):
        noise_results = run_noise_generalization(args)
        all_results.extend(noise_results)

    if args.exp in ('snr', 'all'):
        snr_results = run_snr_sensitivity(args)
        all_results.extend(snr_results)

    # 保存 CSV
    if args.output and all_results:
        keys = list(all_results[0].keys())
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(all_results)
        print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
