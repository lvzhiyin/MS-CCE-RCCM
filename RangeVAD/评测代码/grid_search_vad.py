#!/usr/bin/env python3
"""
Silero VAD & FireRedVAD 超参数网格搜索脚本

自动遍历多组超参数组合，评估帧级指标，输出排序后的表格

用法:
    python3 grid_search_vad.py \
        --data-dir /home/FunASR/FunASR-main/data/vad_train_4cls \
        --vad-type silero \
        --split test \
        --device cuda \
        --output results_silero_grid.csv

    python3 grid_search_vad.py \
        --data-dir /home/FunASR/FunASR-main/data/vad_train_4cls \
        --vad-type firered \
        --split test \
        --device cuda \
        --firered-model-dir ../pretrained_models/FireRedVAD/VAD \
        --output results_firered_grid.csv
"""

import os
import sys
import json
import time
import argparse
import csv
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== 工具函数 ====================
def find_manifest(data_dir, split):
    flat = Path(data_dir) / f'{split}_manifest.json'
    sub = Path(data_dir) / split / 'manifest.json'
    if flat.exists():
        return str(flat)
    if sub.exists():
        return str(sub)
    raise FileNotFoundError(f"找不到 {split} manifest: {flat} 或 {sub}")


def timestamps_to_frame_labels(timestamps, num_frames, hop_ms=10.0):
    labels = np.zeros(num_frames, dtype=np.int64)
    for ts, te in timestamps:
        sf_idx = max(0, int(ts / hop_ms))
        ef_idx = min(num_frames, int(np.ceil(te / hop_ms)))
        labels[sf_idx:ef_idx] = 1
    return labels


def compute_metrics(confusion, total_frames):
    """从混淆矩阵计算指标"""
    # confusion: [3, 2] — 真实3类(0=干净语音,1=带噪语音,2=非语音) × 预测2类(0=非语音,1=语音)
    tp_speech = confusion[0, 1] + confusion[1, 1]  # 干净语音+带噪语音判为语音
    fp_speech = confusion[2, 1]  # 非语音判为语音
    fn_speech = confusion[0, 0] + confusion[1, 0]  # 干净语音+带噪语音判为非语音

    gt_speech = confusion[0].sum() + confusion[1].sum()

    precision = tp_speech / max(tp_speech + fp_speech, 1) * 100
    recall = tp_speech / max(tp_speech + fn_speech, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    GFAR = fp_speech / max(confusion[2].sum(), 1) * 100  # 非语音→语音误报
    SDR = tp_speech / max(gt_speech, 1) * 100  # 语音检测率
    accuracy = (confusion[0, 1] + confusion[1, 1] + confusion[2, 0]) / max(total_frames, 1) * 100

    return {
        'accuracy': accuracy, 'GFAR': GFAR, 'SDR': SDR, 'F1': f1,
        'precision': precision, 'recall': recall,
        'score': SDR * 0.6 - GFAR * 0.4,
    }


# ==================== Silero VAD ====================
def load_silero():
    from silero_vad import load_silero_vad as _load, get_speech_timestamps
    return _load(onnx=True), get_speech_timestamps


SILERO_GRID = []
# 扩展搜索: ~65组, 覆盖 threshold × timing × padding 关键组合
_thr_map = {
    # threshold: 合理的 timing 范围
    0.01: [(10,20), (20,30), (20,50)],          # 极低阈值: 短hangover防过检
    0.02: [(10,20), (20,30), (20,50)],
    0.05: [(10,20), (20,30), (20,50), (30,80)],  # 低阈值: 偏召回
    0.08: [(20,30), (20,50), (30,80), (50,100)],
    0.10: [(20,30), (20,50), (30,80), (50,100)],  # 中低阈值
    0.12: [(20,30), (30,50), (50,80), (50,100)],
    0.15: [(20,30), (30,50), (50,80), (50,100)],
    0.18: [(30,50), (50,80), (50,100)],
    0.20: [(30,50), (50,80), (50,100)],           # 中阈值: 偏平衡
    0.25: [(30,50), (50,80), (50,100), (80,120)],
    0.30: [(30,50), (50,80), (50,100), (80,120)],
    0.35: [(50,80), (50,100), (80,120)],
    0.40: [(50,80), (50,100), (80,120), (80,150)],  # 高阈值: 偏精确
    0.50: [(50,80), (80,100), (80,150), (100,150)],
    0.60: [(80,100), (80,150), (100,150)],
    0.70: [(80,100), (100,150), (100,200)],
}
_pad_vals = [0, 30, 50, 80, 100]
for thr, timing_pairs in _thr_map.items():
    for (min_sp, min_si) in timing_pairs:
        for pad in _pad_vals:
            SILERO_GRID.append((thr, min_sp, min_si, pad))

def silero_grid_search(manifest, get_st_fn, model, grid, max_samples):
    results = []
    audio_cache = []
    gt_cache = []
    from train_vad_v5 import extract_logmel

    # 预加载数据
    print(f"预加载 {len(manifest[:max_samples]) if max_samples>0 else len(manifest)} 条音频...")
    items = manifest[:max_samples] if max_samples > 0 else manifest
    for item in items:
        audio, _ = sf.read(item['audio'])
        gt = np.load(item['label'])
        feat = extract_logmel(audio)
        T = min(len(feat), len(gt))
        if T >= 10:
            audio_cache.append(audio)
            gt_cache.append(gt[:T])
    print(f"有效样本: {len(audio_cache)}")

    for idx, (threshold, min_sp, min_si, pad_ms) in enumerate(grid):
        print(f"\n[{idx+1}/{len(grid)}] Silero: thr={threshold}, min_sp={min_sp}ms, min_si={min_si}ms, pad={pad_ms}ms")
        confusion = np.zeros((3, 2), dtype=np.int64)
        total_frames = 0
        t_start = time.time()

        for audio, gt in zip(audio_cache, gt_cache):
            nf = len(gt)
            ts_list = get_st_fn(
                audio, model, sampling_rate=16000,
                threshold=threshold,
                min_speech_duration_ms=min_sp,
                min_silence_duration_ms=min_si,
                speech_pad_ms=pad_ms,
            )
            clean = []
            for t in ts_list:
                if isinstance(t, dict):
                    clean.append((t['start'], t['end']))
                else:
                    clean.append((t[0], t[1]))
            pred = timestamps_to_frame_labels(clean, nf)
            min_len = min(nf, len(pred))
            gt_c = gt[:min_len]
            pred = pred[:min_len]
            for c_gt in range(3):
                for c_pred in range(2):
                    confusion[c_gt, c_pred] += ((gt_c == c_gt) & (pred == c_pred)).sum()
            total_frames += min_len

        elapsed = time.time() - t_start
        metrics = compute_metrics(confusion, total_frames)
        metrics['elapsed'] = elapsed
        total_audio_s = sum(sf.info(item['audio']).duration for item in items[:len(audio_cache)])
        metrics['RTF'] = elapsed / max(total_audio_s, 1)
        metrics['threshold'] = threshold
        metrics['min_speech_ms'] = min_sp
        metrics['min_silence_ms'] = min_si
        metrics['speech_pad_ms'] = pad_ms
        results.append(metrics)

        print(f"  GFAR={metrics['GFAR']:.1f}% SDR={metrics['SDR']:.1f}% F1={metrics['F1']:.1f}% RTF={metrics['RTF']:.4f}")

    return results


# ==================== FireRedVAD ====================
def load_firered(model_dir, use_gpu, speech_threshold):
    from fireredvad import FireRedVad, FireRedVadConfig
    config = FireRedVadConfig(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=speech_threshold,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    return FireRedVad.from_pretrained(model_dir, config)


FIRERED_GRID = []
# 扩展搜索: ~50组, 覆盖 speech_threshold × min_speech_frame × min_silence_frame
_fr_thr_map = {
    # threshold: 合理的 (min_speech_frame, min_silence_frame) 组合
    0.10: [(10,5), (10,10), (20,10)],                   # 极低阈值: 短帧
    0.15: [(10,5), (10,10), (20,10), (20,15)],
    0.20: [(10,10), (20,10), (20,15), (30,15)],          # 低阈值: 偏召回
    0.25: [(20,10), (20,15), (30,15), (30,20)],
    0.30: [(20,15), (30,15), (30,20), (40,20)],          # 中阈值: 平衡
    0.35: [(30,15), (30,20), (40,20), (40,25)],
    0.40: [(30,20), (40,20), (40,25), (50,30)],
    0.45: [(40,20), (40,25), (50,25), (50,30)],          # 中高阈值
    0.50: [(40,25), (50,25), (50,30), (50,40)],
    0.55: [(50,30), (50,40), (60,40)],                   # 高阈值: 偏精确
    0.60: [(50,30), (50,40), (60,40)],
    0.65: [(50,40), (60,40), (60,50)],
    0.70: [(60,40), (60,50)],
    0.80: [(60,50)],
}
for thr, frame_pairs in _fr_thr_map.items():
    for (min_sp, min_si) in frame_pairs:
        FIRERED_GRID.append((thr, min_sp, min_si))

def firered_grid_search(manifest, model_dir, use_gpu, grid, max_samples):
    results = []
    from train_vad_v5 import extract_logmel
    import tempfile

    print(f"预加载数据...")
    items = manifest[:max_samples] if max_samples > 0 else manifest
    audio_list = []
    gt_list = []
    for item in items:
        audio, _ = sf.read(item['audio'])
        gt = np.load(item['label'])
        feat = extract_logmel(audio)
        T = min(len(feat), len(gt))
        if T >= 10:
            audio_list.append(audio)
            gt_list.append(gt[:T])
    print(f"有效样本: {len(audio_list)}")

    for idx, (threshold, min_sp, min_si) in enumerate(grid):
        print(f"\n[{idx+1}/{len(grid)}] FireRed: thr={threshold}, min_sp={min_sp}frames, min_si={min_si}frames")

        # 加载模型
        from fireredvad import FireRedVad, FireRedVadConfig
        config = FireRedVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=5,
            speech_threshold=threshold,
            min_speech_frame=min_sp,
            max_speech_frame=2000,
            min_silence_frame=min_si,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )
        vad = FireRedVad.from_pretrained(model_dir, config)

        confusion = np.zeros((3, 2), dtype=np.int64)
        total_frames = 0
        t_start = time.time()

        for audio_arr, gt in zip(audio_list, gt_list):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio_arr, 16000, subtype='PCM_16')
                tmp_path = f.name

            result, _ = vad.detect(tmp_path)
            os.unlink(tmp_path)

            timestamps = result['timestamps']
            clean = [(s * 1000, e * 1000) for s, e in timestamps]
            nf = len(gt)
            pred = timestamps_to_frame_labels(clean, nf)
            min_len = min(nf, len(pred))
            gt = gt[:min_len]
            pred = pred[:min_len]
            for c_gt in range(3):
                for c_pred in range(2):
                    confusion[c_gt, c_pred] += ((gt == c_gt) & (pred == c_pred)).sum()
            total_frames += min_len

        elapsed = time.time() - t_start
        metrics = compute_metrics(confusion, total_frames)
        metrics['elapsed'] = elapsed
        total_audio_s = sum(sf.info(item['audio']).duration for item in items[:len(audio_list)])
        metrics['RTF'] = elapsed / max(total_audio_s, 1)
        metrics['speech_threshold'] = threshold
        metrics['min_speech_frame'] = min_sp
        metrics['min_silence_frame'] = min_si
        results.append(metrics)

        print(f"  GFAR={metrics['GFAR']:.1f}% SDR={metrics['SDR']:.1f}% F1={metrics['F1']:.1f}% RTF={metrics['RTF']:.4f}")

    return results


# ==================== WebRTC VAD ====================
WEBRTC_GRID = [
    # (aggressiveness: 0=最不激进~3=最激进, frame_ms: 10/20/30)
    (0, 10), (0, 20), (0, 30),
    (1, 10), (1, 20), (1, 30),
    (2, 10), (2, 20), (2, 30),
    (3, 10), (3, 20), (3, 30),
]


def webrtc_grid_search(manifest, grid, max_samples):
    """WebRTC VAD 网格搜索"""
    import webrtcvad
    results = []
    from train_vad_v5 import extract_logmel

    print(f"预加载 {len(manifest[:max_samples]) if max_samples>0 else len(manifest)} 条音频...")
    items = manifest[:max_samples] if max_samples > 0 else manifest
    audio_list = []
    gt_list = []
    for item in items:
        audio, _ = sf.read(item['audio'])
        gt = np.load(item['label'])
        feat = extract_logmel(audio)
        T = min(len(feat), len(gt))
        if T >= 10:
            audio_list.append(audio)
            gt_list.append(gt[:T])
    print(f"有效样本: {len(audio_list)}")

    for idx, (agg, frame_ms) in enumerate(grid):
        print(f"\n[{idx+1}/{len(grid)}] WebRTC: agg={agg}, frame={frame_ms}ms")

        vad = webrtcvad.Vad(agg)
        confusion = np.zeros((3, 2), dtype=np.int64)
        total_frames = 0
        t_start = time.time()

        hop_samples = int(16000 * frame_ms / 1000)  # 每帧采样数

        for audio, gt in zip(audio_list, gt_list):
            # 转为16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            nf = len(gt)

            # 逐帧预测
            pred = np.zeros(nf, dtype=np.int64)
            num_webrtc_frames = len(audio_int16) // hop_samples
            for fi in range(num_webrtc_frames):
                chunk = audio_int16[fi * hop_samples:(fi + 1) * hop_samples]
                is_speech = vad.is_speech(chunk.tobytes(), 16000)
                sf_idx = fi * frame_ms // 10
                ef_idx = min(nf, ((fi + 1) * frame_ms + 9) // 10)
                if is_speech:
                    pred[sf_idx:ef_idx] = 1

            min_len = min(nf, len(pred))
            gt_c = gt[:min_len]
            pred = pred[:min_len]
            for c_gt in range(3):
                for c_pred in range(2):
                    confusion[c_gt, c_pred] += ((gt_c == c_gt) & (pred == c_pred)).sum()
            total_frames += min_len

        elapsed = time.time() - t_start
        metrics = compute_metrics(confusion, total_frames)
        metrics['elapsed'] = elapsed
        total_audio_s = sum(sf.info(item['audio']).duration for item in items[:len(audio_list)])
        metrics['RTF'] = elapsed / max(total_audio_s, 1)
        metrics['aggressiveness'] = agg
        metrics['frame_ms'] = frame_ms
        results.append(metrics)

        print(f"  GFAR={metrics['GFAR']:.1f}% SDR={metrics['SDR']:.1f}% F1={metrics['F1']:.1f}% RTF={metrics['RTF']:.4f}")

    return results


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--vad-type', required=True, choices=['silero', 'firered', 'webrtc'])
    parser.add_argument('--split', default='test')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--firered-model-dir',
                        default='/home/FunASR/FunASR-main/pretrained_models/FireRedVAD/VAD')
    parser.add_argument('--output', default=None, help='CSV输出路径')
    args = parser.parse_args()

    manifest_path = find_manifest(args.data_dir, args.split)
    with open(manifest_path) as f:
        manifest = json.load(f)

    use_gpu = (args.device == 'cuda' and torch.cuda.is_available())

    if args.vad_type == 'silero':
        model, get_st_fn = load_silero()
        results = silero_grid_search(manifest, get_st_fn, model, SILERO_GRID, args.max_samples)
    elif args.vad_type == 'webrtc':
        results = webrtc_grid_search(manifest, WEBRTC_GRID, args.max_samples)
    else:
        results = firered_grid_search(manifest, args.firered_model_dir, use_gpu, FIRERED_GRID, args.max_samples)

    # ===== 输出表格 =====
    print(f"\n{'='*90}")
    print(f"{args.vad_type.upper()} VAD 超参数网格搜索结果 (按综合评分排序)")
    print(f"{'='*90}")

    # 按score排序
    results.sort(key=lambda r: r['score'], reverse=True)

    header = f"{'Rank':>4} {'GFAR%':>7} {'SDR%':>7} {'F1%':>7} {'Acc%':>7} {'RTF':>7} {'Score':>7}  Params"
    print(header)
    print("-" * 90)

    for rank, r in enumerate(results):
        if args.vad_type == 'silero':
            pstr = f"thr={r['threshold']:.2f} sp={r['min_speech_ms']}ms si={r['min_silence_ms']}ms pad={r['speech_pad_ms']}ms"
        elif args.vad_type == 'webrtc':
            pstr = f"agg={r['aggressiveness']} frame={r['frame_ms']}ms"
        else:
            pstr = f"thr={r['speech_threshold']:.2f} sp={r['min_speech_frame']}f si={r['min_silence_frame']}f"
        print(f"{rank+1:>4} {r['GFAR']:>6.1f}% {r['SDR']:>6.1f}% {r['F1']:>6.1f}% {r['accuracy']:>6.1f}% {r['RTF']:>7.4f} {r['score']:>6.1f}  {pstr}")

    # 前三名详情
    print(f"\n{'='*60}")
    print(f"TOP 3 参数组合:")
    print(f"{'='*60}")
    for rank, r in enumerate(results[:3]):
        if args.vad_type == 'silero':
            pstr = f"threshold={r['threshold']}, min_speech={r['min_speech_ms']}ms, min_silence={r['min_silence_ms']}ms, pad={r['speech_pad_ms']}ms"
            cmd = (f"python3 eval_baseline_vads.py --data-dir {args.data_dir} --vad-type silero "
                   f"--split {args.split} --device {args.device} "
                   f"--silero-threshold {r['threshold']} --silero-min-speech {r['min_speech_ms']} "
                   f"--silero-min-silence {r['min_silence_ms']}")
        elif args.vad_type == 'webrtc':
            pstr = f"aggressiveness={r['aggressiveness']}, frame_ms={r['frame_ms']}"
            cmd = (f"python3 eval_baseline_vads.py --data-dir {args.data_dir} --vad-type webrtc "
                   f"--split {args.split} --device {args.device} "
                   f"--webrtc-agg {r['aggressiveness']} --webrtc-frame {r['frame_ms']}")
        else:
            pstr = f"speech_threshold={r['speech_threshold']}, min_speech_frame={r['min_speech_frame']}, min_silence_frame={r['min_silence_frame']}"
            cmd = (f"python3 eval_baseline_vads.py --data-dir {args.data_dir} --vad-type firered "
                   f"--split {args.split} --device {args.device} "
                   f"--firered-threshold {r['speech_threshold']} "
                   f"--firered-model-dir {args.firered_model_dir}")

        print(f"  #{rank+1}: {pstr}")
        print(f"     GFAR={r['GFAR']:.1f}% SDR={r['SDR']:.1f}% F1={r['F1']:.1f}%")
        print(f"     复现命令: {cmd}")

    # 保存CSV
    if args.output:
        fieldnames = list(results[0].keys())
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
