#!/usr/bin/env python3
"""
评估 Silero VAD / FireRedVAD / WebRTC VAD 在枪声测试集上的帧级指标

三分类标签: 0=干净语音, 1=带噪语音, 2=非语音
二分类VAD映射: 语音(0+1) → "语音"; 非语音(2) → "非语音"

用法:
    python3 eval_baseline_vads.py \
        --data-dir /home/FunASR/FunASR-main/data/vad_train_4cls \
        --vad-type silero \
        --split test \
        --device cuda

    python3 eval_baseline_vads.py \
        --data-dir /home/FunASR/FunASR-main/data/vad_train_4cls \
        --vad-type webrtc \
        --split test
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


def timestamps_to_frame_labels(timestamps, num_frames, hop_ms=10.0):
    """将时间戳列表转为帧级二分类标签"""
    labels = np.zeros(num_frames, dtype=np.int64)
    for ts, te in timestamps:
        sf_idx = max(0, int(ts / hop_ms))
        ef_idx = min(num_frames, int(np.ceil(te / hop_ms)))
        labels[sf_idx:ef_idx] = 1
    return labels


# ==================== Silero VAD ====================
def load_silero_vad():
    try:
        from silero_vad import load_silero_vad as _load, get_speech_timestamps
        model = _load(onnx=True)
        return model, get_speech_timestamps
    except ImportError:
        print("请先安装: pip install silero-vad")
        sys.exit(1)


def silero_predict(model, get_speech_timestamps, audio, num_frames, sr=16000,
                    threshold=0.5, min_speech_ms=100, min_silence_ms=100):
    ts_list = get_speech_timestamps(
        audio, model, sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    clean = []
    for t in ts_list:
        if isinstance(t, dict):
            clean.append((t['start'], t['end']))
        else:
            clean.append((t[0], t[1]))
    return timestamps_to_frame_labels(clean, num_frames, hop_ms=10.0)


# ==================== FireRedVAD ====================
def load_firered_vad(model_dir, use_gpu=True, speech_threshold=0.4):
    try:
        from fireredvad import FireRedVad, FireRedVadConfig
    except ImportError:
        print("请先安装: pip install fireredvad")
        sys.exit(1)

    vad_config = FireRedVadConfig(
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
    vad = FireRedVad.from_pretrained(model_dir, vad_config)
    return vad


def firered_predict(vad, audio_arr, num_frames, sr=16000):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio_arr, sr, subtype='PCM_16')
        tmp_path = f.name
    result, probs = vad.detect(tmp_path)
    os.unlink(tmp_path)
    timestamps = result['timestamps']
    clean = [(s * 1000, e * 1000) for s, e in timestamps]
    return timestamps_to_frame_labels(clean, num_frames, hop_ms=10.0)


# ==================== WebRTC VAD ====================
def load_webrtc_vad(aggressiveness=2):
    try:
        import webrtcvad
        return webrtcvad.Vad(aggressiveness)
    except ImportError:
        print("请先安装: pip install webrtcvad")
        sys.exit(1)


def webrtc_predict(vad, audio_arr, num_frames, sr=16000, frame_ms=20):
    """WebRTC VAD: 逐帧预测, 转为10ms帧级标签"""
    hop_samples = int(sr * frame_ms / 1000)
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    pred = np.zeros(num_frames, dtype=np.int64)
    num_webrtc_frames = len(audio_int16) // hop_samples

    for fi in range(num_webrtc_frames):
        chunk = audio_int16[fi * hop_samples:(fi + 1) * hop_samples]
        is_speech = vad.is_speech(chunk.tobytes(), sr)
        sf_idx = fi * frame_ms // 10
        ef_idx = min(num_frames, ((fi + 1) * frame_ms + 9) // 10)
        if is_speech:
            pred[sf_idx:ef_idx] = 1
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--vad-type', required=True, choices=['silero', 'firered', 'webrtc'])
    parser.add_argument('--split', default='test')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--firered-model-dir',
                        default='/home/FunASR/FunASR-main/pretrained_models/FireRedVAD/VAD')
    # Silero params
    parser.add_argument('--silero-threshold', type=float, default=0.5)
    parser.add_argument('--silero-min-speech', type=int, default=100)
    parser.add_argument('--silero-min-silence', type=int, default=100)
    # FireRed params
    parser.add_argument('--firered-threshold', type=float, default=0.4)
    # WebRTC params
    parser.add_argument('--webrtc-agg', type=int, default=2)
    parser.add_argument('--webrtc-frame', type=int, default=20)
    args = parser.parse_args()

    # 加载数据
    manifest_path = find_manifest(args.data_dir, args.split)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if args.max_samples > 0:
        manifest = manifest[:args.max_samples]

    # 加载 VAD
    print(f"加载 {args.vad_type.upper()} VAD...")
    use_gpu = (args.device == 'cuda' and torch.cuda.is_available())
    if args.vad_type == 'silero':
        model, get_st_fn = load_silero_vad()
        predict_fn = lambda audio, nf, sr=16000: silero_predict(
            model, get_st_fn, audio, nf, sr,
            threshold=args.silero_threshold,
            min_speech_ms=args.silero_min_speech,
            min_silence_ms=args.silero_min_silence,
        )
        print(f"  threshold={args.silero_threshold}, "
              f"min_speech={args.silero_min_speech}ms, min_silence={args.silero_min_silence}ms")
    elif args.vad_type == 'webrtc':
        model = load_webrtc_vad(args.webrtc_agg)
        predict_fn = lambda audio, nf, sr=16000: webrtc_predict(
            model, audio, nf, sr, frame_ms=args.webrtc_frame)
        print(f"  aggressiveness={args.webrtc_agg}, frame={args.webrtc_frame}ms")
    else:
        model = load_firered_vad(args.firered_model_dir, use_gpu=use_gpu,
                                  speech_threshold=args.firered_threshold)
        print(f"  speech_threshold={args.firered_threshold}")
        predict_fn = lambda audio, nf, sr=16000: firered_predict(model, audio, nf)
    print("加载完成\n")

    # 统计
    class_names_gt = ['干净语音', '带噪语音', '非语音']
    total_frames = 0
    total_time = 0
    confusion = np.zeros((3, 2), dtype=np.int64)  # 真实3类 × 预测2类

    from train_vad_v5 import extract_logmel

    for item in manifest:
        audio, sr = sf.read(item['audio'])
        gt = np.load(item['label'])
        feat = extract_logmel(audio)
        T = min(len(feat), len(gt))
        if T < 10:
            continue
        gt = gt[:T]

        t0 = time.time()
        pred_binary = predict_fn(audio, T, sr)
        elapsed = time.time() - t0
        total_time += elapsed
        total_frames += T

        min_len = min(T, len(pred_binary))
        gt = gt[:min_len]
        pred_binary = pred_binary[:min_len]

        for c_gt in range(3):
            for c_pred in range(2):
                confusion[c_gt, c_pred] += ((gt == c_gt) & (pred_binary == c_pred)).sum()

    # ===== 计算指标 =====
    # 三分类: 0=干净语音, 1=带噪语音, 2=非语音
    # 预测: 0=非语音, 1=语音
    tp_speech = confusion[0, 1] + confusion[1, 1]  # 干净语音+带噪语音判为语音
    fp_speech = confusion[2, 1]  # 非语音→语音误报
    fn_speech = confusion[0, 0] + confusion[1, 0]  # 语音判为非语音
    gt_speech = confusion[0].sum() + confusion[1].sum()

    accuracy = (confusion[0, 1] + confusion[1, 1] + confusion[2, 0]) / max(total_frames, 1) * 100
    precision = tp_speech / max(tp_speech + fp_speech, 1) * 100
    recall = tp_speech / max(tp_speech + fn_speech, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    SDR = tp_speech / max(gt_speech, 1) * 100
    GFAR = fp_speech / max(confusion[2].sum(), 1) * 100

    total_audio_s = sum([
        (sf.info(item['audio']).duration) for item in manifest[:len(manifest)]
    ]) if manifest else 1
    RTF = total_time / max(total_audio_s, 1)

    print(f"\n{'='*60}")
    print(f"{args.vad_type.upper()} VAD 帧级评估 ({args.split} 集, {len(manifest)} 条, {total_frames} 帧)")
    print(f"{'='*60}")
    print(f"  GFAR (非语音→语音误报): {GFAR:.2f}%")
    print(f"  SDR  (语音检测率):      {SDR:.2f}%")
    print(f"  F1   (加权F1):          {f1:.2f}%")
    print(f"  RTF  (实时因子):        {RTF:.4f}")
    print(f"  总推理时间:             {total_time:.2f}s")

    print(f"\n{'='*60}")
    print(f"混淆矩阵 (行=真实, 列=预测)")
    print(f"{'='*60}")
    header = f"{'':>10}{'非语音':>8}{'语音':>8}"
    print(header)
    for c_gt in range(3):
        row = f"{class_names_gt[c_gt]:>10}"
        for c_pred in range(2):
            row += f"{confusion[c_gt, c_pred]:>8}"
        print(row)

    print(f"\n归一化混淆矩阵 (每行%)")
    print(header)
    for c_gt in range(3):
        row_total = confusion[c_gt].sum()
        row = f"{class_names_gt[c_gt]:>10}"
        for c_pred in range(2):
            pct = confusion[c_gt, c_pred] / max(row_total, 1) * 100
            row += f"{pct:>7.1f}%"
        print(row)

    print(f"\n{'| 指标 | 值 |':-^35}")
    print(f"{'| GFAR':<12} {'|':<4} {GFAR:>6.2f}% {'|':<4}")
    print(f"{'| SDR':<12} {'|':<4} {SDR:>6.2f}% {'|':<4}")
    print(f"{'| F1':<12} {'|':<4} {f1:>6.2f}% {'|':<4}")
    print(f"{'| RTF':<12} {'|':<4} {RTF:>6.4f} {'|':<4}")
    print(f"{'':-^35}")


if __name__ == '__main__':
    main()
