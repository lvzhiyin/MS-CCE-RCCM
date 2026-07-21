#!/usr/bin/env python3
"""
通用噪声泛化实验 — 对比 RangeVAD-Plus / Silero VAD / FireRedVAD

测试信号: [0.2s纯噪声] [干净语音+噪声 @ SNR] [0.2s纯噪声]
Ground truth: 中间段=语音, 两侧噪声=非语音

噪声类型 (更具现实意义的5种):
  pink      — 1/f 粉红噪声 (模仿自然环境底噪: 风、雨)
  cafe      — 调制多音嘈杂声 (模仿餐厅/人群)
  street    — 低频隆隆声 + 随机脉冲 (模仿街道交通)
  machinery — 谐波机械噪声 (模仿引擎/风扇/压缩机)
  wind      — 1/f² 布朗噪声 (模仿低频闷风)

SNR: 0, 5, 10, 15 dB

用法:
    python3 eval_noise_generalization.py \
        --data-dir ../data/vad_train_4cls \
        --model-path ./vad_plus_v2_output/model.pt.best \
        --device cuda \
        --output results_noise_generalization_v2.csv
"""

import os
import sys
import json
import glob
import argparse
import csv
import tempfile
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vad_v5 import extract_logmel, find_manifest

SR = 16000
HOP_MS = 10.0
HOP = int(SR * HOP_MS / 1000)


# ============================================================
# 噪声生成
# ============================================================

def generate_noise(noise_type, length_samples):
    """程序化生成5种真实噪声"""
    n = length_samples

    if noise_type == 'pink':
        # 1/f 粉红噪声: 白噪声 → FFT → 除以 sqrt(f) → IFFT
        white = np.random.randn(n)
        X = np.fft.rfft(white)
        freqs = np.arange(1, len(X) + 1, dtype=np.float64)
        X = X / np.sqrt(freqs)
        noise = np.fft.irfft(X, n=n)
        noise = noise / np.std(noise) * 0.25
        return noise.astype(np.float32)

    elif noise_type == 'cafe':
        # 模拟餐厅嘈杂声: 8-15个随机频率的语音样正弦波 + 餐具碰撞脉冲
        n_src = np.random.randint(8, 16)
        noise = np.zeros(n, dtype=np.float32)
        t = np.arange(n, dtype=np.float32) / SR
        # 语音频带 (100-500Hz) 的随机基频 + 谐波
        for _ in range(n_src):
            f0 = np.random.uniform(100, 500)
            n_harm = np.random.randint(1, 4)
            amp = np.random.uniform(0.01, 0.04)
            # 幅度调制 (模拟语速变化)
            mod_freq = np.random.uniform(2, 6)
            env = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2*np.pi))
            for h in range(1, n_harm + 1):
                noise += amp * env * np.sin(2 * np.pi * f0 * h * t + np.random.uniform(0, 2*np.pi))
        # 添加随机脉冲 (模拟餐具碰撞)
        n_clicks = np.random.randint(2, 8)
        for _ in range(n_clicks):
            pos = np.random.randint(0, n - HOP)
            width = np.random.randint(HOP // 4, HOP)
            click = np.exp(-np.arange(width) / (width / 3))
            noise[pos:pos+width] += click * np.random.uniform(0.15, 0.3)
        noise = noise / np.std(noise) * 0.3
        return noise.astype(np.float32)

    elif noise_type == 'street':
        # 模拟街道交通: 低频隆隆 + 间歇车辆通过 + 鸣笛
        t = np.arange(n, dtype=np.float32) / SR
        noise = np.zeros(n, dtype=np.float32)
        # 持续低频隆隆 (10-80Hz, 类似引擎怠速)
        for f in [15, 25, 40, 60, 80]:
            noise += np.sin(2 * np.pi * f * t) * (80.0 / f) * 0.03
        # 车辆通过: 低频扫频 + 多普勒效果
        n_cars = np.random.randint(1, 4)
        for _ in range(n_cars):
            dur = int(np.random.uniform(1.5, 3.0) * SR)
            start = np.random.randint(0, max(1, n - dur))
            t_car = np.arange(dur, dtype=np.float32) / SR
            f_start = np.random.uniform(80, 200)
            f_end = f_start * np.random.uniform(0.7, 0.95)
            freq = f_start + (f_end - f_start) * t_car / t_car[-1]
            env = 0.15 + 0.85 * np.sin(np.pi * t_car / t_car[-1])  # 渐入渐出
            sig = env * np.sin(2 * np.pi * freq * t_car)
            noise[start:start+dur] += sig * 0.3
        noise = noise / np.std(noise) * 0.3
        return noise.astype(np.float32)

    elif noise_type == 'machinery':
        # 模拟工业机械: 多个谐波的窄带噪声 + 周期脉冲
        t = np.arange(n, dtype=np.float32) / SR
        noise = np.zeros(n, dtype=np.float32)
        # 基频 + 谐波 (模拟引擎/压缩机)
        base_freqs = [30, 50, 63, 100, 120]
        for bf in base_freqs:
            for h in [1, 2, 3, 4, 5]:
                fh = bf * h
                if fh > SR / 2:
                    break
                amp = 0.15 / h  # 谐波衰减
                phase = np.random.uniform(0, 2 * np.pi)
                noise += amp * np.sin(2 * np.pi * fh * t + phase)
        # 周期脉冲 (模拟活塞/冲压)
        pulse_period = int(np.random.uniform(0.5, 2.0) * SR)
        for start in range(0, n, pulse_period):
            pos = start + np.random.randint(-HOP, HOP)
            if 0 <= pos < n - HOP // 2:
                width = HOP // 4
                pulse = np.exp(-np.arange(width) / (width / 4))
                noise[pos:pos+width] += pulse * 0.25
        noise = noise / np.std(noise) * 0.3
        return noise.astype(np.float32)

    elif noise_type == 'wind':
        # 1/f² 布朗噪声 (模拟闷风、气流) — 对白噪声做两次积分
        white = np.random.randn(n).astype(np.float64)
        brown = np.cumsum(np.cumsum(white))
        brown = brown - np.mean(brown)
        brown = brown / np.std(brown) * 0.3
        return brown.astype(np.float32)

    else:
        raise ValueError(f"未知噪声: {noise_type}")


def mix_at_snr(signal, noise, snr_db):
    sig_pow = np.mean(signal ** 2) + 1e-12
    noise_pow = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(sig_pow / noise_pow * (10 ** (-snr_db / 10)))
    return signal + noise * scale


# ============================================================
# RangeVAD-Plus
# ============================================================

def load_rangevad(model_path, device):
    from RangeVAD_Plus import RangeVADPlus
    ck = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ck['config']
    model = RangeVADPlus(mel_dim=80, hidden_dim=cfg['hidden_dim'], num_classes=3,
                         dfsmn_blocks=cfg['dfsmn_blocks'],
                         look_back=cfg.get('look_back', 20)).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model


# ============================================================
# Silero VAD
# ============================================================

def load_silero():
    from silero_vad import load_silero_vad as _load, get_speech_timestamps
    model = _load(onnx=True)
    return model, get_speech_timestamps


# ============================================================
# FireRedVAD
# ============================================================

def load_firered(model_dir, use_gpu, speech_threshold=0.4):
    from fireredvad import FireRedVad, FireRedVadConfig
    cfg = FireRedVadConfig(
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
    return FireRedVad.from_pretrained(model_dir, cfg)


# ============================================================
# 评估主逻辑
# ============================================================

def evaluate_rangevad(model, device, all_audios, all_clean_lens, all_gt_labels):
    """RangeVAD-Plus 三分类评估"""
    tp = np.zeros(3, dtype=np.int64)
    fp = np.zeros(3, dtype=np.int64)
    fn = np.zeros(3, dtype=np.int64)
    confusion = np.zeros((3,3), dtype=np.int64)
    total = 0

    for audio, clean_len, gt in zip(all_audios, all_clean_lens, all_gt_labels):
        feat = extract_logmel(audio)
        nf = min(feat.shape[0], len(gt))
        gt = gt[:nf]
        feat_t = torch.FloatTensor(feat[:nf]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(feat_t).argmax(-1).cpu().numpy()[:nf]

        for c in range(3):
            tp[c] += ((pred==c) & (gt==c)).sum()
            fp[c] += ((pred==c) & (gt!=c)).sum()
            fn[c] += ((gt==c) & (pred!=c)).sum()
        for cg in range(3):
            for cp in range(3):
                confusion[cg,cp] += ((gt==cg) & (pred==cp)).sum()
        total += nf

    GFAR = (confusion[2,0]+confusion[2,1]) / max(confusion[2].sum(), 1) * 100
    SDR = (tp[0]+tp[1]) / max(tp[0]+tp[1]+fn[0]+fn[1], 1) * 100
    f1t = 0
    for c in range(3):
        pc = tp[c]/max(tp[c]+fp[c], 1)
        rc = tp[c]/max(tp[c]+fn[c], 1)
        f1t += 2*pc*rc/max(pc+rc,1e-10) * (tp[c]+fn[c])
    F1 = f1t / max(total, 1) * 100
    return GFAR, SDR, F1, confusion


def evaluate_silero(silero_model, get_st, all_audios, all_clean_lens, all_gt_labels,
                    thr=0.5, min_sp=100, min_si=100):
    """Silero VAD 二分类评估 → 映射到 SDR + 噪声FA"""
    total_speech_frames = 0  # 真正语音帧数
    detected_speech_frames = 0  # 检测到的语音帧数
    total_noise_frames = 0  # 真正噪声帧数
    noise_false_alarm = 0  # 噪声帧误判为语音

    for audio, clean_len, gt in zip(all_audios, all_clean_lens, all_gt_labels):
        nf = len(gt)
        ts_list = get_st(audio, silero_model, sampling_rate=SR,
                         threshold=thr,
                         min_speech_duration_ms=min_sp,
                         min_silence_duration_ms=min_si)
        # 转为帧标签
        pred_bin = np.zeros(nf, dtype=np.int64)
        for t in ts_list:
            s = max(0, int(t['start'] / HOP_MS))
            e = min(nf, int(np.ceil(t['end'] / HOP_MS)))
            pred_bin[s:e] = 1

        gt_bin = (gt != 2).astype(np.int64)  # 三分类→二分类: 0,1→语音; 2→非语音
        total_speech_frames += gt_bin.sum()
        detected_speech_frames += (pred_bin * gt_bin).sum()
        total_noise_frames += (1 - gt_bin).sum()
        noise_false_alarm += (pred_bin * (1 - gt_bin)).sum()

    SDR = detected_speech_frames / max(total_speech_frames, 1) * 100
    NFA = noise_false_alarm / max(total_noise_frames, 1) * 100
    return SDR, NFA


def evaluate_firered(vad, all_audios, all_clean_lens, all_gt_labels):
    """FireRedVAD 二分类评估"""
    total_speech_frames = 0
    detected_speech_frames = 0
    total_noise_frames = 0
    noise_false_alarm = 0

    for audio, clean_len, gt in zip(all_audios, all_clean_lens, all_gt_labels):
        nf = len(gt)
        # FireRed 需要写临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, SR, subtype='PCM_16')
            tmp = f.name
        result, _ = vad.detect(tmp)
        os.unlink(tmp)

        pred_bin = np.zeros(nf, dtype=np.int64)
        for s, e in result['timestamps']:
            s_f = max(0, int(s * 1000 / HOP_MS))
            e_f = min(nf, int(np.ceil(e * 1000 / HOP_MS)))
            pred_bin[s_f:e_f] = 1

        gt_bin = (gt != 2).astype(np.int64)
        total_speech_frames += gt_bin.sum()
        detected_speech_frames += (pred_bin * gt_bin).sum()
        total_noise_frames += (1 - gt_bin).sum()
        noise_false_alarm += (pred_bin * (1 - gt_bin)).sum()

    SDR = detected_speech_frames / max(total_speech_frames, 1) * 100
    NFA = noise_false_alarm / max(total_noise_frames, 1) * 100
    return SDR, NFA


# ============================================================
# 主函数
# ============================================================

def main():
    p = argparse.ArgumentParser(description='通用噪声泛化实验 — 三模型对比')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--model-path', required=True, help='RangeVAD-Plus 权重路径')
    p.add_argument('--device', default='cuda')
    p.add_argument('--noise-types', default='pink,cafe,street,machinery,wind')
    p.add_argument('--snrs', default='0,5,10,15')
    p.add_argument('--max-clean', type=int, default=200)
    p.add_argument('--output', default=None, help='CSV 输出')
    p.add_argument('--skip-silero', action='store_true', help='跳过Silero评估')
    p.add_argument('--skip-firered', action='store_true', help='跳过FireRed评估')
    p.add_argument('--firered-model-dir',
                   default='/home/FunASR/FunASR-main/pretrained_models/FireRedVAD/VAD')
    p.add_argument('--silero-threshold', type=float, default=0.3)
    p.add_argument('--firered-threshold', type=float, default=0.4)
    args = p.parse_args()

    noise_types = args.noise_types.split(',')
    snr_list = [int(s) for s in args.snrs.split(',')]
    device = torch.device(args.device)

    # ============ 1. 提取干净语音段 ============
    print("提取干净语音段...")
    manifest = json.load(open(find_manifest(args.data_dir, 'test')))
    clean_audios = []
    for item in manifest[:args.max_clean * 3]:
        if item.get('type') != 'overlap':
            continue
        audio, _ = sf.read(item['audio'])
        gt = np.load(item['label'])
        T = min(extract_logmel(audio).shape[0], len(gt))
        if T < 10:
            continue
        gt = gt[:T]
        clean_mask = (gt == 0)
        # 提取连续干净语音段
        segments = []
        in_seg, seg_start = False, 0
        for i, v in enumerate(clean_mask):
            if v and not in_seg:
                seg_start = i; in_seg = True
            elif not v and in_seg:
                if (i - seg_start) * HOP >= SR * 0.5:
                    segments.append((seg_start*HOP, i*HOP))
                in_seg = False
        if in_seg:
            if (len(clean_mask) - seg_start) * HOP >= SR * 0.5:
                segments.append((seg_start*HOP, len(clean_mask)*HOP))
        for s, e in segments:
            clean_audios.append(audio[s:e].astype(np.float32))
        if len(clean_audios) >= args.max_clean:
            break

    print(f"提取 {len(clean_audios)} 段纯语音 (>=0.5s)")

    # ============ 2. 加载模型 ============
    print("\n加载模型...")
    rvad = load_rangevad(args.model_path, device)
    print("  RangeVAD-Plus ✓")

    if not args.skip_silero:
        silero_model, silero_st = load_silero()
        print("  Silero VAD ✓")
    else:
        silero_model = silero_st = None

    if not args.skip_firered:
        use_gpu = (args.device == 'cuda' and torch.cuda.is_available())
        firered_vad = load_firered(args.firered_model_dir, use_gpu, args.firered_threshold)
        print("  FireRedVAD ✓")
    else:
        firered_vad = None

    # ============ 3. 逐噪声逐SNR评估 ============
    all_results = []

    for noise_type in noise_types:
        for snr_db in snr_list:
            print(f"\n{'─'*70}")
            print(f"  {noise_type} @ {snr_db}dB")
            print(f"{'─'*70}")

            # 3.1 构造测试样本
            audios_list = []
            clean_lens_list = []
            gt_labels_list = []
            pad_s = int(0.2 * SR)

            for clean in clean_audios:
                noise_full = generate_noise(noise_type, len(clean) + 2 * pad_s)
                noisy_speech = mix_at_snr(clean, noise_full[pad_s:pad_s+len(clean)], snr_db)
                audio = np.concatenate([
                    noise_full[:pad_s] * 0.5,
                    noisy_speech,
                    noise_full[pad_s+len(clean):] * 0.5
                ])
                # GT: 两侧噪声=2, 中间语音段=0
                nf = extract_logmel(audio).shape[0]
                sp_s = int(0.2 * 1000 / HOP_MS)
                sp_e = sp_s + int(len(clean)/SR * 1000/HOP_MS)
                sp_e = min(sp_e, nf)
                gt = np.full(nf, 2, dtype=np.int64)
                gt[sp_s:sp_e] = 0

                audios_list.append(audio)
                clean_lens_list.append(len(clean))
                gt_labels_list.append(gt)

            n_samples = len(audios_list)

            # 3.2 评估 RangeVAD-Plus
            gf, sd, f1, conf = evaluate_rangevad(rvad, device, audios_list, clean_lens_list, gt_labels_list)
            # 分类分布
            total_pred = conf.sum()
            noise_as_clean = conf[2,0]/max(conf[2].sum(),1)*100
            noise_as_noisy = conf[2,1]/max(conf[2].sum(),1)*100
            noise_as_nonsp = conf[2,2]/max(conf[2].sum(),1)*100
            print(f"  RangeVAD-Plus | SDR={sd:.1f}%  GFAR={gf:.1f}%  F1={f1:.1f}%")
            print(f"    噪声帧分布: →干净{noise_as_clean:.1f}% →带噪{noise_as_noisy:.1f}% →非语音{noise_as_nonsp:.1f}%")
            all_results.append({
                'noise': noise_type, 'snr_dB': snr_db, 'model': 'RangeVAD-Plus',
                'SDR': sd, 'GFAR': gf, 'F1': f1,
                'noise→clean': noise_as_clean, 'noise→noisy': noise_as_noisy,
                'noise→nonsp': noise_as_nonsp, 'n_samples': n_samples,
            })

            # 3.3 评估 Silero
            if silero_model is not None:
                ssd, snfa = evaluate_silero(silero_model, silero_st, audios_list,
                                            clean_lens_list, gt_labels_list,
                                            thr=args.silero_threshold)
                print(f"  Silero VAD    | SDR={ssd:.1f}%  噪声FA={snfa:.1f}%")
                all_results.append({
                    'noise': noise_type, 'snr_dB': snr_db, 'model': 'Silero VAD',
                    'SDR': ssd, 'NFA_2class': snfa, 'n_samples': n_samples,
                })

            # 3.4 评估 FireRed
            if firered_vad is not None:
                fsd, fnfa = evaluate_firered(firered_vad, audios_list, clean_lens_list, gt_labels_list)
                print(f"  FireRed VAD   | SDR={fsd:.1f}%  噪声FA={fnfa:.1f}%")
                all_results.append({
                    'noise': noise_type, 'snr_dB': snr_db, 'model': 'FireRed VAD',
                    'SDR': fsd, 'NFA_2class': fnfa, 'n_samples': n_samples,
                })

    # ============ 4. 汇总表格 ============
    print(f"\n{'='*80}")
    print("通用噪声泛化实验 — 三模型对比汇总")
    print(f"{'='*80}")

    # 分组打印
    for noise_type in noise_types:
        print(f"\n--- {noise_type} ---")
        print(f"  {'SNR':<6} {'RangeVAD SDR':>12} {'RangeVAD GFAR':>13} {'Silero SDR':>10} {'Silero FA':>9} {'FireRed SDR':>11} {'FireRed FA':>10}")
        print(f"  {'─'*6} {'─'*12} {'─'*13} {'─'*10} {'─'*9} {'─'*11} {'─'*10}")
        for snr_db in snr_list:
            rv = [r for r in all_results if r['noise']==noise_type and r['snr_dB']==snr_db]
            rv_r = next((r for r in rv if r['model']=='RangeVAD-Plus'), {})
            rv_s = next((r for r in rv if r['model']=='Silero VAD'), {})
            rv_f = next((r for r in rv if r['model']=='FireRed VAD'), {})
            print(f"  {snr_db:<5}dB "
                  f"{rv_r.get('SDR',0):>10.1f}% {rv_r.get('GFAR',0):>11.1f}% "
                  f"{rv_s.get('SDR',0):>8.1f}% {rv_s.get('NFA_2class',0):>7.1f}% "
                  f"{rv_f.get('SDR',0):>9.1f}% {rv_f.get('NFA_2class',0):>8.1f}%")

    # 保存
    if args.output:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['noise','snr_dB','model','SDR','GFAR','F1',
                                              'NFA_2class','noise→clean','noise→noisy',
                                              'noise→nonsp','n_samples'], extrasaction='ignore')
            w.writeheader()
            w.writerows(all_results)
        print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
