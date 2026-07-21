#!/usr/bin/env python3
"""
不同枪声类型鲁棒性实验

策略: 对所有测试样本的枪声段直接提取声学特征, 聚类分型, 
       然后按类型报告 GFAR/SDR/F1。

用法:
    python3 eval_guntype_robustness.py \
        --data-dir ../data/vad_train_4cls \
        --model-path ./vad_plus_v2_output/model.pt.best \
        --device cuda \
        --output results_guntype.csv
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
import torch
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_vad_v5 import extract_logmel, find_manifest

SR = 16000
HOP_MS = 10.0
HOP = int(SR * HOP_MS / 1000)


# ============================================================
# 工具
# ============================================================

def load_model(model_path, device, vad_variant=None):
    """加载 VAD 模型（支持标准模型和消融变体）"""
    ck = torch.load(model_path, map_location=device, weights_only=False)
    if vad_variant:
        from run_ablation import AblationVAD
        model = AblationVAD(vad_variant, mel_dim=80, hidden_dim=64, num_classes=3).to(device)
    elif 'config' in ck:
        cfg = ck['config']
        model = RangeVADPlus(mel_dim=80, hidden_dim=cfg['hidden_dim'], num_classes=3,
                             dfsmn_blocks=cfg['dfsmn_blocks'],
                             look_back=cfg.get('look_back', 20)).to(device)
    else:
        raise RuntimeError("Checkpoint不含config，请指定 --vad-variant (如 no_ir, no_lstm)")
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model


def extract_gunsegment_features(audio, start_samp, end_samp):
    """从一段枪声音频中提取 6 维声学特征"""
    seg = audio[start_samp:end_samp].astype(np.float32)
    if len(seg) < HOP:
        return None

    duration_ms = len(seg) / SR * 1000
    rms = np.sqrt(np.mean(seg**2) + 1e-12)
    peak = np.max(np.abs(seg))

    spec = np.abs(librosa.stft(seg, n_fft=512, hop_length=HOP))
    centroid = np.mean(librosa.feature.spectral_centroid(S=spec, sr=SR)[0])
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=spec, sr=SR)[0])

    # 脉冲数
    peaks = librosa.util.peak_pick(seg, pre_max=3, post_max=3,
                                    pre_avg=3, post_avg=3, delta=0.3, wait=3)
    n_peaks = len(peaks)

    return np.array([duration_ms, rms, peak, centroid, bandwidth, n_peaks], dtype=np.float32)


def compute_metrics(pred, gt):
    tp = np.zeros(3, dtype=np.int64)
    fp = np.zeros(3, dtype=np.int64)
    fn = np.zeros(3, dtype=np.int64)
    confusion = np.zeros((3,3), dtype=np.int64)
    for c in range(3):
        tp[c] += ((pred==c) & (gt==c)).sum()
        fp[c] += ((pred==c) & (gt!=c)).sum()
        fn[c] += ((gt==c) & (pred!=c)).sum()
    for cg in range(3):
        for cp in range(3):
            confusion[cg,cp] += ((gt==cg) & (pred==cp)).sum()
    return tp, fp, fn, confusion


def report(tp, fp, fn, confusion):
    total = confusion.sum()
    gf = (confusion[2,0]+confusion[2,1]) / max(confusion[2].sum(), 1) * 100
    sd = (tp[0]+tp[1]) / max(tp[0]+tp[1]+fn[0]+fn[1], 1) * 100
    f1t = 0
    for c in range(3):
        pc = tp[c]/max(tp[c]+fp[c],1)
        rc = tp[c]/max(tp[c]+fn[c],1)
        f1t += 2*pc*rc/max(pc+rc,1e-10) * (tp[c]+fn[c])
    f1v = f1t/max(total,1)*100
    return {'GFAR':gf, 'SDR':sd, 'F1':f1v, 'frames':int(total)}


# ============================================================
# 主逻辑
# ============================================================

def main():
    p = argparse.ArgumentParser(description='不同枪声类型鲁棒性实验')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--n-clusters', type=int, default=3)
    p.add_argument('--output', default=None)
    p.add_argument('--vad-variant', default=None,
                   help='消融变体名称 (no_ir/no_lstm/no_dfsmn等)，checkpoint不含config时需指定')
    args = p.parse_args()

    # ====== Step 1: 跑全测试集推理 + 收集每个样本枪声段特征 ======    
    print("=" * 60)
    print("Step 1: 全测试集推理 + 枪声段特征收集")
    print("=" * 60)

    device = torch.device(args.device)
    model = load_model(args.model_path, device, args.vad_variant)

    manifest = json.load(open(find_manifest(args.data_dir, 'test')))
    print(f"测试集: {len(manifest)} 条")

    sample_features = []   # 每个样本的平均枪声特征
    sample_clusters = []   # 聚类后填充
    per_sample_metrics = []  # (tp, fp, fn, confusion) per sample

    tp_all, fp_all, fn_all = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    conf_all = np.zeros((3,3), dtype=np.int64)

    for idx, item in enumerate(manifest):
        if idx % 1000 == 0:
            print(f"  处理: {idx}/{len(manifest)}")

        audio, _ = sf.read(item['audio'])
        gt = np.load(item['label'])
        feat = extract_logmel(audio)
        T = min(feat.shape[0], len(gt))
        if T < 10:
            continue
        gt = gt[:T]

        # 推理
        feat_t = torch.FloatTensor(feat[:T]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feat_t)
            pred = logits[0].argmax(-1).cpu().numpy()[:T]

        tpi, fpi, fni, confi = compute_metrics(pred, gt)
        per_sample_metrics.append((tpi, fpi, fni, confi))

        tp_all += tpi
        fp_all += fpi
        fn_all += fni
        conf_all += confi

        # 从标签中提取枪声段特征
        # label=1 (重叠中的枪声) 和 label=2 (纯枪声)
        gun_mask = (gt == 1) | (gt == 2)
        changes = np.diff(np.concatenate([[0], gun_mask.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        seg_feats = []
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            s_samp, e_samp = s * HOP, min(e * HOP, len(audio))
            feats = extract_gunsegment_features(audio, s_samp, e_samp)
            if feats is not None:
                seg_feats.append(feats)

        if len(seg_feats) == 0:
            sample_features.append(None)
        else:
            sample_features.append(np.mean(seg_feats, axis=0))

    # 过滤无效样本
    valid_indices = [i for i, f in enumerate(sample_features) if f is not None]
    valid_features = np.array([sample_features[i] for i in valid_indices], dtype=np.float32)

    print(f"\n有效枪声样本: {len(valid_features)} (共 {len(manifest)} 条)")
    r = report(tp_all, fp_all, fn_all, conf_all)
    print(f"全测试集: GFAR={r['GFAR']:.2f}% SDR={r['SDR']:.2f}% F1={r['F1']:.2f}%")

    # ====== Step 2: K-means 聚类 ======
    if len(valid_features) < args.n_clusters * 10:
        print(f"\n样本不足 (需要 {args.n_clusters * 10}), 跳过聚类")
        return

    print(f"\n{'='*60}")
    print(f"Step 2: {args.n_clusters} 类 K-means 聚类")
    print("=" * 60)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    scaler = StandardScaler()
    X = scaler.fit_transform(valid_features)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)

    # 聚类统计 + 命名
    cluster_stats = []
    for c in range(args.n_clusters):
        mask = labels == c
        c_X = valid_features[mask]
        cluster_stats.append({
            'id': c,
            'n': mask.sum(),
            'duration': c_X[:,0].mean(),
            'rms': c_X[:,1].mean(),
            'peak': c_X[:,2].mean(),
            'centroid': c_X[:,3].mean(),
            'bandwidth': c_X[:,4].mean(),
            'n_peaks': c_X[:,5].mean(),
        })

    # 按时长排序
    cluster_stats.sort(key=lambda x: x['duration'])
    type_names = ['短脉冲型', '中等脉冲型', '长持续/连发型'][:args.n_clusters]
    for i, cs in enumerate(cluster_stats):
        cs['name'] = type_names[i]

    # ====== Step 3: 按聚类分组评估 ======
    print(f"\n{'='*60}")
    print("Step 3: 按枪声类型分组评估")
    print("=" * 60)

    cluster_m = {}
    for cs in cluster_stats:
        cluster_m[cs['id']] = {
            'tp': np.zeros(3, dtype=np.int64),
            'fp': np.zeros(3, dtype=np.int64),
            'fn': np.zeros(3, dtype=np.int64),
            'confusion': np.zeros((3,3), dtype=np.int64),
        }

    label_dict = {vi: l for vi, l in zip(valid_indices, labels)}
    for i, (tpi, fpi, fni, confi) in enumerate(per_sample_metrics):
        c = label_dict.get(i)
        if c is None:
            continue
        m = cluster_m[c]
        m['tp'] += tpi
        m['fp'] += fpi
        m['fn'] += fni
        m['confusion'] += confi

    # ====== Step 4: 报告 ======
    print(f"\n{'='*90}")
    print("不同枪声类型鲁棒性实验")
    print(f"{'='*90}")
    print(f"{'枪声类型':<20} {'样本数':>6} {'时长(ms)':>9} {'峰值':>7} {'频谱中心':>9} {'峰数':>5} {'GFAR%':>7} {'SDR%':>7} {'F1%':>7}")
    print("-" * 90)

    results = []
    for cs in cluster_stats:
        m = cluster_m[cs['id']]
        r = report(m['tp'], m['fp'], m['fn'], m['confusion'])
        print(f"{cs['name']:<20} {cs['n']:>6} {cs['duration']:>8.0f}ms "
              f"{cs['peak']:>6.3f} {cs['centroid']:>8.0f}Hz {cs['n_peaks']:>4.1f} "
              f"{r['GFAR']:>6.2f} {r['SDR']:>6.2f} {r['F1']:>6.2f}")
        results.append({**cs, **r})

    # 保存
    if args.output:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys(), extrasaction='ignore')
            w.writeheader()
            w.writerows(results)
        print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
