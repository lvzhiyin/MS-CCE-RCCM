#!/usr/bin/env python3
"""
RangeVAD V6 + MS-CCE ParaformerStreaming 端到端测试
与 test_baseline.py 保持相同 ASR 配置 (chunk=0,6,4, lfr_n=4, CCE k=1,3,5, w=0.8)

用法:
    python rccm/test_vad_asr.py \
        --test-data /home/FunASR/FunASR-main/data/gunshot_30h/test/test_nopunct.jsonl \
        --vad-model rccm/RangeVAD_V6_model.pt \
        --asr-init-param outputs_ms_cce_mixed_k135_n4/model.pt.best \
        --device cuda
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_cer(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1] + (0 if ref[i-1]==hyp[j-1] else 1))
    return dp[m][n] / len(ref)


def load_rangevad(model_path, device='cpu'):
    """自动识别 V5 或 Plus 模型并加载"""
    from train_vad_v5 import BiLSTMVAD_v5, extract_logmel
    ck = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ck['config']

    # 判断模型类型: Plus 有 dfsmn_blocks 字段, V5 有 num_layers 字段
    if 'dfsmn_blocks' in cfg:
        from RangeVAD_Plus import RangeVADPlus
        m = RangeVADPlus(
            mel_dim=80,
            hidden_dim=cfg['hidden_dim'],
            num_classes=4,
            dfsmn_blocks=cfg['dfsmn_blocks'],
            look_back=cfg.get('look_back', 20)
        )
    else:
        m = BiLSTMVAD_v5(
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            use_freq_attn=cfg.get('use_freq_attn', True)
        )

    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, extract_logmel


def get_speech_segments(vad, extract_logmel, audio, sr=16000):
    feat = extract_logmel(audio)
    vad_device = next(vad.parameters()).device
    with torch.no_grad():
        logits = vad(torch.FloatTensor(feat).unsqueeze(0).to(vad_device))
    preds = logits[0].argmax(-1).cpu().numpy()
    # 4分类: 0=拼接纯枪声→切, 1=语音→留, 2=静音→切, 3=重叠枪声→留
    segs = []
    in_sp = False
    start = 0
    for t in range(len(preds)):
        if preds[t] in (1, 3) and not in_sp:
            in_sp = True
            start = t
        elif preds[t] not in (1, 3) and in_sp:
            in_sp = False
            if t - start >= 5:
                segs.append((start * 10, min(t * 10, len(audio) * 1000 // sr)))
    if in_sp:
        segs.append((start * 10, len(audio) * 1000 // sr))
    return segs


def build_asr_model(args):
    """构建与 test_baseline.py 一致的 ASR 模型"""
    model_kwargs = {
        "model": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
        "model_class": args.model_class,
        "init_param": args.asr_init_param,
    }
    if args.use_cce:
        model_kwargs.update({
            "use_cce": True,
            "cce_kernel_sizes": [int(x) for x in args.cce_kernel_sizes.split(',')],
            "cce_chunk_size": args.cce_chunk_size,
            "cce_fixed_weight": args.cce_fixed_weight,
        })

    model = AutoModel(**model_kwargs)

    # 运行时修改 LFR n 参数
    frontend_obj = None
    if hasattr(model, 'frontend'):
        frontend_obj = model.frontend
    elif hasattr(model, 'kwargs') and 'frontend' in model.kwargs:
        frontend_obj = model.kwargs['frontend']

    if frontend_obj is not None and args.lfr_n != frontend_obj.lfr_n:
        old_n = frontend_obj.lfr_n
        frontend_obj.lfr_n = args.lfr_n
        if hasattr(model, 'kwargs') and 'frontend' in model.kwargs:
            model.kwargs['frontend'].lfr_n = args.lfr_n
        if hasattr(model, 'frontend'):
            model.frontend.lfr_n = args.lfr_n
        if hasattr(model, '_base_kwargs_map') and 'kwargs' in model._base_kwargs_map:
            if 'frontend' in model._base_kwargs_map['kwargs']:
                model._base_kwargs_map['kwargs']['frontend'].lfr_n = args.lfr_n
        print(f"  LFR_n: {old_n} -> {args.lfr_n} (首字延迟 {args.lfr_n*10}ms/帧)")

    # 强制设置注入权重
    if args.use_cce and hasattr(model, 'cce_module'):
        model.cce_module.fixed_weight = args.cce_fixed_weight
        print(f"  cce_fixed_weight 已强制设置为: {args.cce_fixed_weight}")

    return model


def asr_infer(asr_model, audio_f32, chunk_size):
    """ASR 推理，接收 f32 音频数组"""
    tmp_path = f"/tmp/asr_{os.getpid()}_{int(time.time()*1000)%1000000}.wav"
    sf.write(tmp_path, audio_f32, 16000, subtype='PCM_16')
    try:
        res = asr_model.generate(input=tmp_path, batch_size=1, chunk_size=chunk_size)
        text = res[0]['text'].strip() if res and res[0].get('text') else ''
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', required=True, help='JSONL测试数据路径')
    parser.add_argument('--vad-model', required=True, help='RangeVAD V6权重路径')
    parser.add_argument('--asr-init-param', required=True, help='ASR微调权重路径')
    parser.add_argument('--model-class', type=str, default='ParaformerStreamingMSCce',
                        help='ASR模型类名')
    parser.add_argument('--use-cce', action='store_true', default=False,
                        help='启用MS-CCE')
    parser.add_argument('--cce-kernel-sizes', type=str, default='1,3,5',
                        help='MS-CCE卷积核大小，逗号分隔')
    parser.add_argument('--cce-chunk-size', type=int, default=16,
                        help='CCE注入chunk大小')
    parser.add_argument('--cce-fixed-weight', type=float, default=0.8,
                        help='CCE固定注入权重')
    parser.add_argument('--chunk-size', type=str, default='0,6,4',
                        help='chunk配置，逗号分隔')
    parser.add_argument('--lfr-n', type=int, default=4,
                        help='LFR n参数')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int, default=0, help='0=全部')
    parser.add_argument('--output', default=None, help='结果JSON路径')
    parser.add_argument('--no-vad', action='store_true', help='只测无VAD')
    parser.add_argument('--only-vad', action='store_true', help='只测有VAD')
    args = parser.parse_args()

    chunk_size = [int(x) for x in args.chunk_size.split(',')]
    cce_kernel_sizes = [int(x) for x in args.cce_kernel_sizes.split(',')]

    # 加载测试数据
    data = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({'source': item['source'], 'target': item['target']})
    if args.max_samples > 0:
        data = data[:args.max_samples]

    # 加载 VAD
    print("加载 RangeVAD V6...")
    vad, mel_fn = load_rangevad(args.vad_model, args.device)
    print("RangeVAD V6 加载完成\n")

    # 加载 ASR
    print("加载 ASR...")
    asr = build_asr_model(args)
    print("ASR 加载完成\n")

    results_noVAD = []
    results_VAD = []
    total_cer_noVAD = 0.0
    total_cer_VAD = 0.0
    total_time_noVAD = 0.0
    total_time_VAD = 0.0

    cce_str = "有CCE" if args.use_cce else "无CCE"
    print(f"测试 {len(data)} 条 — RangeVAD + {args.model_class} ({cce_str})")
    print(f"Chunk: {chunk_size}, LFR_n: {args.lfr_n}, CCE k={args.cce_kernel_sizes}, w={args.cce_fixed_weight}")
    print()

    for idx, item in enumerate(data):
        wav_path = item['source']
        ref = item['target']

        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio.mean(1)
        audio = audio.astype(np.float32)

        # 无VAD
        if not args.only_vad:
            t0 = time.time()
            hyp1 = asr_infer(asr, audio, chunk_size)
            t1 = time.time() - t0
            cer1 = compute_cer(ref, hyp1)
            results_noVAD.append({'ref': ref, 'hyp': hyp1, 'cer': cer1, 'time': t1})
            total_cer_noVAD += cer1
            total_time_noVAD += t1

        # 有VAD
        if not args.no_vad:
            t0 = time.time()
            segs = get_speech_segments(vad, mel_fn, audio)
            if segs:
                parts = [audio[int(s / 1000 * sr):int(e / 1000 * sr)] for s, e in segs]
                hyp2 = asr_infer(asr, np.concatenate(parts).astype(np.float32), chunk_size)
            else:
                hyp2 = ''
            t2 = time.time() - t0
            cer2 = compute_cer(ref, hyp2)
            results_VAD.append({'ref': ref, 'hyp': hyp2, 'cer': cer2, 'time': t2, 'segs': len(segs)})
            total_cer_VAD += cer2
            total_time_VAD += t2

        # 打印
        if args.only_vad:
            print(f'[{idx}] ref={ref}  hyp_VAD={results_VAD[-1]["hyp"]}  cer={results_VAD[-1]["cer"]*100:.1f}%')
        elif args.no_vad:
            print(f'[{idx}] ref={ref}  hyp={results_noVAD[-1]["hyp"]}  cer={results_noVAD[-1]["cer"]*100:.1f}%')
        else:
            print(f'[{idx}] ref={ref}')
            print(f'      无VAD: {results_noVAD[-1]["hyp"]}  (CER={results_noVAD[-1]["cer"]*100:.1f}%)')
            print(f'      +VAD:  {results_VAD[-1]["hyp"]}  (CER={results_VAD[-1]["cer"]*100:.1f}%)  segs={results_VAD[-1]["segs"]}')

        if (idx + 1) % 100 == 0:
            n_done = idx + 1
            cer_no = total_cer_noVAD / n_done * 100 if not args.only_vad else 0
            cer_va = total_cer_VAD / n_done * 100 if not args.no_vad else 0
            print(f'  [{n_done}/{len(data)}] 无VAD_CER={cer_no:.1f}%  VAD_CER={cer_va:.1f}%')

    n = len(data)
    print(f"\n{'='*60}")
    print("最终结果")
    print('='*60)
    if not args.only_vad:
        avg_cer1 = total_cer_noVAD / n * 100
        avg_t1 = total_time_noVAD / n
        print(f'无VAD: 平均CER={avg_cer1:.2f}%  平均{avg_t1:.3f}s/条')
    if not args.no_vad:
        avg_cer2 = total_cer_VAD / n * 100
        avg_t2 = total_time_VAD / n
        print(f'+VAD:  平均CER={avg_cer2:.2f}%  平均{avg_t2:.3f}s/条')

    if args.output:
        out = {
            'avg_cer_noVAD': avg_cer1 if not args.only_vad else 0,
            'avg_cer_VAD': avg_cer2 if not args.no_vad else 0,
            'no_vad': results_noVAD,
            'vad': results_VAD,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'结果已保存: {args.output}')


if __name__ == '__main__':
    main()
