#!/usr/bin/env python3
"""
VAD + ASR 级联评估脚本
支持: RangeVAD-Plus (no_ir), FireRedVAD, SileroVAD, 无VAD
指标: CER, CA (指令准确率)

用法:
    # 无VAD (全送入)
    python3 rccm/eval_vad_asr_cascade.py \
        --test-data data/gunshot_30h/test/test.jsonl \
        --vad-type none \
        --asr-init-param outputs_ms_cce_mixed_k135_n4/model.pt.best \
        --model-class ParaformerStreamingMSCce --use-cce \
        --cce-kernel-sizes 1,3,5 --cce-chunk-size 16 --cce-fixed-weight 0.8 \
        --chunk-size 0,6,4 --lfr-n 4 --device cuda \
        --output results/asr_novad.json

    # RangeVAD-Plus
    python3 rccm/eval_vad_asr_cascade.py \
        --test-data data/gunshot_30h/test/test.jsonl \
        --vad-type rangevad --vad-model rccm/vad_no_ir/model.pt.best \
        --asr-init-param outputs_ms_cce_mixed_k135_n4/model.pt.best \
        --model-class ParaformerStreamingMSCce --use-cce \
        --cce-kernel-sizes 1,3,5 --cce-chunk-size 16 --cce-fixed-weight 0.8 \
        --chunk-size 0,6,4 --lfr-n 4 --device cuda \
        --output results/asr_rangevad.json

    # RangeVAD-Plus + SNR后处理
    python3 rccm/eval_vad_asr_cascade.py \
        --test-data data/gunshot_30h/test/test.jsonl \
        --vad-type rangevad --vad-model rccm/vad_no_ir/model.pt.best \
        --snr-postprocess --snr-threshold 0.05 \
        --asr-init-param outputs_ms_cce_mixed_k135_n4/model.pt.best \
        --model-class ParaformerStreamingMSCce --use-cce \
        --cce-kernel-sizes 1,3,5 --cce-chunk-size 16 --cce-fixed-weight 0.8 \
        --chunk-size 0,6,4 --lfr-n 4 --device cuda \
        --output results/asr_rangevad_snr.json

    # FireRed VAD
    python3 rccm/eval_vad_asr_cascade.py \
        --test-data data/gunshot_30h/test/test.jsonl \
        --vad-type firered \
        --asr-init-param outputs_ms_cce_mixed_k135_n4/model.pt.best \
        --model-class ParaformerStreamingMSCce --use-cce \
        --cce-kernel-sizes 1,3,5 --cce-chunk-size 16 --cce-fixed-weight 0.8 \
        --chunk-size 0,6,4 --lfr-n 4 --device cuda \
        --output results/asr_firered.json

    # Silero VAD (低GFAR)
    python3 rccm/eval_vad_asr_cascade.py \
        --test-data data/gunshot_30h/test/test.jsonl \
        --vad-type silero --silero-threshold 0.30 --min-speech-ms 50 --min-silence-ms 80 \
        --asr-init-param outputs_ms_cce_mixed_k135_n4/model.pt.best \
        --model-class ParaformerStreamingMSCce --use-cce \
        --cce-kernel-sizes 1,3,5 --cce-chunk-size 16 --cce-fixed-weight 0.8 \
        --chunk-size 0,6,4 --lfr-n 4 --device cuda \
        --output results/asr_silero.json
"""

import os
import sys
import json
import time
import argparse
import uuid
import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==================== 热词表 ====================
HOTWORDS = {
    '三号', '七号', '四号', '站立', '旋转', '前进', '暂停',
    '匍匐', '九号', '停止', '向右', '倒靶', '后退', '向左',
    '卧倒', '右转', '开始', '立靶', '六号', '射击', '放靶',
    '二号', '冲击', '十号', '八号', '五号', '一号', '跃进', '左转'
}


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


def extract_hotword_seq(text):
    seq = []
    i = 0
    while i < len(text):
        matched = False
        for hw in sorted(HOTWORDS, key=len, reverse=True):
            if text[i:i+len(hw)] == hw:
                seq.append(hw)
                i += len(hw)
                matched = True
                break
        if not matched:
            i += 1
    return seq


def seq_edit_distance(ref_seq, hyp_seq):
    m, n = len(ref_seq), len(hyp_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_seq[i-1] == hyp_seq[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]


def compute_ca(ref, hyp):
    ref_seq = extract_hotword_seq(ref)
    hyp_seq = extract_hotword_seq(hyp)
    if not ref_seq:
        return 1.0
    dist = seq_edit_distance(ref_seq, hyp_seq)
    return max(0, 1 - dist / len(ref_seq))


# ==================== VAD 加载与推理 ====================
def load_rangevad_plus(model_path, device='cpu', variant='no_ir'):
    """加载 RangeVAD-Plus (支持三分类、四分类、二分类checkpoint)"""
    from train_vad_v5 import extract_logmel
    ck = torch.load(model_path, map_location=device, weights_only=False)

    if 'config' in ck:
        cfg = ck['config']
        ncls = cfg.get('num_classes', 3)
        if ncls == 2:
            from RangeVAD_Plus_2class import RangeVADPlus2Class
            m = RangeVADPlus2Class(
                mel_dim=80,
                hidden_dim=cfg.get('hidden_dim', 64),
                dfsmn_blocks=cfg.get('dfsmn_blocks', 3),
                look_back=cfg.get('look_back', 20),
            )
        else:
            from RangeVAD_Plus import RangeVADPlus
            m = RangeVADPlus(
                mel_dim=80,
                hidden_dim=cfg.get('hidden_dim', 64),
                num_classes=ncls,
                dfsmn_blocks=cfg.get('dfsmn_blocks', 3),
                look_back=cfg.get('look_back', 20),
                variant=cfg.get('variant', 'full'),
            )
        print(f"  VAD类别数: {ncls}")
    else:
        from run_ablation import AblationVAD
        m = AblationVAD(
            variant=variant,
            mel_dim=80,
            hidden_dim=64,
            num_classes=3,
        )
        print(f"  消融变体: {variant}")

    m.load_state_dict(ck['model_state_dict'])
    m.eval().to(device)
    return m, extract_logmel


def rangevad_predict(vad, mel_fn, audio, device, snr_postprocess=False, snr_threshold=0.05):
    """RangeVAD-Plus 推理 → speech segments (ms)
    自动适配2分类(0=非语音,1=语音)和3/4分类(0/1=保留语音)"""
    feat = mel_fn(audio)
    T = feat.shape[0]
    with torch.no_grad():
        logits = vad(torch.FloatTensor(feat).unsqueeze(0).to(device))
    ncls = logits.shape[-1]
    preds = logits[0].argmax(-1).cpu().numpy()

    # 2分类: 1=语音, 0=非语音
    if ncls == 2:
        speech_classes = {1}
        non_speech_classes = {0}
    else:
        # 3/4分类: 干净=0, 带噪=1 保留; 非语音=2 丢弃
        speech_classes = {0, 1}

    # SNR 后处理 (仅多分类)
    if snr_postprocess and ncls >= 3 and (preds == 0).sum() > 0:
        clean_template = feat[preds == 0].mean(axis=0)
        for t in range(T):
            if preds[t] == 1:
                dist = ((feat[t] - clean_template) ** 2).mean()
                if dist >= snr_threshold:
                    preds[t] = 2

    segs = []
    in_sp = False
    start = 0
    for t in range(T):
        if preds[t] in speech_classes and not in_sp:
            in_sp = True; start = t
        elif (preds[t] not in speech_classes if ncls == 2 else preds[t] == 2) and in_sp:
            in_sp = False
            segs.append((start * 10, min(t * 10, len(audio) * 1000 // 16000)))
    if in_sp:
        segs.append((start * 10, len(audio) * 1000 // 16000))
    return segs


def load_firered_vad(device='cuda', speech_threshold=0.35,
                    min_speech_frame=20, min_silence_frame=20,
                    model_dir='/home/FunASR/FunASR-main/pretrained_models/FireRedVAD/VAD'):
    try:
        from fireredvad import FireRedVad, FireRedVadConfig
    except ImportError:
        print("请先安装: pip install fireredvad"); return None
    use_gpu = (device == 'cuda')
    vad_config = FireRedVadConfig(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=speech_threshold,
        min_speech_frame=min_speech_frame,
        max_speech_frame=2000,
        min_silence_frame=min_silence_frame,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    return FireRedVad.from_pretrained(model_dir, vad_config)


def firered_predict(vad, audio, sr=16000):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr, subtype='PCM_16')
        tmp_path = f.name
    result, _ = vad.detect(tmp_path)
    os.unlink(tmp_path)
    timestamps = result['timestamps']
    if not timestamps:
        return []
    return [(s * 1000, e * 1000) for s, e in timestamps]


def load_silero_vad():
    try:
        from silero_vad import load_silero_vad as _ld, get_speech_timestamps
        return _ld(onnx=True), get_speech_timestamps
    except ImportError:
        print("请先安装 silero-vad: pip install silero-vad"); return None, None


def silero_predict(model, get_ts, audio, sr=16000,
                   threshold=0.3, min_ms=50, min_sil=80, speech_pad_ms=30):
    ts_list = get_ts(audio, model, sampling_rate=sr,
                     threshold=threshold,
                     min_speech_duration_ms=min_ms,
                     min_silence_duration_ms=min_sil,
                     speech_pad_ms=speech_pad_ms)
    return [(t['start'] if isinstance(t, dict) else t[0],
             t['end'] if isinstance(t, dict) else t[1]) for t in ts_list]


# ==================== ASR ====================
def build_asr_model(args):
    model_kwargs = {
        "model": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
        "model_class": args.model_class,
    }
    if args.asr_init_param:
        model_kwargs["init_param"] = args.asr_init_param
    if args.use_cce:
        model_kwargs.update({
            "use_cce": True,
            "cce_kernel_sizes": [int(x) for x in args.cce_kernel_sizes.split(',')],
            "cce_chunk_size": args.cce_chunk_size,
            "cce_fixed_weight": args.cce_fixed_weight,
        })

    model = AutoModel(**model_kwargs)

    # 运行时修改 LFR n
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
        print(f"  LFR_n: {old_n} -> {args.lfr_n}")

    if args.use_cce:
        found = False
        for attr_name in dir(model):
            try:
                obj = getattr(model, attr_name)
            except Exception:
                continue
            if hasattr(obj, 'cce_module') and hasattr(obj.cce_module, 'fixed_weight'):
                obj.cce_module.fixed_weight = args.cce_fixed_weight
                print(f"  cce_fixed_weight 已强制设置为: {args.cce_fixed_weight} (via {attr_name}.cce_module)")
                found = True
                break
        if not found:
            print(f"  [WARN] 找不到 CCE 模块，无法设置 fixed_weight={args.cce_fixed_weight}")

    return model


def _force_set_fixed_weight(module, weight, depth=0, visited=None):
    """深度递归搜索所有子对象，强制设置 fixed_weight"""
    if visited is None:
        visited = set()
    obj_id = id(module)
    if obj_id in visited or depth > 10:
        return False
    visited.add(obj_id)

    if hasattr(module, 'fixed_weight'):
        module.fixed_weight = weight
        return True

    # 搜索所有属性（非私有）
    for attr_name in dir(module):
        if attr_name.startswith('_'):
            continue
        try:
            obj = getattr(module, attr_name, None)
        except Exception:
            continue
        if obj is None or obj is module:
            continue
        if id(obj) in visited:
            continue
        if hasattr(obj, 'fixed_weight'):
            obj.fixed_weight = weight
            return True

    # 递归进入所有子对象
    for attr_name in dir(module):
        if attr_name.startswith('_'):
            continue
        try:
            obj = getattr(module, attr_name, None)
        except Exception:
            continue
        if obj is None or obj is module or id(obj) in visited:
            continue
        if isinstance(obj, (torch.nn.Module, object)):
            if _force_set_fixed_weight(obj, weight, depth + 1, visited):
                return True
    return False

def asr_infer(asr_model, audio_f32, chunk_size, fixed_weight=None):
    tmp_path = f"/tmp/asr_{os.getpid()}_{uuid.uuid4().hex[:8]}.wav"
    sf.write(tmp_path, audio_f32, 16000, subtype='PCM_16')
    try:
        if fixed_weight is not None:
            # 直接暴力设 cce_module.fixed_weight（绕过 AutoModel 封装层）
            for attr_name in dir(asr_model):
                try:
                    obj = getattr(asr_model, attr_name)
                except Exception:
                    continue
                if hasattr(obj, 'cce_module') and hasattr(obj.cce_module, 'fixed_weight'):
                    obj.cce_module.fixed_weight = fixed_weight
                    break
            if not _force_set_fixed_weight(asr_model, fixed_weight):
                # 兜底：递归搜
                pass
        res = asr_model.generate(input=tmp_path, batch_size=1, chunk_size=chunk_size)
        text = res[0]['text'].strip() if res and res[0].get('text') else ''
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return text


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', required=True, help='JSONL 测试数据')
    parser.add_argument('--vad-type', default='none',
                        choices=['none', 'rangevad', 'firered', 'silero'])
    parser.add_argument('--vad-model', default=None, help='RangeVAD 权重路径')
    parser.add_argument('--vad-variant', default='no_ir',
                        help='消融变体名称 (no_ir/no_lstm/no_dfsmn等), 默认 no_ir')
    parser.add_argument('--snr-postprocess', action='store_true')
    parser.add_argument('--snr-threshold', type=float, default=0.05)
    parser.add_argument('--silero-threshold', type=float, default=0.30)
    parser.add_argument('--min-speech-ms', type=int, default=50)
    parser.add_argument('--min-silence-ms', type=int, default=80)
    parser.add_argument('--silero-speech-pad-ms', type=int, default=30)
    parser.add_argument('--firered-threshold', type=float, default=0.35)
    parser.add_argument('--firered-min-speech-frame', type=int, default=50)
    parser.add_argument('--firered-min-silence-frame', type=int, default=40)
    # ASR 参数
    parser.add_argument('--asr-init-param', default=None, help='ASR 微调权重路径，不指定则用官方预训练')
    parser.add_argument('--model-class', default='ParaformerStreamingMSCce')
    parser.add_argument('--use-cce', action='store_true', default=False)
    parser.add_argument('--cce-kernel-sizes', type=str, default='1,3,5')
    parser.add_argument('--cce-chunk-size', type=int, default=16)
    parser.add_argument('--cce-fixed-weight', type=float, default=0.8)
    parser.add_argument('--chunk-size', type=str, default='0,6,4')
    parser.add_argument('--lfr-n', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int, default=0, help='0=全部')
    parser.add_argument('--output', default=None, help='结果 JSON 路径')
    args = parser.parse_args()

    chunk_size = [int(x) for x in args.chunk_size.split(',')]

    # 加载测试数据
    data = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({'source': item['source'],
                         'target': item.get('target', item.get('text', ''))})
    if args.max_samples > 0:
        data = data[:args.max_samples]
    print(f"测试样本数: {len(data)}")

    # 加载 VAD
    vad_fn = None
    if args.vad_type == 'rangevad':
        print(f"加载 RangeVAD-Plus ({args.vad_model})...")
        vad, mel_fn = load_rangevad_plus(args.vad_model, args.device, args.vad_variant)
        if args.snr_postprocess:
            print(f"  SNR后处理已启用, MSE阈值={args.snr_threshold}")
        vad_fn = lambda a: rangevad_predict(
            vad, mel_fn, a, args.device, args.snr_postprocess, args.snr_threshold)
    elif args.vad_type == 'firered':
        print(f"加载 FireRedVAD (speech_threshold={args.firered_threshold}, "
              f"min_sp={args.firered_min_speech_frame}, min_si={args.firered_min_silence_frame})...")
        fvad = load_firered_vad(args.device,
                                speech_threshold=args.firered_threshold,
                                min_speech_frame=args.firered_min_speech_frame,
                                min_silence_frame=args.firered_min_silence_frame)
        vad_fn = lambda a: firered_predict(fvad, a)
    elif args.vad_type == 'silero':
        print(f"加载 Silero VAD (threshold={args.silero_threshold})...")
        svad, get_ts = load_silero_vad()
        vad_fn = lambda a: silero_predict(
            svad, get_ts, a, 16000,
            args.silero_threshold, args.min_speech_ms, args.min_silence_ms,
            args.silero_speech_pad_ms)
    else:
        print("无VAD (全送入)")

    # 加载 ASR
    print("加载 ASR...")
    cce_str = "有CCE" if args.use_cce else "无CCE"
    print(f"  Model: {args.model_class} ({cce_str}), init_param={args.asr_init_param}")
    if args.use_cce:
        print(f"  CCE k={args.cce_kernel_sizes}, chunk={args.cce_chunk_size}, w={args.cce_fixed_weight}")
    print(f"  Chunk: {chunk_size}, LFR_n: {args.lfr_n}")
    asr = build_asr_model(args)
    print("加载完成\n")

    # 测试循环
    total_cer, total_ca, total_time = 0.0, 0.0, 0.0
    n_correct = 0
    results = []

    for idx, item in enumerate(data):
        wav_path = item['source']
        ref = item['target']

        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio.mean(1)
        audio = audio.astype(np.float32)

        t0 = time.time()

        if vad_fn:
            segs = vad_fn(audio)
            if segs:
                parts = [audio[int(s/1000*sr):int(e/1000*sr)] for s, e in segs]
                audio_in = np.concatenate(parts).astype(np.float32)
            else:
                audio_in = np.zeros(0, dtype=np.float32)
        else:
            audio_in = audio

        if len(audio_in) == 0:
            hyp = ''
        else:
            hyp = asr_infer(asr, audio_in, chunk_size, fixed_weight=args.cce_fixed_weight if args.use_cce else None)

        t_elapsed = time.time() - t0
        cer = compute_cer(ref, hyp)
        ca = compute_ca(ref, hyp)
        total_cer += cer; total_ca += ca; total_time += t_elapsed
        if ca == 1.0: n_correct += 1

        results.append({'ref': ref, 'hyp': hyp, 'cer': cer, 'ca': ca, 'time': t_elapsed})

        if (idx + 1) % 50 == 0 or idx < 5:
            print(f'[{idx+1}/{len(data)}] ref={ref}')
            print(f'  hyp={hyp}  CER={cer*100:.1f}%  CA={ca*100:.1f}%')

    n = len(data)
    avg_cer = total_cer / n * 100
    avg_ca = total_ca / n * 100
    acc = n_correct / n * 100
    avg_time = total_time / n

    print(f"\n{'='*60}")
    print("最终结果")
    print(f"{'='*60}")
    print(f"VAD:           {args.vad_type}" + ("+SNR" if args.snr_postprocess else ""))
    print(f"样本数:         {n}")
    print(f"平均CER:        {avg_cer:.2f}%")
    print(f"平均CA:         {avg_ca:.2f}%")
    print(f"指令准确率:     {acc:.2f}%")
    print(f"平均耗时:       {avg_time:.3f}s/条")

    if args.output:
        out = {
            'vad_type': args.vad_type,
            'snr_postprocess': args.snr_postprocess,
            'avg_cer': avg_cer,
            'avg_ca': avg_ca,
            'accuracy': acc,
            'avg_time': avg_time,
            'n_samples': n,
            'results': results,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'\n结果已保存: {args.output}')


if __name__ == '__main__':
    main()
