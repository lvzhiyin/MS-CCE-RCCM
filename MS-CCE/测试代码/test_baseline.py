#!/usr/bin/env python3
"""
基线测试: 官方预训练 ParaformerStreaming（无CCE、无微调）
输出: CER、热词召回率、RTF

使用方法:
    python3 test_baseline.py \
        --test-data ../data/gunshot_30h/test/test.jsonl \
        --device cuda
"""

import os
import sys

# 【关键修复】在导入 funasr 之前，将项目根目录加入 sys.path
# 否则 Python 会从 site-packages 导入 pip 安装的官方 funasr，
# 官方版没有 model_msa_cce.py，导致 ParaformerStreamingMSCce 无法注册，
# build_model 会 fallback 到基础 ParaformerStreaming（没有 CCE），
# fixed_weight 注入权重自然无效。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import json
import time
import argparse
import uuid
import soundfile as sf
import numpy as np

import torch
from funasr import AutoModel
from funasr.register import tables

# 验证自定义模型是否注册成功
_model_cls = tables.model_classes.get("ParaformerStreamingMSCce")
if _model_cls is not None:
    print(f"[DIAG] ParaformerStreamingMSCce 已注册: {_model_cls}")
else:
    print(f"[DIAG] *** 警告: ParaformerStreamingMSCce 未注册! CCE 将不会生效! ***")
    print(f"[DIAG] funasr 导入路径: {os.path.dirname(os.path.abspath(sys.modules['funasr'].__file__))}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    """按顺序提取热词序列"""
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
    """计算序列编辑距离"""
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


def compute_hotword_recall(ref, hyp):
    """基于热词序列编辑距离的HRR"""
    ref_seq = extract_hotword_seq(ref)
    hyp_seq = extract_hotword_seq(hyp)
    if not ref_seq:
        return 1.0
    dist = seq_edit_distance(ref_seq, hyp_seq)
    return max(0, 1 - dist / len(ref_seq))


def load_rangevad(model_path, device='cpu', variant=None):
    """加载 RangeVAD 模型（支持标准模型和消融变体）"""
    from train_vad_v5 import BiLSTMVAD_v5, extract_logmel
    ck = torch.load(model_path, map_location=device, weights_only=False)
    if variant:
        from run_ablation import AblationVAD
        m = AblationVAD(variant, mel_dim=80, hidden_dim=64, num_classes=3)
        print(f"  [RangeVAD] 消融变体 '{variant}': {sum(p.numel() for p in m.parameters()):,} 参数")
    elif 'config' in ck:
        cfg = ck['config']
        if 'dfsmn_blocks' in cfg:
            from RangeVAD_Plus import RangeVADPlus
            m = RangeVADPlus(mel_dim=80, hidden_dim=cfg['hidden_dim'], num_classes=3,
                             dfsmn_blocks=cfg['dfsmn_blocks'],
                             look_back=cfg.get('look_back', 20))
        else:
            m = BiLSTMVAD_v5(hidden_dim=cfg['hidden_dim'], num_layers=cfg['num_layers'],
                             dropout=cfg['dropout'],
                             use_freq_attn=cfg.get('use_freq_attn', True))
    else:
        raise RuntimeError("Checkpoint不含config，请指定 --rangevad-variant")
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m.to(device), extract_logmel


def apply_rangevad(wav_path, vad_model, extract_logmel):
    """用 RangeVAD 提取语音段，返回拼接后的临时 wav 路径。
    3分类: 0=干净语音(留), 1=带噪语音(留), 2=非语音(丢)"""
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = audio.astype(np.float32)
    feat = extract_logmel(audio)
    vad_device = next(vad_model.parameters()).device
    with torch.no_grad():
        logits = vad_model(torch.FloatTensor(feat).unsqueeze(0).to(vad_device))
    preds = logits[0].argmax(-1).cpu().numpy()
    # 3分类: 0=干净语音, 1=带噪语音 → 保留; 2=非语音 → 丢弃
    HOP_MS = 10
    HOP = int(sr * HOP_MS / 1000)
    segs = []
    in_sp = False
    start = 0
    for t in range(len(preds)):
        if preds[t] in (0, 1) and not in_sp:
            in_sp = True
            start = t
        elif preds[t] == 2 and in_sp:
            in_sp = False
            if t - start >= 5:
                s_samp = start * HOP
                e_samp = min(t * HOP, len(audio))
                segs.append(audio[s_samp:e_samp])
    if in_sp:
        s_samp = start * HOP
        segs.append(audio[s_samp:])
    if not segs:
        return None
    speech_audio = np.concatenate(segs)
    tmp_path = f"/tmp/rvad_{uuid.uuid4().hex[:8]}.wav"
    sf.write(tmp_path, speech_audio, sr, subtype='PCM_16')
    return tmp_path


def load_test_data(test_jsonl):
    data = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                'source': item['source'],
                'target': item.get('target', item.get('text', ''))
            })
    return data


def main():
    parser = argparse.ArgumentParser(description='基线测试')
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--model-class', type=str, default='ParaformerStreaming')
    parser.add_argument('--use-cce', action='store_true', default=False)
    parser.add_argument('--init-param', type=str, default=None)
    parser.add_argument('--chunk-size', type=str, default='0,10,5',
                        help='chunk配置, 逗号分隔, 如 0,10,5')
    parser.add_argument('--cce-kernel-sizes', type=str, default='3,6,9',
                        help='MS-CCE卷积核大小, 逗号分隔, 如 2,4,6')
    parser.add_argument('--cce-chunk-size', type=int, default=16,
                        help='CCE注入chunk大小')
    parser.add_argument('--cce-fixed-weight', type=float, default=0.8,
                        help='CCE固定注入权重')
    parser.add_argument('--sacm-model-path', type=str, default=None,
                        help='SACM纠错模型路径 (ParaformerStreamingSACM)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='最多测试N条, 0=全部')
    parser.add_argument('--start', type=int, default=0,
                        help='分片起始索引 (0-based)')
    parser.add_argument('--count', type=int, default=0,
                        help='分片取多少条, 0=取到末尾')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--lfr-n', type=int, default=6,
                        help='LFR n参数 (默认6, 改小降低首字延迟)')
    parser.add_argument('--use-vad', action='store_true', default=False,
                        help='启用FSMN-VAD端点检测，过滤静音/纯噪声段')
    parser.add_argument('--vad-model', type=str, 
                        default='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                        help='VAD模型名称')
    parser.add_argument('--vad-max-end-silence', type=int, default=400,
                        help='VAD最大句尾静音时间(ms), 默认400')
    parser.add_argument('--vad-speech-noise-thres', type=float, default=0.6,
                        help='VAD语音/噪声判决阈值, 越高越倾向判为语音, 默认0.6')
    parser.add_argument('--vad-speech2noise-ratio', type=float, default=1.0,
                        help='VAD语音噪声比, 默认1.0')
    parser.add_argument('--vad-do-extend', type=int, default=1,
                        help='VAD是否扩展语音段边界, 0/1, 默认1')
    parser.add_argument('--rangevad-model', type=str, default=None,
                        help='RangeVAD模型路径 (用于枪声环境VAD)')
    parser.add_argument('--rangevad-variant', type=str, default=None,
                        help='RangeVAD消融变体 (no_ir/no_lstm等), checkpoint不含config时需指定')
    args = parser.parse_args()

    chunk_size = [int(x) for x in args.chunk_size.split(',')]

    cce_str = "有CCE" if args.use_cce else "无CCE"
    init_str = f", init_param={args.init_param}" if args.init_param else ", 无微调"
    print("=" * 70)
    print(f"{args.model_class} 测试: {cce_str}{init_str}")
    print("=" * 70)
    print(f"测试数据: {args.test_data}")
    print(f"设备: {args.device}")
    print(f"Chunk: {chunk_size}")
    print("=" * 70)

    test_data = load_test_data(args.test_data)
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
    # 分片支持: --start 起始索引, --count 取多少条
    if args.start > 0 or args.count > 0:
        total = len(test_data)
        s = args.start
        e = s + args.count if args.count > 0 else total
        test_data = test_data[s:e]
        print(f"  分片: [{s}:{e}] / {total} 条")
    print(f"\n测试样本数: {len(test_data)}")

    print("\n加载模型...")
    model_kwargs = {
        "model": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
        "model_class": args.model_class,
        "use_cce": args.use_cce,
        "cce_kernel_sizes": [int(x) for x in args.cce_kernel_sizes.split(',')],
        "cce_chunk_size": args.cce_chunk_size,
        "cce_fixed_weight": args.cce_fixed_weight,
    }
    if args.init_param:
        model_kwargs["init_param"] = args.init_param
    if args.sacm_model_path:
        model_kwargs["sacm_model_path"] = args.sacm_model_path
    model = AutoModel(**model_kwargs)

    # 【诊断】确认加载的模型类是否正确
    actual_model_cls = type(model.model).__name__
    print(f"  [DIAG] 实际模型类: {actual_model_cls}")
    if actual_model_cls != args.model_class:
        print(f"  [DIAG] *** 严重错误: 期望 {args.model_class}，实际加载了 {actual_model_cls}! ***")
        print(f"  [DIAG] *** 这意味着自定义模型类未被注册，CCE 不会生效! ***")
    has_cce = hasattr(model.model, 'cce_module')
    print(f"  [DIAG] cce_module 存在: {has_cce}")
    if has_cce:
        print(f"  [DIAG] cce_module.fixed_weight 当前值: {model.model.cce_module.fixed_weight}")

    # 运行时修改LFR n参数（降低首字延迟）
    frontend_obj = None
    if hasattr(model, 'frontend'):
        frontend_obj = model.frontend
    elif hasattr(model, 'kwargs') and 'frontend' in model.kwargs:
        frontend_obj = model.kwargs['frontend']
    
    if frontend_obj is not None and args.lfr_n != frontend_obj.lfr_n:
        old_n = frontend_obj.lfr_n
        frontend_obj.lfr_n = args.lfr_n
        # 同步所有位置
        if hasattr(model, 'kwargs') and 'frontend' in model.kwargs:
            model.kwargs['frontend'].lfr_n = args.lfr_n
        if hasattr(model, 'frontend'):
            model.frontend.lfr_n = args.lfr_n
        # 关键：_base_kwargs_map 会被 _reset_runtime_configs 恢复，必须同步修改
        if hasattr(model, '_base_kwargs_map') and 'kwargs' in model._base_kwargs_map:
            if 'frontend' in model._base_kwargs_map['kwargs']:
                model._base_kwargs_map['kwargs']['frontend'].lfr_n = args.lfr_n
        print(f"  LFR_n: {old_n} -> {args.lfr_n} (首字延迟 {args.lfr_n*10}ms/帧)")

    # 强制设置注入权重（AutoModel 把真实模型放在 self.model）
    if args.use_cce and hasattr(model.model, 'cce_module'):
        model.model.cce_module.fixed_weight = args.cce_fixed_weight
        print(f"  cce_fixed_weight 已强制设置为: {args.cce_fixed_weight}")
    elif args.use_cce:
        print(f"  [WARN] 找不到 model.model.cce_module，无法设置 fixed_weight={args.cce_fixed_weight}")

    print("模型加载完成")

    # VAD
    vad_pipeline = None
    if args.use_vad:
        print("\n加载VAD模型...")
        print(f"  max_end_silence_time={args.vad_max_end_silence}ms")
        print(f"  speech_noise_thres={args.vad_speech_noise_thres}")
        print(f"  speech_2_noise_ratio={args.vad_speech2noise_ratio}")
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model=args.vad_model,
            vad_post_args={
                'max_end_silence_time': args.vad_max_end_silence,
                'speech_noise_thres': args.vad_speech_noise_thres,
                'speech_2_noise_ratio': args.vad_speech2noise_ratio,
                'do_extend': args.vad_do_extend,
            },
        )
        vad_cfg = {'max_end_sil': args.vad_max_end_silence}
        print("VAD模型加载完成")

    # RangeVAD (枪声环境三分类VAD)
    rangevad_model = None
    rangevad_mel_fn = None
    if args.rangevad_model:
        print("\n加载 RangeVAD...")
        rangevad_model, rangevad_mel_fn = load_rangevad(
            args.rangevad_model, args.device, args.rangevad_variant)
        print("RangeVAD 加载完成")

    results = []
    total_audio_s = 0.0
    total_proc_s = 0.0

    vad_tag = ' [VAD]' if args.use_vad else ''
    rvad_tag = ' [RangeVAD]' if args.rangevad_model else ''
    print(f"\n开始测试 ({len(test_data)} 条)...{rvad_tag}{vad_tag}\n")

    for idx, item in enumerate(test_data):
        wav_path = item['source']
        ref = item['target']

        try:
            info = sf.info(wav_path)
            audio_dur = info.duration
            total_audio_s += audio_dur

            # RangeVAD 过滤 (枪声环境三分类VAD，优先于FSMN-VAD)
            rvad_wav_path = None
            if rangevad_model is not None:
                rvad_wav_path = apply_rangevad(wav_path, rangevad_model, rangevad_mel_fn)
                if rvad_wav_path is None:
                    results.append({
                        'ref': ref, 'hyp': '', 'hyp_asr': '',
                        'cer': 1.0, 'hotword_recall': 0.0,
                        'cer_asr': 1.0, 'hotword_recall_asr': 0.0,
                        'rccm_commits': [],
                        'time': 0, 'audio_dur': audio_dur
                    })
                    continue
                wav_path = rvad_wav_path

            # VAD过滤
            vad_wav_path = wav_path
            if args.use_vad:
                vad_result = vad_pipeline(audio_in=wav_path)
                segments = vad_result.get('text', [])
                if segments:
                    audio, sr = sf.read(wav_path)
                    speech_parts = []
                    for seg in segments:
                        start_sample = int(seg[0] * sr / 1000)
                        end_sample = int(seg[1] * sr / 1000)
                        speech_parts.append(audio[start_sample:end_sample])
                    if speech_parts:
                        speech_audio = np.concatenate(speech_parts)
                        vad_wav_path = f"/tmp/vad_{uuid.uuid4().hex[:8]}.wav"
                        sf.write(vad_wav_path, speech_audio, sr, subtype='PCM_16')
                    else:
                        # 无语音，返回空结果
                        results.append({
                            'ref': ref, 'hyp': '', 'hyp_asr': '',
                            'cer': 1.0, 'hotword_recall': 0.0,
                            'cer_asr': 1.0, 'hotword_recall_asr': 0.0,
                            'sacm_commits': [],
                            'time': 0, 'audio_dur': audio_dur
                        })
                        continue
                else:
                    # VAD未检测到语音
                    results.append({
                        'ref': ref, 'hyp': '', 'hyp_asr': '',
                        'cer': 1.0, 'hotword_recall': 0.0,
                        'cer_asr': 1.0, 'hotword_recall_asr': 0.0,
                        'rccm_commits': [],
                        'time': 0, 'audio_dur': audio_dur
                    })
                    continue

            t0 = time.time()
            res = model.generate(vad_wav_path, batch_size=1, chunk_size=chunk_size)
            elapsed = time.time() - t0
            total_proc_s += elapsed

            # 清理临时文件
            if rvad_wav_path and os.path.exists(rvad_wav_path):
                os.unlink(rvad_wav_path)
            if args.use_vad and vad_wav_path != item['source'] and vad_wav_path != rvad_wav_path:
                os.unlink(vad_wav_path)

            text = res[0]['text'] if res and res[0].get('text') else ''
            text_asr = res[0].get('text_asr', text) if res else text  # 原始ASR文本
            sacm_commits = res[0].get('sacm_commits', []) if res else []

            cer = compute_cer(ref, text)
            hwr = compute_hotword_recall(ref, text)
            cer_asr = compute_cer(ref, text_asr)
            hwr_asr = compute_hotword_recall(ref, text_asr)

            results.append({
                'ref': ref, 'hyp': text,
                'hyp_asr': text_asr,
                'cer': cer, 'hotword_recall': hwr,
                'cer_asr': cer_asr, 'hotword_recall_asr': hwr_asr,
                'sacm_commits': sacm_commits,
                'time': elapsed, 'audio_dur': audio_dur
            })

        except Exception as e:
            print(f"  [{idx}] 错误: {e}")
            results.append({
                'ref': ref, 'hyp': '', 'hyp_asr': '',
                'cer': 1.0, 'hotword_recall': 0.0,
                'cer_asr': 1.0, 'hotword_recall_asr': 0.0,
                'rccm_commits': [],
                'time': 0, 'audio_dur': 0
            })

        if (idx + 1) % 500 == 0:
            cur_cer = sum(r['cer'] for r in results) / len(results) * 100
            cur_hwr = sum(r['hotword_recall'] for r in results) / len(results) * 100
            cur_cer_asr = sum(r['cer_asr'] for r in results) / len(results) * 100
            cur_hwr_asr = sum(r['hotword_recall_asr'] for r in results) / len(results) * 100
            print(f"  [{idx+1}/{len(test_data)}] "
                  f"ASR_CER={cur_cer_asr:.2f}%  SACM_CER={cur_cer:.2f}%  "
                  f"ASR_HRR={cur_hwr_asr:.2f}%  SACM_HRR={cur_hwr:.2f}%")

    n = len(results)
    rtf = total_proc_s / total_audio_s if total_audio_s > 0 else 0
    avg_cer = sum(r['cer'] for r in results) / n * 100
    avg_hwr = sum(r['hotword_recall'] for r in results) / n * 100
    avg_cer_asr = sum(r['cer_asr'] for r in results) / n * 100
    avg_hwr_asr = sum(r['hotword_recall_asr'] for r in results) / n * 100
    avg_time = sum(r['time'] for r in results) / n
    total_time = sum(r['time'] for r in results)

    print(f"\n{'='*70}")
    print("测试结果")
    print('='*70)
    print(f"  样本数:          {n}")
    print(f"  总音频时长:      {total_audio_s:.1f} s  ({total_audio_s/3600:.1f} h)")
    print(f"  总推理时间:      {total_time:.1f} s")
    print(f"  ASR  CER:        {avg_cer_asr:.2f}%")
    print(f"  ASR  HRR:        {avg_hwr_asr:.2f}%")
    print(f"  SACM CER:        {avg_cer:.2f}%")
    print(f"  SACM HRR:        {avg_hwr:.2f}%")
    print(f"  RTF (实时因子):  {rtf:.4f}  ({'✅ 实时' if rtf < 1.0 else '❌ 非实时'})")
    print(f"  平均推理时间:    {avg_time*1000:.1f} ms/句")

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'model': f'{args.model_class}',
                'num_samples': n,
                'cer': avg_cer,
                'hotword_recall': avg_hwr,
                'cer_asr': avg_cer_asr,
                'hotword_recall_asr': avg_hwr_asr,
                'rtf': rtf,
                'total_audio_s': total_audio_s,
                'total_time_s': total_time,
                'avg_time_s': avg_time,
                'details': results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
