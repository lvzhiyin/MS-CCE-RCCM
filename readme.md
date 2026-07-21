# 面向靶场实弹训练的离线无人靶车语音控制系统

> 硕士论文代码仓库  
> 第三章 RangeVAD 三分类枪声鲁棒 VAD  
> 第四章 MS-CCE 多尺度因果上下文嵌入流式 ASR  
> 第五章 RCCM 靶场指令纠错模型

## 环境

| 项目 | 配置 |
|---|---|
| 深度学习框架 | PyTorch 1.13.0 |
| 语音工具包 | FunASR 1.3.1 |
| 云服务器 | 阿里云 GPU 云服务器 |
| GPU | NVIDIA A10 24GB |
| CPU | Intel Xeon Platinum 8369B @ 2.90GHz |
| 内存 | 28 GB |
| OS | Ubuntu 22.04.5 LTS |
| CUDA | 11.7 |

## 安装

```bash
pip install -r requirements.txt
pip install torch==1.13.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117
pip install funasr
```

## 目录结构

```
FunASR-main/
├── MS-CCE/                # 第四章：多尺度因果上下文嵌入
│   ├── 训练代码/          # train_ms_cce_mixed_k135.sh（MS-CCE 训练）
│   ├── 模型结构/          # model_msa_cce.py（覆盖 FunASR 原文件）
│   ├── 模型权重/          # outputs_ms_cce_mixed_k135_n4/（最优权重，855MB，k={1,3,5}, w=0.3）
│   │                       # ⚠ 因 GitHub 100MB 限制未上传，请从百度网盘获取：链接见仓库首页
│   ├── 数据生成/          # generate_gunshot_30h.py（靶场数据合成）
│   ├── 测试代码/          # test_baseline.py（ASR 测试，兼容 MS-CCE 与 CCE）
│   └── 测试结果/          # 对比实验 / chunk消融 / 卷积核消融 / 注入权重消融 / 门控消融
│
├── RangeVAD/              # 第三章：三分类枪声鲁棒 VAD
│   ├── 训练代码/          # train_vad_v5.py / train_vad_plus.py / train_vad_plus_2class.py
│   ├── 模型结构/          # RangeVAD_Plus.py（三分类） / RangeVAD_Plus_2class.py（二分类消融）
│   ├── 评测代码/          # eval_vad_frame.py / eval_vad_asr_cascade.py / eval_baseline_vads.py 等
│   ├── 模型权重/          # model.pt.best / no_lstm / no_dfsmn / dfsmn_1/2/5
│   └── 测试结果/          # 帧级评估 / 级联 ASR 结果 / Silero & FireRed 网格搜索
│
├── RCCM/                  # 第五章：靶场指令纠错模型
│   ├── 训练代码/          # train_rccm_hotword.py（热词感知损失训练）
│   ├── 模型结构/          # light_rccm.py（RCCM 模型定义）
│   ├── 模型权重/          # outputs_rccm_k135_n4_hw2/（最优权重，λ_hw=2.0）
│   ├── 测试代码/          # test_rccm_simple.py / rule_correct.py
│   └── 训练数据/          # real_asr_pairs_train_k135_n4.jsonl（ASR 错误对）
│
├── readme.md
└── requirements.txt
```

## 使用

### 第三章 — RangeVAD 帧级评估

```bash
cd RangeVAD/评测代码

# 帧级评估（支持三分类/二分类）
python eval_vad_frame.py --data-dir <测试数据路径> --model-path <权重路径> --split test --device cuda

# 基线 VAD 对比
python eval_baseline_vads.py --data-dir <测试数据路径>

# Silero VAD 超参数网格搜索
python grid_search_vad.py --data-dir <测试数据路径>
```

### 第三章 — RangeVAD + ASR 级联评估

```bash
cd RangeVAD/评测代码

# 三分类 RangeVAD 级联
python eval_vad_asr_cascade.py \
    --vad-type rangevad --vad-model <RangeVAD权重路径> \
    --asr-init-param <微调后Paraformer权重路径> \
    --model-class ParaformerStreaming \
    --chunk-size 0,6,4 --lfr-n 4 --device cuda \
    --test-data <测试数据路径> --output <结果路径>

# 二分类消融级联
python eval_vad_asr_cascade.py \
    --vad-type rangevad --vad-model <二分类RangeVAD权重路径> \
    --asr-init-param <微调后Paraformer权重路径> \
    --model-class ParaformerStreaming \
    --chunk-size 0,6,4 --lfr-n 4 --device cuda \
    --test-data <测试数据路径> --output <结果路径>

# FireRed VAD / Silero VAD 级联
python eval_vad_asr_cascade.py \
    --vad-type firered --firered-threshold 0.35 \
    --asr-init-param <微调后Paraformer权重路径> \
    --model-class ParaformerStreaming \
    --chunk-size 0,6,4 --lfr-n 4 --device cuda \
    --test-data <测试数据路径> --output <结果路径>
```

### 第四章 — MS-CCE 流式 ASR 测试

```bash
cd MS-CCE/测试代码

# MS-CCE 多核测试
python test_baseline.py \
    --init-param ../模型权重/outputs_ms_cce_mixed_k135_n4/model.pt.best \
    --test-data <测试数据路径> \
    --model-class ParaformerStreamingMSCce \
    --use-cce --cce-kernel-sizes 1,3,5 --cce-chunk-size 16 \
    --cce-fixed-weight 0.3 --chunk-size 0,6,4 --lfr-n 4 \
    --device cuda --output <结果路径>

# CCE 单核消融测试（k={1,3,5,6}）
python test_baseline.py \
    --init-param <单核CCE权重路径> \
    --test-data <测试数据路径> \
    --model-class ParaformerStreamingCCE \
    --use-cce --cce-kernel-sizes <k> --cce-chunk-size 16 \
    --cce-fixed-weight 0.3 --chunk-size 0,6,4 --lfr-n 4 \
    --device cuda --output <结果路径>
```

### 第五章 — RCCM 纠错测试

```bash
cd RCCM/测试代码

# RCCM 热词纠错测试
python test_rccm_simple.py \
    --rccm-model ../模型权重/outputs_rccm_k135_n4_hw2/model.pt.best \
    --asr-results <ASR测试结果JSON> \
    --pairs ../训练数据/real_asr_pairs_train_k135_n4.jsonl

# 基于规则的后处理纠错
python rule_correct.py --asr-results <ASR测试结果JSON>
```

## 重新训练

### RangeVAD 三分类训练

```bash
cd RangeVAD/训练代码

python train_vad_plus.py --mode train \
    --data-dir <VAD数据集路径> --output-dir <输出路径> \
    --epochs 30 --batch-size 32 --device cuda
```

### RangeVAD 二分类消融训练

```bash
cd RangeVAD/训练代码

python train_vad_plus_2class.py --mode train \
    --data-dir <VAD数据集路径> --output-dir <输出路径> \
    --epochs 30 --batch-size 32 --device cuda
```

### RangeVAD 消融变体训练

```bash
cd RangeVAD/评测代码

python run_ablation.py --data-dir <VAD数据集路径> --ablation <no_ir|no_lstm|no_dfsmn|dfsmn_1|dfsmn_2|dfsmn_5> \
    --output-dir <输出路径>
```

### RCCM 纠错模型训练

```bash
cd RCCM/训练代码

nohup python3 -u train_rccm_hotword.py \
    --pairs ../训练数据/real_asr_pairs_train_k135_n4.jsonl \
    --output-dir ../模型权重/outputs_rccm_k135_n4_hw2 \
    --epochs 10 --batch-size 64 --lr 5e-4 \
    --hotword-weight 2.0 --max-len 30 \
    --d-model 256 --nhead 8 --num-layers 4 --dim-ff 1024 \
    > train.log 2>&1 &
```
