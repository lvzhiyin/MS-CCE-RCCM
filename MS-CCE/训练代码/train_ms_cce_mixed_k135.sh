#!/bin/bash
# ==============================================================================
# MS-CCE 混合训练 - kernel=[1,3,5], chunk_size=16
# ==============================================================================
#
# 改进点 vs MS-CCE([2,4,6]):
#   ① kernel=[1,3,5] (60/180/300ms) 进一步缩小核尺度
#   ② k=1(60ms) 聚焦单帧音素瞬态, k=5(300ms) 回看不超半个chunk
#   ③ 其他参数与训练策略完全对齐k246
#
# 训练策略:
#   冻结: Decoder, Predictor, CTC
#   训练: Encoder全40层 + MS-CCE模块
# ==============================================================================

workspace=`pwd`

export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

echo "=========================================="
echo "MS-CCE 混合训练 (kernel=[1,3,5], chunk=16)"
echo "=========================================="
echo "工作目录: ${workspace}"
echo "使用GPU: ${CUDA_VISIBLE_DEVICES} (${gpu_num}张)"
echo "时间: $(date)"
echo ""

model_name_or_model_dir="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online"

train_data="/home/FunASR/FunASR-main/data/mixed/train.jsonl"
val_data="/home/FunASR/FunASR-main/data/mixed/val.jsonl"

if [ ! -f "$train_data" ]; then
    echo "训练数据不存在: $train_data"
    exit 1
fi

if [ ! -f "$val_data" ]; then
    echo "验证数据不存在: $val_data"
    exit 1
fi

echo "数据通过:"
echo "  训练: $train_data ($(wc -l < $train_data) 条, 43%枪声 + 57%THCHS-30)"
echo "  验证: $val_data ($(wc -l < $val_data) 条)"
echo ""

output_dir="/home/FunASR/FunASR-main/outputs_ms_cce_mixed_k135_n4"
log_file="${output_dir}/log.txt"

mkdir -p ${output_dir}
echo "输出: ${output_dir}"
echo ""

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-29506}
"

echo "=========================================="
echo "训练参数"
echo "=========================================="
echo "  模型类:    ParaformerStreamingMSCce"
echo "  数据:      混合 (43%枪声 + 57%THCHS-30)"
echo "  冻结:      decoder, predictor, ctc"
echo "  训练:      Encoder全40层 + MS-CCE(k=[1,3,5])"
echo "  LR:        5e-5"
echo "  Batch:     8000 tokens"
echo "  Epochs:    10"
echo "  Warmup:    3000"
echo "  CCE:       dim=560, kernels=[1,3,5], fixed_w=0.8, chunk=16"
echo "=========================================="
echo ""

echo "启动训练..."
echo ""

nohup torchrun $DISTRIBUTED_ARGS \
./funasr/bin/train_ds.py \
++model="${model_name_or_model_dir}" \
++model_class="ParaformerStreamingMSCce" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset="AudioDataset" \
++dataset_conf.index_ds="IndexDSJsonl" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=8000 \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=2 \
++train_conf.max_epoch=20 \
++train_conf.log_interval=50 \
++train_conf.resume=true \
++frontend_conf.lfr_n=4 \
++train_conf.validate_interval=1000 \
++train_conf.save_checkpoint_interval=1000 \
++train_conf.keep_nbest_models=5 \
++train_conf.avg_nbest_model=5 \
++train_conf.use_deepspeed=false \
++optim="adam" \
++optim_conf.lr=0.00005 \
++scheduler="warmuplr" \
++scheduler_conf.warmup_steps=3000 \
++use_cce=true \
++cce_dim=560 \
++cce_kernel_sizes='[1,3,5]' \
++cce_fixed_weight=0.8 \
++cce_chunk_size=16 \
++output_dir="${output_dir}" >> "${log_file}" 2>&1 &








