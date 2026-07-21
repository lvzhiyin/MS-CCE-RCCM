#!/usr/bin/env python3
"""
生成30小时枪声融合数据
- 1,452条训练音频 × 25变体 = 36,300条 (30.25小时)
- 255条验证音频 × 8变体 = 2,040条
- 257条测试音频 × 8变体 = 2,056条
- 插入位置限制在0-80%，确保时长不变
- 生成对应的标签文件
"""

import os
import glob
import csv
import argparse
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ================= SNR档位配置 =================
SNR_BUCKETS = [
    {'name': 'extreme', 'range': (-15, -15), 'weight': 0.05, 'desc': '极端嘈杂'},
    {'name': 'core', 'range': (-10, 0), 'weight': 0.55, 'desc': '核心区间'},
    {'name': 'moderate', 'range': (0, 5), 'weight': 0.30, 'desc': '较易识别'},
    {'name': 'quiet', 'range': (5, 15), 'weight': 0.10, 'desc': '安静对照'},
]


def sample_snr():
    """按权重随机采样SNR"""
    weights = [b['weight'] for b in SNR_BUCKETS]
    bucket = np.random.choice(SNR_BUCKETS, p=weights)
    snr_min, snr_max = bucket['range']
    snr = np.random.uniform(snr_min, snr_max)
    return snr, bucket['name']


def load_labels(csv_path):
    """加载CSV标签文件"""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                audio_id = row[0].replace('.wav', '')
                text = row[1].strip()
                labels[audio_id] = text
    return labels


def load_gunshots(noise_dir, sr=16000):
    """加载枪声音频"""
    noise_files = glob.glob(os.path.join(noise_dir, '*.wav'))
    noise_pool = []
    for f in noise_files:
        wav, _ = librosa.load(f, sr=sr, mono=True)
        if len(wav) > 0:
            noise_pool.append({
                'audio': wav,
                'name': Path(f).stem
            })
    return noise_pool


def mix_gunshot(clean, gunshot_info, sr=16000, target_snr_db=None, fade_ms=10):
    """
    将枪声混入干净音频
    
    Args:
        clean: 干净音频数组
        gunshot_info: 枪声音频信息 {'audio': ..., 'name': ...}
        sr: 采样率
        target_snr_db: 目标SNR
        fade_ms: 淡入淡出时间(毫秒)
    
    Returns:
        mixed: 混合后的音频
        info: 混合信息字典
    """
    cmd_len = len(clean)
    gun = gunshot_info['audio'].copy()
    gun_len = len(gun)
    
    # 枪声不能超过指令长度的50%
    if gun_len >= cmd_len * 0.5:
        return None, None
    
    # 采样SNR
    if target_snr_db is None:
        target_snr_db, bucket_name = sample_snr()
    else:
        bucket_name = 'specified'
    
    # 计算整段指令的RMS功率
    cmd_rms = np.sqrt(np.mean(clean ** 2))
    gun_rms = np.sqrt(np.mean(gun ** 2))
    
    # 计算目标枪声RMS
    gun_rms_target = cmd_rms / (10 ** (target_snr_db / 20.0))
    
    # 计算增益
    gain = gun_rms_target / (gun_rms + 1e-10)
    gun_scaled = gun * gain
    
    # 添加淡入淡出
    fade_samples = int(fade_ms * sr / 1000)
    if len(gun_scaled) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        gun_scaled[:fade_samples] *= fade_in
        gun_scaled[-fade_samples:] *= fade_out
    
    # 插入位置：0% - 80%，确保不超出
    max_pos = int(cmd_len * 0.8) - gun_len
    if max_pos <= 0:
        return None, None
    
    pos = np.random.randint(0, max_pos + 1)
    
    # 混合
    mixed = clean.copy()
    mixed[pos:pos+gun_len] += gun_scaled
    
    # 防削波
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
    
    info = {
        'position': pos,
        'snr_db': target_snr_db,
        'bucket': bucket_name,
        'gun_name': gunshot_info['name'],
        'gain': gain,
    }
    
    return mixed, info


def process_dataset(clean_dir, label_file, gunshot_pool, output_dir, 
                    num_variants, sr=16000, max_duration=3.0):
    """
    处理一个数据集（train/val/test）
    
    Returns:
        生成的音频数量
    """
    # 创建输出目录
    audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    
    # 加载标签
    labels = load_labels(label_file)
    print(f"  加载了 {len(labels)} 条标签")
    
    # 获取干净音频列表
    clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
    print(f"  找到 {len(clean_files)} 个干净音频")
    
    # 准备标签文件
    label_output_path = os.path.join(output_dir, 'labels.txt')
    label_lines = []
    
    max_samples = int(max_duration * sr)
    generated_count = 0
    
    # 处理每个干净音频
    for clean_path in tqdm(clean_files, desc="处理音频"):
        clean_name = Path(clean_path).stem
        
        # 获取标签
        if clean_name not in labels:
            continue
        text = labels[clean_name]
        
        # 加载干净音频
        clean, _ = librosa.load(clean_path, sr=sr, mono=True)
        
        # 截断到最大长度
        if len(clean) > max_samples:
            clean = clean[:max_samples]
        
        # 跳过太短的音频
        if len(clean) < sr * 0.5:
            continue
        
        # 生成多个变体
        for variant_idx in range(num_variants):
            # 随机选择枪声
            gun_info = np.random.choice(gunshot_pool)
            
            # 混合
            mixed, mix_info = mix_gunshot(clean, gun_info, sr)
            
            if mixed is not None:
                # 生成文件名
                snr_tag = f"snr{mix_info['snr_db']:.0f}"
                out_name = f"{clean_name}_{gun_info['name']}_{snr_tag}_v{variant_idx}"
                
                # 保存音频
                out_path = os.path.join(audio_dir, f"{out_name}.wav")
                sf.write(out_path, mixed, sr, subtype='PCM_16')
                
                # 记录标签
                label_lines.append(f"{out_name}.wav,{text}")
                generated_count += 1
    
    # 保存标签文件
    with open(label_output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_lines))
    
    print(f"  生成了 {generated_count} 条音频")
    print(f"  标签文件: {label_output_path}")
    
    return generated_count


def main():
    parser = argparse.ArgumentParser(description='生成30小时枪声融合数据')
    parser.add_argument('--data-root', default='/home/FunASR/FunASR-main/data',
                        help='数据根目录')
    parser.add_argument('--output-dir', default='/home/FunASR/FunASR-main/data/gunshot_30h',
                        help='输出目录')
    parser.add_argument('--train-variants', type=int, default=25,
                        help='训练集每个音频的变体数')
    parser.add_argument('--val-variants', type=int, default=8,
                        help='验证集每个音频的变体数')
    parser.add_argument('--test-variants', type=int, default=8,
                        help='测试集每个音频的变体数')
    parser.add_argument('--sr', type=int, default=16000, help='采样率')
    args = parser.parse_args()
    
    print("="*60)
    print("生成30小时枪声融合数据")
    print("="*60)
    
    # 路径配置
    gunshot_dir = os.path.join(args.data_root, 'raw', 'gunsounds')
    
    train_clean = os.path.join(args.data_root, 'raw', 'command', 'train')
    train_label = os.path.join(args.data_root, 'raw', 'command', 'train_labels.csv')
    
    val_clean = os.path.join(args.data_root, 'raw', 'command', 'val')
    val_label = os.path.join(args.data_root, 'raw', 'command', 'val_labels.csv')
    
    test_clean = os.path.join(args.data_root, 'raw', 'command', 'test')
    test_label = os.path.join(args.data_root, 'raw', 'command', 'test_labels.csv')
    
    # 加载枪声
    print("\n[1] 加载枪声音频...")
    gunshot_pool = load_gunshots(gunshot_dir, args.sr)
    print(f"    加载了 {len(gunshot_pool)} 个枪声")
    
    # 处理训练集
    print("\n[2] 处理训练集...")
    train_output = os.path.join(args.output_dir, 'train')
    train_count = process_dataset(
        train_clean, train_label, gunshot_pool, train_output,
        args.train_variants, args.sr
    )
    
    # 处理验证集
    print("\n[3] 处理验证集...")
    val_output = os.path.join(args.output_dir, 'val')
    val_count = process_dataset(
        val_clean, val_label, gunshot_pool, val_output,
        args.val_variants, args.sr
    )
    
    # 处理测试集
    print("\n[4] 处理测试集...")
    test_output = os.path.join(args.output_dir, 'test')
    test_count = process_dataset(
        test_clean, test_label, gunshot_pool, test_output,
        args.test_variants, args.sr
    )
    
    # 统计
    total_count = train_count + val_count + test_count
    total_hours = total_count * 3.0 / 3600
    
    print("\n" + "="*60)
    print("生成完成!")
    print("="*60)
    print(f"训练集: {train_count} 条 ({train_count*3/3600:.1f} 小时)")
    print(f"验证集: {val_count} 条 ({val_count*3/3600:.1f} 小时)")
    print(f"测试集: {test_count} 条 ({test_count*3/3600:.1f} 小时)")
    print(f"总计: {total_count} 条 ({total_hours:.1f} 小时)")
    print(f"\n输出目录: {args.output_dir}")
    print(f"预计存储: {total_count * 96 / 1024 / 1024:.1f} GB")


if __name__ == '__main__':
    main()
