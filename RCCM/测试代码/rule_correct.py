#!/usr/bin/env python3
"""
规则后处理纠正 V3 — 先检测后纠正，宁漏不杀

直接读取 ASR 测试结果 JSON，对 hyp 做规则纠正后重新计算 CER/CA。
不需要训练集 —— 合法指令从测试集 ref 字段自动提取。
"""

import json
import argparse
from collections import Counter
from pypinyin import lazy_pinyin, Style


# ── 热词表 & CER/CA 计算 (与 test_baseline.py 完全一致) ──

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
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1))
    return dp[m][n] / len(ref)


def extract_hotword_seq(text):
    seq = []
    i = 0
    while i < len(text):
        matched = False
        for hw in sorted(HOTWORDS, key=len, reverse=True):
            if text[i:i + len(hw)] == hw:
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
            cost = 0 if ref_seq[i - 1] == hyp_seq[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def compute_hotword_recall(ref, hyp):
    ref_seq = extract_hotword_seq(ref)
    hyp_seq = extract_hotword_seq(hyp)
    if not ref_seq:
        return 1.0
    dist = seq_edit_distance(ref_seq, hyp_seq)
    return max(0, 1 - dist / len(ref_seq))


# ── 词表构建 (仅用 HOTWORDS, 不从测试集提取) ─────────────

def build_lexicon_from_hotwords():
    """仅用 28 个热词 + 合法 2 词组合，不生成三词组合避免爆炸"""
    numbers = [hw for hw in HOTWORDS if '号' in hw]
    actions = [hw for hw in HOTWORDS if '号' not in hw]

    two_word_combos = set()
    for num in numbers:
        for act in actions:
            two_word_combos.add(num + act)
    for a1 in actions:
        for a2 in actions:
            if a1 != a2:
                two_word_combos.add(a1 + a2)

    all_commands = set(HOTWORDS) | two_word_combos

    char_counter = Counter()
    bigram_counter = Counter()
    for cmd in all_commands:
        for c in cmd:
            char_counter[c] += 1
        for i in range(len(cmd) - 1):
            bigram_counter[cmd[i:i + 2]] += 1

    pinyin_to_chars = {}
    for char in char_counter:
        py = lazy_pinyin(char, style=Style.NORMAL)[0]
        pinyin_to_chars.setdefault(py, set()).add(char)

    # 按长度分桶, 加速 fix_by_vocabulary 查找
    length_buckets = {}
    for cmd in all_commands:
        length_buckets.setdefault(len(cmd), []).append(cmd)

    return {
        'commands': all_commands,
        'char_freq': char_counter,
        'bigram_freq': bigram_counter,
        'pinyin_to_chars': pinyin_to_chars,
        'length_buckets': length_buckets,
    }


# ── 异常检测 ──────────────────────────────────────────────

def detect_repeats(text):
    """检测连续重复字: [(位置, 重复次数, 重复字)]"""
    repeats = []
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and text[j] == text[i]:
            j += 1
        if j - i >= 2:
            repeats.append((i, j - i, text[i]))
        i = j
    return repeats


def is_valid_bigram(bigram, lexicon):
    return bigram in lexicon['bigram_freq']


# ── 纠正三步 ──────────────────────────────────────────────

def fix_repeats(text, lexicon):
    """连续重复字 → 去重为1个 (叠词如 '看看' 保留)"""
    repeats = detect_repeats(text)
    if not repeats:
        return text, False
    chars = list(text)
    fixed = False
    common_verb_repeats = {'看', '听', '说', '走', '跑', '好', '慢', '快', '小', '大', '明'}
    for pos, count, char in reversed(repeats):
        if char in common_verb_repeats:
            continue  # 合法叠词保留
        del chars[pos + 1:pos + count]
        fixed = True
    return ''.join(chars), fixed


def fix_by_vocabulary(text, lexicon):
    """编辑距离 ≤1 且长度差 ≤1 → 替换为合法指令 (用长度分桶加速)"""
    commands = lexicon['commands']
    if text in commands:
        return text, False

    length_buckets = lexicon['length_buckets']
    best_cmd, best_dist = '', float('inf')

    # 只搜索与 text 长度差 ≤1 的 bucket
    for L in range(len(text) - 1, len(text) + 2):
        for cmd in length_buckets.get(L, []):
            if cmd == text:
                return text, False
            m, n = len(text), len(cmd)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if text[i - 1] == cmd[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            d = dp[m][n]
            if d < best_dist:
                best_dist = d
                best_cmd = cmd
            if d == 0:
                break  # 精确匹配，无需继续

    if best_dist <= 1 and best_cmd and abs(len(best_cmd) - len(text)) <= 1:
        return best_cmd, True
    return text, False


def fix_homophone_chunks(text, lexicon):
    """同音字替换: 替换后 bigram 匹配度提升才换"""
    if not text or text in lexicon['commands']:
        return text, False

    original_bigrams = sum(1 for i in range(len(text) - 1)
                          if is_valid_bigram(text[i:i + 2], lexicon))
    chars = list(text)
    pinyins = lazy_pinyin(text, style=Style.NORMAL)
    fixed = False

    for i, (c, py) in enumerate(zip(chars, pinyins)):
        candidates = lexicon['pinyin_to_chars'].get(py, {c})
        if len(candidates) <= 1:
            continue
        best_char, best_score = c, original_bigrams
        for cand in candidates:
            if cand == c:
                continue
            test_chars = chars[:]
            test_chars[i] = cand
            test_text = ''.join(test_chars)
            score = sum(1 for j in range(len(test_text) - 1)
                       if is_valid_bigram(test_text[j:j + 2], lexicon))
            if test_text in lexicon['commands']:
                score += 10
            if score > best_score:
                best_score = score
                best_char = cand
        if best_char != c:
            chars[i] = best_char
            fixed = True

    return ''.join(chars), fixed


def correct_text(text, lexicon):
    did_fix = False
    text, f1 = fix_repeats(text, lexicon)
    did_fix = did_fix or f1
    text, f2 = fix_by_vocabulary(text, lexicon)
    did_fix = did_fix or f2
    if text not in lexicon['commands']:
        text, f3 = fix_homophone_chunks(text, lexicon)
        did_fix = did_fix or f3
    return text, did_fix


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr-results', required=True, help='ASR 测试结果 JSON')
    parser.add_argument('--output', default='rule_corrected_result.json')
    args = parser.parse_args()

    print("构建词表 (仅用 HOTWORDS + 合法组合)...")
    lexicon = build_lexicon_from_hotwords()
    print(f"  合法指令: {len(lexicon['commands'])} 条")
    print(f"  字符: {len(lexicon['char_freq'])} 个, bigram: {len(lexicon['bigram_freq'])} 个")
    print()

    with open(args.asr_results, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = data.get('details', [])
    print(f"样本数: {len(details)}")
    print()

    results = []
    n, n_corrected = 0, 0
    n_improved, n_degraded, n_unchanged = 0, 0, 0
    total_cer_before, total_cer_after = 0, 0
    total_ca_before, total_ca_after = 0, 0

    for item in details:
        ref = item.get('ref', '').strip()
        pred = item.get('hyp', '').strip()
        if not ref:
            continue
        n += 1

        corrected, did_fix = correct_text(pred, lexicon)
        if did_fix:
            n_corrected += 1

        cer_before = compute_cer(ref, pred) * 100
        cer_after = compute_cer(ref, corrected) * 100
        ca_before = compute_hotword_recall(ref, pred) * 100
        ca_after = compute_hotword_recall(ref, corrected) * 100

        if cer_after < cer_before - 0.01:
            n_improved += 1
        elif cer_after > cer_before + 0.01:
            n_degraded += 1
        else:
            n_unchanged += 1

        total_cer_before += cer_before
        total_cer_after += cer_after
        total_ca_before += ca_before
        total_ca_after += ca_after

        results.append({
            'ref': ref, 'pred': pred, 'corrected': corrected,
            'did_fix': did_fix,
            'cer_before': round(cer_before, 2),
            'cer_after': round(cer_after, 2),
            'ca_before': round(ca_before, 2),
            'ca_after': round(ca_after, 2),
        })

    avg_cer_before = total_cer_before / n
    avg_cer_after = total_cer_after / n
    avg_ca_before = total_ca_before / n
    avg_ca_after = total_ca_after / n

    result = {
        'method': 'rule_detect_fix_v3',
        'num_samples': n,
        'num_corrected': n_corrected,
        'n_improved': n_improved,
        'n_degraded': n_degraded,
        'n_unchanged': n_unchanged,
        'cer_before': round(avg_cer_before, 2),
        'cer_after': round(avg_cer_after, 2),
        'ca_before': round(avg_ca_before, 2),
        'ca_after': round(avg_ca_after, 2),
        'cer_delta': round(avg_cer_before - avg_cer_after, 2),
        'ca_delta': round(avg_ca_after - avg_ca_before, 2),
        'details': results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{'='*60}")
    print(f"总样本:    {n}")
    print(f"触发纠正:  {n_corrected} ({n_corrected/n*100:.1f}%)")
    print(f"  提升:    {n_improved}")
    print(f"  持平:    {n_unchanged}")
    print(f"  变差:    {n_degraded}")
    print(f"{'='*60}")
    print(f"原始 CER:  {avg_cer_before:.2f}%")
    print(f"纠正 CER:  {avg_cer_after:.2f}%")
    print(f"CER Δ:     {avg_cer_before - avg_cer_after:+.2f} pp")
    print(f"原始 CA:   {avg_ca_before:.2f}%")
    print(f"纠正 CA:   {avg_ca_after:.2f}%")
    print(f"CA  Δ:     {avg_ca_after - avg_ca_before:+.2f} pp")
    print(f"\n输出: {args.output}")


if __name__ == '__main__':
    main()
