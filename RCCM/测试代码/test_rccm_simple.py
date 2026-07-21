# #!/usr/bin/env python3
# """SACM纠错测试 - 从ASR结果JSON读取并统计CER/热词/延迟"""
# import sys, os, json, time, torch

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from light_rccm import LightSACM, CharTokenizer

# HOTWORDS = {'三号','七号','四号','站立','旋转','前进','暂停','匍匐','九号','停止','向右','倒靶','后退','向左','卧倒','右转','开始','立靶','六号','射击','放靶','二号','冲击','十号','八号','五号','一号','跃进','左转'}

# def cer(r, h):
#     r, h = r.replace(' ',''), h.replace(' ','')
#     if not r: return 0.0 if not h else 1.0
#     m, n = len(r), len(h)
#     dp = [[0]*(n+1) for _ in range(m+1)]
#     for i in range(m+1): dp[i][0] = i
#     for j in range(n+1): dp[0][j] = j
#     for i in range(1,m+1):
#         for j in range(1,n+1):
#             dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
#     return dp[m][n]/len(r)


# def extract_hotword_seq(text):
#     """按顺序提取热词序列（与 test_baseline.py 一致）"""
#     seq = []
#     i = 0
#     while i < len(text):
#         matched = False
#         for hw in sorted(HOTWORDS, key=len, reverse=True):
#             if text[i:i+len(hw)] == hw:
#                 seq.append(hw)
#                 i += len(hw)
#                 matched = True
#                 break
#         if not matched:
#             i += 1
#     return seq


# def seq_edit_distance(ref_seq, hyp_seq):
#     """计算热词序列编辑距离"""
#     m, n = len(ref_seq), len(hyp_seq)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     for i in range(m + 1):
#         dp[i][0] = i
#     for j in range(n + 1):
#         dp[0][j] = j
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             cost = 0 if ref_seq[i-1] == hyp_seq[j-1] else 1
#             dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
#     return dp[m][n]


# def hwr(r, h):
#     """基于热词序列编辑距离的HRR（与 test_baseline.py 一致）"""
#     ref_seq = extract_hotword_seq(r)
#     hyp_seq = extract_hotword_seq(h)
#     if not ref_seq:
#         return 1.0
#     dist = seq_edit_distance(ref_seq, hyp_seq)
#     return max(0, 1 - dist / len(ref_seq))

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--rccm-model', required=True)
# parser.add_argument('--asr-results', required=True)
# parser.add_argument('--pairs', default=None, help='训练对JSONL，用于重建词表')
# parser.add_argument('--use-trie', action='store_true', help='启用Trie约束解码')
# parser.add_argument('--device', default='cuda')
# args = parser.parse_args()

# print(f"加载ASR结果: {args.asr_results}")
# with open(args.asr_results) as f:
#     data = json.load(f)
# samples = data['details']
# print(f"  样本: {len(samples)}")

# print(f"\n加载模型: {args.rccm_model}")
# ckpt = torch.load(args.rccm_model, map_location=args.device)
# cfg = ckpt.get('config', {})
# sd = ckpt['model_state_dict']
# tokenizer = ckpt.get('tokenizer')
# if tokenizer is None:
#     # 从训练数据重建词表
#     if not args.pairs or not os.path.exists(args.pairs):
#         print("❌ checkpoint 无 tokenizer, 请用 --pairs 指定训练对文件"); sys.exit(1)
#     print(f"  从训练数据重建词表: {args.pairs}")
#     from light_rccm import CharTokenizer
#     texts = set()
#     with open(args.pairs, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line.strip())
#             for k in ('tgt', 'target', 'clean'):
#                 t = item.get(k, '')
#                 if t: texts.add(t); break
#     tokenizer = CharTokenizer(max_vocab=3000)
#     tokenizer.load_vocab(list(texts), max_vocab=2996)
# vocab_size = len(tokenizer.token2id)

# # 从 state_dict 自动推断模型维度（兼容无 config 的 checkpoint）
# _d_model = cfg.get('d_model', sd['embedding.weight'].shape[1])
# _nhead = cfg.get('nhead', 4)
# _nlayers = cfg.get('num_encoder_layers', len(set(int(k.split('.')[2]) for k in sd if k.startswith('encoder.layers.'))) or 4)
# _ff = cfg.get('dim_feedforward', sd['encoder.layers.0.linear1.weight'].shape[0])
# print(f"  推断: d={_d_model}, h={_nhead}, L={_nlayers}, ff={_ff}")
# # 自动检测是否有 error_detector
# _use_det = cfg.get('use_error_detector', 
#                    any(k.startswith('error_detector.') for k in sd))
# model = LightSACM(
#     vocab_size=vocab_size, d_model=_d_model,
#     nhead=_nhead,
#     num_encoder_layers=_nlayers, num_decoder_layers=_nlayers,
#     dim_feedforward=_ff,
#     max_seq_len=cfg.get('max_seq_len',30), dropout=0.0,
#     use_error_detector=_use_det,
# )
# model.load_state_dict(ckpt['model_state_dict'])
# model.to(args.device)
# model.eval()
# print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")

# # === 构建 Trie 约束解码 ===
# trie = None
# trie_cleaner = None
# if args.use_trie:
#     from light_rccm import CommandTrie
#     trie = CommandTrie.from_hotwords(sorted(HOTWORDS))
#     print(f"  Trie: {len(HOTWORDS)} 条热词约束")
    
#     # After generation, greedily clean output to valid hotword sequence
#     def clean_to_hotwords(text, trie):
#         """后处理: 贪心提取合法热词序列"""
#         result = ""
#         i = 0
#         while i < len(text):
#             if text[i] not in trie.start_chars:
#                 i += 1
#                 continue
#             # 尝试在 Trie 中匹配最长热词
#             node = trie.root
#             best_end = -1
#             j = i
#             while j < len(text) and text[j] in node:
#                 node = node[text[j]]
#                 j += 1
#                 if '__END__' in node:
#                     best_end = j
#             if best_end > i:
#                 result += text[i:best_end]
#                 i = best_end
#             else:
#                 # 无法匹配完整热词，但首字符在 start_chars 中则保留首字符
#                 if text[i] in trie.start_chars:
#                     node = trie.root
#                     j = i
#                     while j < len(text) and text[j] in node:
#                         node = node[text[j]]
#                         result += text[j]
#                         j += 1
#                     i = j
#                 else:
#                     i += 1
#         return result
#     trie_cleaner = clean_to_hotwords

# label_extra = " (Trie约束)" if trie else ""
# cer_b, cer_a = [], []
# hb, ha = [], []
# total_t = 0.0
# better = worse = same = 0

# for i, s in enumerate(samples):
#     hyp = s['hyp']
#     ref = s['ref']

#     src_ids = torch.tensor([tokenizer.encode(hyp)[:30]], dtype=torch.long).to(args.device)
#     t0 = time.perf_counter()
#     with torch.no_grad():
#         out_ids = model.generate(src_ids, tokenizer, max_len=30, device=args.device)
#     t1 = time.perf_counter()
#     total_t += t1-t0
#     corrected = tokenizer.decode(out_ids[0].cpu().numpy())
#     if trie_cleaner:
#         corrected = trie_cleaner(corrected, trie)

#     cb = cer(ref, hyp)
#     ca = cer(ref, corrected)
#     cer_b.append(cb); cer_a.append(ca)
#     hb.append(hwr(ref, hyp)); ha.append(hwr(ref, corrected))
#     if ca < cb: better += 1
#     elif ca > cb: worse += 1
#     else: same += 1

# n = len(cer_b)
# print(f"\n{'='*60}")
# print(f"结果 ({n}条){label_extra}")
# print(f"{'='*60}")
# print(f"CER (纠正前):     {sum(cer_b)/n*100:.2f}%")
# print(f"CER (纠正后):     {sum(cer_a)/n*100:.2f}%")
# print(f"CER 降低:         {(sum(cer_b)-sum(cer_a))/n*100:.2f}%")
# print(f"热词召回 (纠正前): {sum(hb)/n*100:.2f}%")
# print(f"热词召回 (纠正后): {sum(ha)/n*100:.2f}%")
# print(f"平均延迟:         {total_t/n*1000:.2f}ms/条")
# print(f"更好/更差/相同:    {better}/{worse}/{same}")







# #!/usr/bin/env python3
# """SACM纠错测试 - 支持全句纠错 / 伪流式分块纠错"""
# import sys, os, json, time, torch

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from light_rccm import LightSACM, CharTokenizer

# HOTWORDS = {'三号','七号','四号','站立','旋转','前进','暂停','匍匐','九号','停止','向右','倒靶','后退','向左','卧倒','右转','开始','立靶','六号','射击','放靶','二号','冲击','十号','八号','五号','一号','跃进','左转'}

# def cer(r, h):
#     r, h = r.replace(' ',''), h.replace(' ','')
#     if not r: return 0.0 if not h else 1.0
#     m, n = len(r), len(h)
#     dp = [[0]*(n+1) for _ in range(m+1)]
#     for i in range(m+1): dp[i][0] = i
#     for j in range(n+1): dp[0][j] = j
#     for i in range(1,m+1):
#         for j in range(1,n+1):
#             dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
#     return dp[m][n]/len(r)


# def extract_hotword_seq(text):
#     seq = []
#     i = 0
#     while i < len(text):
#         matched = False
#         for hw in sorted(HOTWORDS, key=len, reverse=True):
#             if text[i:i+len(hw)] == hw:
#                 seq.append(hw); i += len(hw); matched = True; break
#         if not matched:
#             i += 1
#     return seq


# def seq_edit_distance(ref_seq, hyp_seq):
#     m, n = len(ref_seq), len(hyp_seq)
#     dp = [[0]*(n+1) for _ in range(m+1)]
#     for i in range(m+1): dp[i][0] = i
#     for j in range(n+1): dp[0][j] = j
#     for i in range(1,m+1):
#         for j in range(1,n+1):
#             c = 0 if ref_seq[i-1]==hyp_seq[j-1] else 1
#             dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
#     return dp[m][n]


# def hwr(r, h):
#     ref_seq = extract_hotword_seq(r)
#     hyp_seq = extract_hotword_seq(h)
#     if not ref_seq: return 1.0
#     dist = seq_edit_distance(ref_seq, hyp_seq)
#     return max(0, 1 - dist / len(ref_seq))


# # ========== 连续重复字去重 ==========
# def dedup_consecutive(text):
#     """去掉连续重复字: "一号前前前进" → "一号前进" """
#     if not text:
#         return text
#     result = [text[0]]
#     for ch in text[1:]:
#         if ch != result[-1]:
#             result.append(ch)
#     return ''.join(result)


# def smart_dedup_chunked(text, chunk_size=4):
#     """
#     逐块去重 + 跨块边界去重:
#     1. 切分为 chunk_size 字一块
#     2. 每块内部去重
#     3. 和记忆库最新块拼接 → 整体去重 → 放回记忆库
#     4. 最终整体去重
#     """
#     if len(text) <= chunk_size:
#         return dedup_consecutive(text)
    
#     chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
#     memory = dedup_consecutive(chunks[0])  # 第一个块去重后放入记忆
    
#     for ch in chunks[1:]:
#         ch_dedup = dedup_consecutive(ch)                # 块内去重
#         merged = memory + ch_dedup                       # 拼接最新记忆
#         memory = dedup_consecutive(merged)               # 跨块边界去重
    
#     return memory


# # ========== 伪流式 + 边缘补偿纠错 ==========
# def streaming_correct(model, tokenizer, raw_text, chunk_size=4, device='cuda',
#                       trie=None, conservative=False, dedup=False, smart_dedup=False,
#                       edge_width=1):
#     """
#     伪流式边缘补偿纠错 (邓飞腾论文 4.1.2):
#     1. 切 chunk_size 字一块
#     2. 第一块: 解码但不输出, 等下一块联合确认
#     3. 后续块: prev_raw + cur_raw 联合解码 → 确认前块 → 当前块右边缘截断
#     4. 句尾 flush: 输出最后一块剩余
#     """

#     def decode_one(text):
#         pids = tokenizer.encode(text)[:model.max_seq_len]
#         src = torch.tensor([pids], dtype=torch.long, device=device)
#         t0 = time.perf_counter()
#         with torch.no_grad():
#             out_ids = model.generate(src, tokenizer, max_len=model.max_seq_len,
#                                      device=device, trie=trie)
#         return tokenizer.decode(out_ids[0].cpu().numpy()), (time.perf_counter() - t0) * 1000

#     if smart_dedup:
#         raw_text = smart_dedup_chunked(raw_text, chunk_size=args.dedup_chunk_size)
#     elif dedup:
#         raw_text = dedup_consecutive(raw_text)

#     total_latency = 0.0
#     confirmed_parts = []
#     prev_raw = ""
#     prev_corr = ""

#     # 切块
#     chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]

#     for i, cur_raw in enumerate(chunks):
#         if i == 0:
#             # 第一块：解码但不输出，等下一块联合确认
#             cur_corr, lat = decode_one(cur_raw)
#             total_latency += lat
#             prev_raw = cur_raw
#             prev_corr = cur_corr
#         else:
#             # 联合解码: prev_raw + cur_raw
#             joint = prev_raw + cur_raw
#             joint_corr, lat = decode_one(joint)
#             total_latency += lat

#             # 分离前一块修正结果
#             confirmed = joint_corr[:len(prev_raw)]
#             if confirmed:
#                 confirmed_parts.append(confirmed)

#             # 当前块修正结果: 截断右边缘 edge_width 字（留给下一块确认）
#             cur_corr = joint_corr[len(prev_raw):]
#             if len(cur_corr) > edge_width:
#                 cur_corr = cur_corr[:-edge_width]

#             prev_raw = cur_raw
#             prev_corr = cur_corr

#     # flush: 最后一块直接输出
#     if prev_corr:
#         confirmed_parts.append(prev_corr)

#     final_result = "".join(confirmed_parts)

#     # 保守门控
#     if conservative and final_result != raw_text:
#         edits_val = cer(raw_text, final_result) * max(len(raw_text), 1)
#         if edits_val > len(raw_text) * 0.5:
#             final_result = raw_text
#     if smart_dedup or dedup:
#         final_result = dedup_consecutive(final_result)
#     return final_result, total_latency


# # ========== 参数解析 ==========
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--rccm-model', required=True)
# parser.add_argument('--asr-results', required=True)
# parser.add_argument('--pairs', default=None)
# parser.add_argument('--use-trie', action='store_true', help='启用Trie约束解码（限制输出为合法指令组合）')
# parser.add_argument('--conservative', action='store_true', help='保守纠错: SACM改动>50%时回退ASR原文')
# parser.add_argument('--dedup', action='store_true', help='全局去连续重复字 (如"前前前进"→"前进")')
# parser.add_argument('--smart-dedup', action='store_true', help='逐块去重+跨块边界去重(先切4字块,块内去重,拼接记忆再跨块去重)')
# parser.add_argument('--dedup-chunk-size', type=int, default=4, help='smart-dedup块大小')
# parser.add_argument('--streaming', action='store_true', help='启用伪流式分块纠错')
# parser.add_argument('--chunk-size', type=int, default=4, help='流式窗口字数 (默认4)')
# parser.add_argument('--device', default='cuda')
# args = parser.parse_args()

# # ========== 构建 Trie（从 30 条热词） ==========
# trie = None
# if args.use_trie:
#     from light_rccm import CommandTrie
#     trie = CommandTrie(sorted(HOTWORDS))
#     print(f"  Trie约束解码: {len(HOTWORDS)} 条指令前缀树")

# print(f"加载ASR结果: {args.asr_results}")
# with open(args.asr_results) as f:
#     data = json.load(f)
# samples = data['details']
# print(f"  样本: {len(samples)}")

# print(f"\n加载模型: {args.rccm_model}")
# ckpt = torch.load(args.rccm_model, map_location=args.device)
# cfg = ckpt.get('config', {})
# sd = ckpt['model_state_dict']
# tokenizer = ckpt.get('tokenizer')
# if tokenizer is None:
#     if not args.pairs or not os.path.exists(args.pairs):
#         print("ERROR checkpoint 无 tokenizer, 请用 --pairs 指定训练对文件"); sys.exit(1)
#     print(f"  从训练数据重建词表: {args.pairs}")
#     texts = set()
#     with open(args.pairs, 'r', encoding='utf-8') as f:
#         for line in f:
#             item = json.loads(line.strip())
#             for k in ('tgt', 'target', 'clean'):
#                 t = item.get(k, '')
#                 if t: texts.add(t); break
#     tokenizer = CharTokenizer(max_vocab=3000)
#     tokenizer.load_vocab(list(texts), max_vocab=2996)
# vocab_size = len(tokenizer.token2id)

# _d_model = cfg.get('d_model', sd['embedding.weight'].shape[1])
# _nhead = cfg.get('nhead', 4)
# _nlayers = cfg.get('num_encoder_layers', len(set(int(k.split('.')[2]) for k in sd if k.startswith('encoder.layers.'))) or 4)
# _ff = cfg.get('dim_feedforward', sd['encoder.layers.0.linear1.weight'].shape[0])
# print(f"  推断: d={_d_model}, h={_nhead}, L={_nlayers}, ff={_ff}")
# _use_det = cfg.get('use_error_detector', any(k.startswith('error_detector.') for k in sd))
# model = LightSACM(
#     vocab_size=vocab_size, d_model=_d_model, nhead=_nhead,
#     num_encoder_layers=_nlayers, num_decoder_layers=_nlayers,
#     dim_feedforward=_ff, max_seq_len=cfg.get('max_seq_len',30), dropout=0.0,
#     use_error_detector=_use_det,
# )
# model.load_state_dict(ckpt['model_state_dict'])
# model.to(args.device)
# model.eval()
# print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")


# def run_test(mode_label, correct_fn):
#     """通用测试循环: correct_fn(hyp) -> corrected_text, latency_ms"""
#     cer_b, cer_a = [], []
#     hb_, ha_ = [], []
#     total_t = 0.0
#     better = worse = same = 0
#     bad_cases = []   # 纠错后更差
#     good_cases = []  # 纠错后更好
#     for s in samples:
#         hyp = s['hyp']; ref = s['ref']
#         ct, lat = correct_fn(hyp)
#         total_t += lat / 1000  # ms → s
#         cb = cer(ref, hyp); ca = cer(ref, ct)
#         cer_b.append(cb); cer_a.append(ca)
#         hb_.append(hwr(ref, hyp)); ha_.append(hwr(ref, ct))
#         if ca < cb:
#             better += 1
#             good_cases.append((ref, hyp, ct, cb*100, ca*100))
#         elif ca > cb:
#             worse += 1
#             bad_cases.append((ref, hyp, ct, cb*100, ca*100))
#         else:
#             same += 1

#     n = len(cer_b)
#     if n == 0:
#         print(f"\n{mode_label}: 无有效样本")
#         return
#     print(f"\n{'='*60}")
#     print(f"{mode_label} ({n}条)")
#     print(f"{'='*60}")
#     print(f"CER (纠正前):     {sum(cer_b)/n*100:.2f}%")
#     print(f"CER (纠正后):     {sum(cer_a)/n*100:.2f}%")
#     print(f"CER 降低:         {(sum(cer_b)-sum(cer_a))/n*100:.2f}%")
#     print(f"热词召回 (纠正前): {sum(hb_)/n*100:.2f}%")
#     print(f"热词召回 (纠正后): {sum(ha_)/n*100:.2f}%")
#     print(f"平均延迟:         {total_t/n*1000:.2f}ms/条")
#     print(f"更好/更差/相同:    {better}/{worse}/{same}")

#     # 打印更差的样本
#     if bad_cases:
#         print(f"\n--- 纠错后 CER 上升的样本 ({len(bad_cases)}条) ---")
#         bad_cases.sort(key=lambda x: x[4]-x[3], reverse=True)
#         for ref, hyp, ct, cb, ca in bad_cases[:30]:
#             print(f"  {ref}")
#             print(f"  → ASR: {hyp}  (CER={cb:.1f}%)")
#             print(f"  → SACM:{ct}  (CER={ca:.1f}% | +{ca-cb:.1f}%)")
#             print()

#     # 打印更好的样本
#     if good_cases:
#         print(f"\n--- 纠错后 CER 下降的样本 ({len(good_cases)}条) ---")
#         good_cases.sort(key=lambda x: x[3]-x[4], reverse=True)
#         for ref, hyp, ct, cb, ca in good_cases[:30]:
#             print(f"  {ref}")
#             print(f"  → ASR: {hyp}  (CER={cb:.1f}%)")
#             print(f"  → SACM:{ct}  (CER={ca:.1f}% | -{cb-ca:.1f}%)")
#             print()


# # === 全句纠错 ===
# def full_correct(hyp):
#     if args.smart_dedup:
#         hyp = smart_dedup_chunked(hyp, chunk_size=args.dedup_chunk_size)
#     elif args.dedup:
#         hyp = dedup_consecutive(hyp)
#     src_ids = torch.tensor([tokenizer.encode(hyp)[:30]], dtype=torch.long).to(args.device)
#     t0 = time.perf_counter()
#     with torch.no_grad():
#         out_ids = model.generate(src_ids, tokenizer, max_len=30, device=args.device, trie=trie)
#     lat = time.perf_counter() - t0
#     corrected = tokenizer.decode(out_ids[0].cpu().numpy())
#     if args.smart_dedup:
#         corrected = dedup_consecutive(corrected)  # SACM 输出最终去重
#     elif args.dedup:
#         corrected = dedup_consecutive(corrected)
#     # 保守门控: SACM改动超过50%则回退ASR原文
#     if args.conservative and corrected != hyp:
#         edits = cer(hyp, corrected) * max(len(hyp), 1)
#         if edits > len(hyp) * 0.5:
#             corrected = hyp
#     return corrected, lat * 1000


# # === 伪流式分块纠错 ===
# def stream_correct(hyp):
#     return streaming_correct(model, tokenizer, hyp, chunk_size=args.chunk_size, device=args.device, trie=trie, conservative=args.conservative, dedup=args.dedup, smart_dedup=args.smart_dedup)


# if args.streaming:
#     run_test(f"全句纠错模式", full_correct)
#     run_test(f"伪流式分块纠错 ({args.chunk_size}字/块)", stream_correct)
# else:
#     run_test("全句纠错模式", full_correct)








#!/usr/bin/env python3
"""SACM纠错测试 - 支持全句纠错 / 伪流式分块纠错"""
import sys, os, json, time, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from light_rccm import LightSACM, CharTokenizer

HOTWORDS = {'三号','七号','四号','站立','旋转','前进','暂停','匍匐','九号','停止','向右','倒靶','后退','向左','卧倒','右转','开始','立靶','六号','射击','放靶','二号','冲击','十号','八号','五号','一号','跃进','左转'}

# 标点 + 口语词 (洗掉非指令内容)
PUNCT = set('，,。.！!？?、；;：:""''（）()【】[] \t\n\r')
STOP_WORDS = {'然后', '接着', '之后', '完了', '这个', '那个', '啊', '呃', '嗯', '嘛', '吧', '呀'}

def clean_text(text):
    """去标点、去口语词、去空格"""
    for sw in STOP_WORDS:
        text = text.replace(sw, '')
    text = ''.join(ch for ch in text if ch not in PUNCT)
    return text

def cer(r, h):
    r, h = clean_text(r), clean_text(h)
    if not r: return 0.0 if not h else 1.0
    m, n = len(r), len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
    return dp[m][n]/len(r)


def extract_hotword_seq(text):
    seq = []
    i = 0
    while i < len(text):
        matched = False
        for hw in sorted(HOTWORDS, key=len, reverse=True):
            if text[i:i+len(hw)] == hw:
                seq.append(hw); i += len(hw); matched = True; break
        if not matched:
            i += 1
    return seq


def seq_edit_distance(ref_seq, hyp_seq):
    m, n = len(ref_seq), len(hyp_seq)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            c = 0 if ref_seq[i-1]==hyp_seq[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+c)
    return dp[m][n]


def hwr(r, h):
    r, h = clean_text(r), clean_text(h)
    ref_seq = extract_hotword_seq(r)
    hyp_seq = extract_hotword_seq(h)
    if not ref_seq: return 1.0
    dist = seq_edit_distance(ref_seq, hyp_seq)
    return max(0, 1 - dist / len(ref_seq))


# ========== 连续重复字去重 ==========
def dedup_consecutive(text):
    """去掉连续重复字: "一号前前前进" → "一号前进" """
    if not text:
        return text
    result = [text[0]]
    for ch in text[1:]:
        if ch != result[-1]:
            result.append(ch)
    return ''.join(result)


def smart_dedup_chunked(text, chunk_size=4):
    """
    逐块去重 + 跨块边界去重:
    1. 切分为 chunk_size 字一块
    2. 每块内部去重
    3. 和记忆库最新块拼接 → 整体去重 → 放回记忆库
    4. 最终整体去重
    """
    if len(text) <= chunk_size:
        return dedup_consecutive(text)
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    memory = dedup_consecutive(chunks[0])  # 第一个块去重后放入记忆
    
    for ch in chunks[1:]:
        ch_dedup = dedup_consecutive(ch)                # 块内去重
        merged = memory + ch_dedup                       # 拼接最新记忆
        memory = dedup_consecutive(merged)               # 跨块边界去重
    
    return memory


# ========== 伪流式 + 边缘补偿纠错 ==========
def streaming_correct(model, tokenizer, raw_text, chunk_size=4, device='cuda',
                      trie=None, conservative=False, dedup=False, smart_dedup=False,
                      edge_width=1):
    """
    伪流式边缘补偿纠错 (邓飞腾论文 4.1.2):
    1. 切 chunk_size 字一块
    2. 第一块: 解码但不输出, 等下一块联合确认
    3. 后续块: prev_raw + cur_raw 联合解码 → 确认前块 → 当前块右边缘截断
    4. 句尾 flush: 输出最后一块剩余
    """

    def decode_one(text):
        pids = tokenizer.encode(text)[:model.max_seq_len]
        src = torch.tensor([pids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(src, tokenizer, max_len=model.max_seq_len,
                                     device=device, trie=trie)
        return tokenizer.decode(out_ids[0].cpu().numpy()), (time.perf_counter() - t0) * 1000

    if smart_dedup:
        raw_text = smart_dedup_chunked(raw_text, chunk_size=args.dedup_chunk_size)
    elif dedup:
        raw_text = dedup_consecutive(raw_text)

    total_latency = 0.0
    confirmed_parts = []
    prev_raw = ""
    prev_corr = ""

    # 切块
    chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]

    for i, cur_raw in enumerate(chunks):
        if i == 0:
            # 第一块：解码但不输出，等下一块联合确认
            cur_corr, lat = decode_one(cur_raw)
            total_latency += lat
            prev_raw = cur_raw
            prev_corr = cur_corr
        else:
            # 联合解码: prev_raw + cur_raw
            joint = prev_raw + cur_raw
            joint_corr, lat = decode_one(joint)
            total_latency += lat

            # 分离前一块修正结果
            confirmed = joint_corr[:len(prev_raw)]
            if confirmed:
                confirmed_parts.append(confirmed)

            # 当前块修正结果: 截断右边缘 edge_width 字（留给下一块确认）
            cur_corr = joint_corr[len(prev_raw):]
            if len(cur_corr) > edge_width:
                cur_corr = cur_corr[:-edge_width]

            prev_raw = cur_raw
            prev_corr = cur_corr

    # flush: 最后一块直接输出
    if prev_corr:
        confirmed_parts.append(prev_corr)

    final_result = "".join(confirmed_parts)

    # 保守门控
    if conservative and final_result != raw_text:
        edits_val = cer(raw_text, final_result) * max(len(raw_text), 1)
        if edits_val > len(raw_text) * 0.5:
            final_result = raw_text
    if smart_dedup or dedup:
        final_result = dedup_consecutive(final_result)
    return final_result, total_latency


# ========== 参数解析 ==========
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rccm-model', required=True)
parser.add_argument('--asr-results', required=True)
parser.add_argument('--pairs', default=None)
parser.add_argument('--use-trie', action='store_true', help='启用Trie约束解码（限制输出为合法指令组合）')
parser.add_argument('--conservative', action='store_true', help='保守纠错: SACM改动>50%时回退ASR原文')
parser.add_argument('--dedup', action='store_true', help='全局去连续重复字 (如"前前前进"→"前进")')
parser.add_argument('--smart-dedup', action='store_true', help='逐块去重+跨块边界去重(先切4字块,块内去重,拼接记忆再跨块去重)')
parser.add_argument('--dedup-chunk-size', type=int, default=4, help='smart-dedup块大小')
parser.add_argument('--streaming', action='store_true', help='启用伪流式分块纠错')
parser.add_argument('--chunk-size', type=int, default=4, help='流式窗口字数 (默认4)')
parser.add_argument('--device', default='cuda')
parser.add_argument('--nat', action='store_true', help='使用 NAT (非自回归) 模型')
parser.add_argument('--det-weight', type=float, default=0.3, help='错误检测 loss 权重')
args = parser.parse_args()

# ========== 构建 Trie（从 30 条热词） ==========
trie = None
if args.use_trie:
    from light_rccm import CommandTrie
    trie = CommandTrie(sorted(HOTWORDS))
    print(f"  Trie约束解码: {len(HOTWORDS)} 条指令前缀树")

print(f"加载ASR结果: {args.asr_results}")
with open(args.asr_results) as f:
    data = json.load(f)
samples = data['details']
print(f"  样本: {len(samples)}")

print(f"\n加载模型: {args.rccm_model}")
ckpt = torch.load(args.rccm_model, map_location=args.device)
cfg = ckpt.get('config', {})
sd = ckpt['model_state_dict']
tokenizer = ckpt.get('tokenizer')
if tokenizer is None:
    if not args.pairs or not os.path.exists(args.pairs):
        print("ERROR checkpoint 无 tokenizer, 请用 --pairs 指定训练对文件"); sys.exit(1)
    print(f"  从训练数据重建词表: {args.pairs}")
    texts = set()
    with open(args.pairs, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            for k in ('tgt', 'target', 'clean'):
                t = item.get(k, '')
                if t: texts.add(t); break
    tokenizer = CharTokenizer(max_vocab=3000)
    tokenizer.load_vocab(list(texts), max_vocab=2996)
vocab_size = len(tokenizer.token2id)

_d_model = cfg.get('d_model', sd['embedding.weight'].shape[1])
_nhead = cfg.get('nhead', 4)
_nlayers = cfg.get('num_encoder_layers', len(set(int(k.split('.')[2]) for k in sd if k.startswith('encoder.layers.'))) or 4)
_ff = cfg.get('dim_feedforward', sd.get('encoder.layers.0.linear1.weight', sd.get('decoder.decoder.layers.0.linear1.weight')).shape[0])
print(f"  推断: d={_d_model}, h={_nhead}, L={_nlayers}, ff={_ff}")
_use_det = cfg.get('use_error_detector', any(k.startswith('error_detector.') for k in sd))

# 自动检测模型类型
_is_nat = args.nat or 'nat' in str(args.rccm_model).lower() or cfg.get('model_type') == 'nat'

if _is_nat:
    from nat_rccm import NATCorrectionModel, nat_generate
    print(f"  类型: NAT (非自回归)")
    model = NATCorrectionModel(
        vocab_size=vocab_size, d_model=_d_model, nhead=_nhead,
        num_encoder_layers=_nlayers, num_decoder_layers=_nlayers,
        dim_ff=_ff, max_seq_len=cfg.get('max_seq_len', 30), dropout=0.0,
        use_error_detector=_use_det,
    )
    # NAT 模型用的 key 名不同: 'decoder' 是 nn.Module, 不是 layers 列表
else:
    _use_det = cfg.get('use_error_detector', any(k.startswith('error_detector.') for k in sd))
    model = LightSACM(
        vocab_size=vocab_size, d_model=_d_model, nhead=_nhead,
        num_encoder_layers=_nlayers, num_decoder_layers=_nlayers,
        dim_feedforward=_ff, max_seq_len=cfg.get('max_seq_len', 30), dropout=0.0,
        use_error_detector=_use_det,
    )
model.load_state_dict(ckpt['model_state_dict'])
model.to(args.device)
model.eval()
print(f"  参数: {sum(p.numel() for p in model.parameters()):,}")


def run_test(mode_label, correct_fn):
    """通用测试循环: correct_fn(hyp) -> corrected_text, latency_ms"""
    cer_b, cer_a = [], []
    hb_, ha_ = [], []
    total_t = 0.0
    better = worse = same = 0
    bad_cases = []   # 纠错后更差
    good_cases = []  # 纠错后更好
    for s in samples:
        hyp = s['hyp']; ref = s['ref']
        ct, lat = correct_fn(hyp)
        total_t += lat / 1000  # ms → s
        cb = cer(ref, hyp); ca = cer(ref, ct)
        cer_b.append(cb); cer_a.append(ca)
        hb_.append(hwr(ref, hyp)); ha_.append(hwr(ref, ct))
        if ca < cb:
            better += 1
            good_cases.append((ref, hyp, ct, cb*100, ca*100))
        elif ca > cb:
            worse += 1
            bad_cases.append((ref, hyp, ct, cb*100, ca*100))
        else:
            same += 1

    n = len(cer_b)
    if n == 0:
        print(f"\n{mode_label}: 无有效样本")
        return
    print(f"\n{'='*60}")
    print(f"{mode_label} ({n}条)")
    print(f"{'='*60}")
    print(f"CER (纠正前):     {sum(cer_b)/n*100:.2f}%")
    print(f"CER (纠正后):     {sum(cer_a)/n*100:.2f}%")
    print(f"CER 降低:         {(sum(cer_b)-sum(cer_a))/n*100:.2f}%")
    print(f"热词召回 (纠正前): {sum(hb_)/n*100:.2f}%")
    print(f"热词召回 (纠正后): {sum(ha_)/n*100:.2f}%")
    print(f"平均延迟:         {total_t/n*1000:.2f}ms/条")
    print(f"更好/更差/相同:    {better}/{worse}/{same}")

    # 打印更差的样本
    if bad_cases:
        print(f"\n--- 纠错后 CER 上升的样本 ({len(bad_cases)}条) ---")
        bad_cases.sort(key=lambda x: x[4]-x[3], reverse=True)
        for ref, hyp, ct, cb, ca in bad_cases[:30]:
            print(f"  {ref}")
            print(f"  → ASR: {hyp}  (CER={cb:.1f}%)")
            print(f"  → SACM:{ct}  (CER={ca:.1f}% | +{ca-cb:.1f}%)")
            print()

    # 打印更好的样本
    if good_cases:
        print(f"\n--- 纠错后 CER 下降的样本 ({len(good_cases)}条) ---")
        good_cases.sort(key=lambda x: x[3]-x[4], reverse=True)
        for ref, hyp, ct, cb, ca in good_cases[:30]:
            print(f"  {ref}")
            print(f"  → ASR: {hyp}  (CER={cb:.1f}%)")
            print(f"  → SACM:{ct}  (CER={ca:.1f}% | -{cb-ca:.1f}%)")
            print()


# === 全句纠错 ===
def full_correct(hyp):
    hyp = clean_text(hyp)  # 先去标点和口语词
    if args.smart_dedup:
        hyp = smart_dedup_chunked(hyp, chunk_size=args.dedup_chunk_size)
    elif args.dedup:
        hyp = dedup_consecutive(hyp)

    if _is_nat:
        # NAT: 单次并行推理
        corrected, lat = nat_generate(model, tokenizer, hyp, device=args.device, max_len=30)
    else:
        # SACM: 自回归推理
        src_ids = torch.tensor([tokenizer.encode(hyp)[:30]], dtype=torch.long).to(args.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(src_ids, tokenizer, max_len=30, device=args.device, trie=trie)
        lat = (time.perf_counter() - t0) * 1000
        corrected = tokenizer.decode(out_ids[0].cpu().numpy())

    if args.smart_dedup:
        corrected = dedup_consecutive(corrected)  # SACM 输出最终去重
    elif args.dedup:
        corrected = dedup_consecutive(corrected)
    # 保守门控: SACM改动超过50%则回退ASR原文
    if args.conservative and corrected != hyp:
        edits = cer(hyp, corrected) * max(len(hyp), 1)
        if edits > len(hyp) * 0.5:
            corrected = hyp
    return corrected, lat


# === 伪流式分块纠错 ===
def stream_correct(hyp):
    return streaming_correct(model, tokenizer, hyp, chunk_size=args.chunk_size, device=args.device, trie=trie, conservative=args.conservative, dedup=args.dedup, smart_dedup=args.smart_dedup)


if args.streaming:
    if _is_nat:
        print("  ⚠ NAT 模型不支持流式纠错，改用全句模式")
    run_test("全句纠错模式", full_correct)
    if not _is_nat:
        run_test(f"伪流式分块纠错 ({args.chunk_size}字/块)", stream_correct)
else:
    run_test("全句纠错模式", full_correct)
