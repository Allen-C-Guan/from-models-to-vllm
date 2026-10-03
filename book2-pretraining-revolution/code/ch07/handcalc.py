# 用途：Book2 ch7 章前手算核验——6ND 算力账、PF-days 换算、指标整流器小数字例（正文引用数字的唯一来源）
# 所属章节：Book2 第 7 章 §7.1、§7.5
# 运行方式：cd 工作区根目录 && source env.sh && python "code/ch07/handcalc.py"
# 产物：log/book2-ch07/handcalc.txt（正文每个手算数字与此文件逐项对应）

import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "log", "book2-ch07")
os.makedirs(OUT_DIR, exist_ok=True)

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


emit("Book2 ch7 手算核验（对照锚：GPT-3 2005.14165 v4 附录 Table D.1；Wei 2206.07682 v2 Table 1）")
emit("=" * 72)

# ---- 1) 6ND：GPT-3 175B 训练总算力手算（Table D.1 锚） ----
N = 174_600e6          # GPT-3 175B 行的参数量口径 = 174,600 million（Table D.1 原值）
D = 300e9              # 训练 tokens（Table 2.1 注：全部模型 300B tokens）
fwd = 2                # 每参数每 token 前向浮点运算：1 乘 + 1 加 = 2
bwd = 4                # 反向 ≈ 2× 前向 = 4（D.1 表注：整体乘 3，2×3=6，同一账）
C = 6 * N * D
pfdays = C / 8.64e19   # 1 PF-day = 10^15 FLOP/s × 86400 s = 8.64e19 FLOPs
emit("[1] 6ND 算力账（§7.1 式 7.1）")
emit(f"    N = 174,600e6 = {N:.4e}（D.1 口径；标称 175B）")
emit(f"    D = 300e9 tokens")
emit(f"    前向每参数每 token = {fwd} FLOPs；反向 = {bwd} FLOPs；合计 6")
emit(f"    C = 6 × N × D = 6 × {N:.4e} × {D:.1e} = {C:.4e} FLOPs")
emit(f"    ↔ Table D.1 官方值 3.14E+23 → 对上（{abs(C - 3.14e23) / 3.14e23:.2%} 以内）")
emit(f"    PF-days 换算：{C:.4e} / 8.64e19 = {pfdays:.1f} ≈ 3.64e3")
emit(f"    ↔ Table D.1 官方值 3.64E+03 petaflop/s-day → 对上")
emit()

# ---- 2) 每参数/每 token 口径的量级感：175B vs 124M 同吃 300B 的假想账 ----
N124 = 124_439_808
emit("[2] 量级感：同为 6ND 口径，GPT-2 124M 若也吃 300B tokens")
emit(f"    C = 6 × {N124:.4e} × {D:.1e} = {6 * N124 * D:.4e} FLOPs（≈ 175B 档的 1/1404）")
emit(f"    参数比 174,600e6 / 124,439,808 = {N / N124:.1f} 倍")
emit(f"    d_model 比 12288 / 768 = {12288 / 768:.0f} 倍；层数比 96 / 12 = {96 / 12:.0f} 倍；头积 96×128 = {96 * 128}（=d_model ✓）")
emit()

# ---- 3) 300B/175B ≈ 1.7 token/参数（ch8 Chinchilla 审判的伏笔数字） ----
emit("[3] GPT-3 实配的数据-参数比（§7.2 明钩）")
emit(f"    D/N = 300e9 / 174.6e9 = {300e9 / N:.2f} tokens/参数（Chinchilla 表 3 同档最优 ≈ 21——第 8 章回收）")
emit()

# ---- 4) 指标整流器小数字例（§7.5 教学算例；图 7.3 同源数据） ----
emit("[4] 指标整流器：同一批模型、同一份逐 token 正确率 p，换指标换剧本")
emit("    四个递增规模档位的逐 token 正确率 p（教学设定值）：")
ps = [0.30, 0.50, 0.65, 0.80]
L = 4  # 四位数字答案的目标串长
for i, p in enumerate(ps, 1):
    em = p ** L
    lin = 4 * p  # 线性指标读数：期望答对 token 数 / 4
    emit(f"      档{i}: p={p:.2f}  →  exact match p^{L} = {em:.4f}   线性读数 4p/4 = {lin/4:.2f}")
emit(f"    随机底线：4 位数字瞎猜全对 = 10^-4 = {1e-4:.4f}")
emit("    读法：p 平滑 +0.15/档，EM 却 0.008→0.063→0.179→0.410，像「突然会了」")
emit("    （独立假设仅为教学近似；Schaeffer v2 脚注 1 同样声明该假设不真但定性匹配）")
emit()

# ---- 5) LAMBADA / TriviaQA 三设定（Table 3.2/3.3 抄录锚，正文引用对拍） ----
emit("[5] 三设定数字抄录锚（GPT-3 Table 3.2/3.3，正文引用以此为准）")
emit("    LAMBADA acc: zero-shot 76.2 (ppl 3.00) / one-shot 72.5 (3.35) / few-shot 86.4 (1.92)")
emit("    TriviaQA (closed book): 64.3 / 68.0 / 71.2（对照 T5-11B 50.1、T5-11B+SSM 60.5、RAG 68.0）")
emit()

# ---- 6) Wei 表 1 few-shot 阈值跨度 ----
emit("[6] Wei v2 Table 1（few-shot panel）阈值跨度：1.3e22 ~ 2.5e24")
emit("    下限 = CivilComments 毒性分类 @ Gopher 7.1B 的 1.3e22（全表最小）；上限 = Word in Context @ PaLM 540B 的 2.5e24")
emit(f"    跨度 = {2.5e24 / 1.3e22:.0f} 倍（约两个数量级）——图 7.2 数据源")

with open(os.path.join(OUT_DIR, "handcalc.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n落盘:", os.path.join(OUT_DIR, "handcalc.txt"))
