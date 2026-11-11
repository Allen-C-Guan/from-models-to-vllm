# 用途：Book2 ch7 章前手算核验——6ND 算力账、PF-days 换算、指标变换小数字例（算术核验，不是论文结果的独立复现）
# 所属章节：Book2 第 7 章 §7.1、§7.5
# 运行方式：激活配套Python环境后，在本册目录运行 python "code/ch07/handcalc.py"
# 产物：log/book2-ch07/handcalc.txt与handcalc.json，记录正文预算及概率构造

from pathlib import Path
import json

OUT_DIR = Path("log/book2-ch07")

def main():
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
    emit(f"    C = 6 × {N124:.4e} × {D:.1e} = {6 * N124 * D:.4e} FLOPs（≈ 175B 档的 1/1403.1）")
    emit(f"    参数比 174,600e6 / 124,439,808 = {N / N124:.1f} 倍")
    emit(f"    d_model 比 12288 / 768 = {12288 / 768:.0f} 倍；层数比 96 / 12 = {96 / 12:.0f} 倍；头积 96×128 = {96 * 128}（=d_model ✓）")
    emit()

    # ---- 3) 300B/175B ≈ 1.7 token/参数（ch8 Chinchilla 审判的伏笔数字） ----
    emit("[3] GPT-3 实配的数据-参数比（§7.2 明钩）")
    emit(f"    D/N = 300e9 / 174.6e9 = {300e9 / N:.2f} tokens/参数（Chinchilla Table3方法1外推：3.7T/175B≈21.1；不是GPT3实测最佳值，第8章回收）")
    emit()

    # ---- 4) 指标变换小数字例（§7.5 教学算例；图7.3的独立概率构造） ----
    emit("[4] 指标变换：四个假想档位的逐位置正确率p，两种期望分数")
    emit("    四个递增规模档位的逐 token 正确率 p（教学设定值）：")
    ps = [0.30, 0.50, 0.65, 0.80]
    L = 4  # 四位数字答案的目标串长
    for i, p in enumerate(ps, 1):
        em = p ** L
        lin = 4 * p  # 线性指标读数：期望答对 token 数 / 4
        emit(f"      档{i}: p={p:.2f}  →  exact match p^{L} = {em:.4f}   线性读数 4p/4 = {lin/4:.2f}")
    emit(f"    随机底线：4 位数字瞎猜全对 = 10^-4 = {1e-4:.4f}")
    emit("    读法：p增量为0.20、0.15、0.15；p^4仍连续，不是数学阶跃")
    emit("    （固定4位、一位视作一个token、各位置独立同分布；真实生成不保证这些条件）")
    emit()

    # ---- 5) LAMBADA / TriviaQA 三设定（Table 3.2/3.3 抄录锚，正文引用对拍） ----
    emit("[5] 三设定原表摘录（GPT-3 Table3.2/3.3；非本机实验，最终以原论文为准）")
    emit("    LAMBADA acc: zero-shot 76.2 (ppl 3.00) / one-shot 72.5 (3.35) / few-shot 86.4 (1.92)")
    emit("    TriviaQA (closed book): 64.3 / 68.0 / 71.2（对照 T5-11B 50.1、T5-11B+SSM 60.5、RAG 68.0）")
    emit()

    # ---- 6) Wei 表 1 few-shot 阈值跨度 ----
    emit("[6] Wei v2 Table 1（few-shot panel）阈值跨度：1.3e22 ~ 2.5e24")
    emit("    下限 = CivilComments 毒性分类 @ Gopher 7.1B 的 1.3e22（few-shot九项数值记录中的最小）；上限 = Word in Context @ PaLM 540B 的 2.5e24")
    emit(f"    跨度 = {2.5e24 / 1.3e22:.0f} 倍（约两个数量级）——图 7.2 数据源")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "handcalc.txt").open("w") as f:
        f.write("\n".join(lines) + "\n")
    checks = {
        "train_flops": C, "pf_days": pfdays, "parameter_ratio": N / N124,
        "token_parameter_ratio": D / N,
        "p_values": ps, "em_values": [p**L for p in ps],
        "zero_success_20": (1-.3**4)**20,
        "zero_success_500": (1-.3**4)**500,
        "dependent_em_at_p_half": .5,
        "independent_em_at_p_half": .5**4,
        "brier_two_class_sum": [2*(1-p)**2 for p in (.49,.51)],
        "source": "公式计算与论文表格摘录；不运行模型训练或API",
    }
    (OUT_DIR / "handcalc.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n")
    print("\n落盘:", OUT_DIR / "handcalc.txt")


if __name__ == "__main__":
    main()
