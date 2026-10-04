# account.py —— Book4 ch1 三口径账本：总参 / 激活参 / 每 token FLOPs 公式 + 四样本参数账复算
# 用途：① 立式验证——d=512 mini 档（探针 B 实测 65,042,944 参）逐位对账；② 四样本参数账自算
#       （GShard 600B / Switch-C 1571B / Mixtral 46.7B / gpt-oss-20b 20.91B）与官方口径对拍；
#       ③ 单份/双份激活口径切换打印（本册口径=含 lm_head、不含输入 embedding lookup，双份并列）。
# 所属章节：Book4 第 1 章（1.3 三口径立式 / 1.4 表 1.2 自算对拍 / 1.5 显存与开销账）。
# 运行方式：cd 工作区根目录 && source env.sh && python code/ch01/account.py [--out-name run1]
# 口径：纯整数算术（Python int 无浮点误差）；无随机源（固定种子纪律不适用）。
#       公式承 code/00-feasibility/moe_mla_slots.py 的 hand_account /
#       activated_account 与 notes/00 §六。
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "..", "..", "log", "book4-ch01")


# ---------------- 三口径公式（decoder-only、untied、RMSNorm 只 weight、无 bias） ----------------
def attn_params(d, h, d_k, h_kv, bias=0):
    """每层注意力参数（GQA 矩形记法承 Book3 ch6）：q/o 全宽 h·d_k，k/v 按 h_kv 缩。"""
    qo = 2 * d * h * d_k
    kv = 2 * d * h_kv * d_k
    return qo + kv + bias


def total_params(L, A, V, d, ffn_slot, moe=None, first_dense=0):
    """总参 N = L·(A + FFN 槽) + 2Vd（untied 两份）+ 归一化。
    ffn_slot: dense 槽 = n_mat·d·w；moe 槽 = (E+K_s)·n_mat·d·w + E·d（路由门）。
    moe=(E, k, w, K_s, n_mat) 或 None（纯 dense）；first_dense=首 k 层保持 dense 的层数。"""
    dense_slot = ffn_slot
    moe_slot = ((moe[0] + moe[3]) * moe[4] * d * moe[2] + moe[0] * d) if moe else None
    norms = L * 2 * d + d
    body = sum(dense_slot if (moe is None or i < first_dense) else moe_slot for i in range(L))
    return L * A + body + 2 * V * d + norms


def active_params(L, A, V, d, ffn_slot, moe=None, first_dense=0, double_embed=False,
                  with_router=True):
    """激活参 N_act（每 token 点亮的参数）：注意力全亮 + MoE 层 k 个路由专家 + 全部共享专家
    （+路由门，with_router 可关以便与探针 B 口径对齐）+ lm_head 单份；double_embed=True 时
    输入 embedding 也计入（双份口径）。"""
    dense_slot = ffn_slot
    if moe:
        E, k, w, K_s, n_mat = moe
        moe_slot = k * n_mat * d * w + K_s * n_mat * d * w + (E * d if with_router else 0)
    else:
        moe_slot = None
    body = sum(dense_slot if (moe is None or i < first_dense) else moe_slot for i in range(L))
    norms = L * 2 * d + d
    emb = 2 * V * d if double_embed else V * d
    return L * A + body + norms + emb


# ---------------- 验证一：d=512 mini 档（探针 B 实机，等激活配臂） ----------------
def check_mini():
    d, L, h, h_kv, d_k, V = 512, 6, 8, 4, 64, 8192
    A = attn_params(d, h, d_k, h_kv)
    dense = total_params(L, A, V, d, ffn_slot=3 * d * 1408)
    moe_total = total_params(L, A, V, d, ffn_slot=3 * d * 1408, moe=(8, 2, 704, 0, 3))
    moe_act_single = active_params(L, A, V, d, ffn_slot=3 * d * 1408, moe=(8, 2, 704, 0, 3))
    dense_act_single = active_params(L, A, V, d, ffn_slot=3 * d * 1408, moe=None)
    moe_act_no_router = active_params(L, A, V, d, ffn_slot=3 * d * 1408, moe=(8, 2, 704, 0, 3),
                                      with_router=False)
    dense_total_eq = total_params(L, A, V, d, ffn_slot=3 * d * (8 * 704))  # 等总参臂 d_ff=E·w
    assert dense == 26_089_984 and moe_total == 65_042_944, "探针 B 实测对账失败"
    return {"dense_total": dense, "moe_total": moe_total,
            "dense_act_single": dense_act_single, "moe_act_single": moe_act_single,
            "moe_act_single_no_router": moe_act_no_router,
            "dense_eqtotal_total": dense_total_eq,
            "moe_minus_eqtotal": moe_total - dense_total_eq}


# ---------------- 验证二~五：四样本参数账（论文级 → config 级） ----------------
def gshard():
    """GShard MoE(2048E,36L)——论文级：Table 1 专家总数 36684 + 附录 A.2 维度 1024/8192。
    专家=ReLU 两矩阵（2·d·d_ff/个）；enc-dec 无公开 config，注意力/嵌入按 1024 维估 ~5B。"""
    per_expert = 2 * 1024 * 8192
    experts = 36_684 * per_expert
    return {"per_expert": per_expert, "experts_total": experts,
            "plus_attn_emb_est": 5_000_000_000, "self_total": experts + 5_000_000_000,
            "official": 600_000_000_000, "official_str": "600B（摘要 beyond 600 billion，取整）"}


def switch_c():
    """Switch-C——论文级：Table 9（d=2080/d_ff=6144/2048 专家/频率 1/15 层/top-1）。
    槽位数由官方 1571B 反解=30（15 编+15 解，每层 FFN 全换；T5 解码器第二处 FFN 未给账）。"""
    per_expert = 2 * 2080 * 6144
    per_slot = 2048 * per_expert
    slots = round(1571e9 / per_slot)
    experts = slots * per_slot
    attn = slots * 4 * 2080 * 2080
    emb = 2 * 32_000 * 2080
    return {"per_expert": per_expert, "slots_solved": slots, "experts": experts,
            "attn": attn, "emb": emb, "self_total": experts + attn + emb,
            "official": 1_571_000_000_000, "official_str": "1571B（Table 9）"}


def mixtral():
    """Mixtral 8x7B——config 级：d=4096/L=32/h=32/h_kv=8/d_k=128/V=32000/E=8/top-2/w=14336。"""
    d, L, h, h_kv, d_k, V = 4096, 32, 32, 8, 128, 32000
    A = attn_params(d, h, d_k, h_kv)
    moe = (8, 2, 14336, 0, 3)  # E=8, k=2, w=14336, K_s=0, SwiGLU 三矩阵
    total = total_params(L, A, V, d, ffn_slot=3 * d * 14336, moe=moe)
    act1 = active_params(L, A, V, d, ffn_slot=3 * d * 14336, moe=moe)
    act2 = active_params(L, A, V, d, ffn_slot=3 * d * 14336, moe=moe, double_embed=True)
    return {"total": total, "active_single": act1, "active_double": act2,
            "official_str": "论文 47B/13B（取整）；HF 卡 46.7B/12.9B"}


def gptoss_20b():
    """gpt-oss-20b——config 级（含偏置与 sink，承 papers/06 §A.3 公式）：
    d=2880/L=24/h=64/h_kv=8/d_k=64/E=32/top-4/w=2880/V=201088，untied，全 24 层 MoE。"""
    d, L, h, h_kv, d_k, V = 2880, 24, 64, 8, 64, 201_088
    A = (h * d_k) * d + d * (h * d_k) + (h_kv * d_k) * d * 2 \
        + (h * d_k + d + 2 * h_kv * d_k) + h          # q,o,k,v 权重+偏置+每头 1 sink
    E, k, w, K_s, n_mat = (32, 4, 2880, 0, 3)
    mlp_layer = E * (n_mat * d * w + 2 * w + d) + E * d + E   # 专家（含偏置）+路由门
    total = L * (A + mlp_layer + 2 * d) + 2 * V * d + d
    act1 = L * (A + k * (n_mat * d * w + 2 * w + d) + E * d + E + 2 * d) + V * d + d
    act2 = act1 + V * d
    assert total == 20_914_757_184 and act1 == 3_608_307_264, "gpt-oss 官方 Table 1 对账失败"
    return {"attn_per_layer": A, "mlp_per_layer": mlp_layer, "total": total,
            "active_single": act1, "active_double": act2,
            "official_str": "官方 Table 1 20.91B / Active 3.61B（单份口径）"}


def main():
    ap = argparse.ArgumentParser(description="ch1 三口径账本：四样本参数账复算")
    ap.add_argument("--out-name", type=str, default="run1", help="产物文件名后缀（防覆写）")
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    mini = check_mini()
    res = {"mini_d512": mini, "gshard": gshard(), "switch_c": switch_c(),
           "mixtral": mixtral(), "gptoss_20b": gptoss_20b()}

    B = 1_000_000_000
    print("== 验证一：d=512 mini 档（探针 B 实机对账，逐位） ==")
    print(f"  dense 总参            {mini['dense_total']:,}")
    print(f"  MoE 总参（E8/top2/w704）{mini['moe_total']:,}  （2.49x）")
    print(f"  等总参 dense（d_ff=5632）{mini['dense_eqtotal_total']:,}  （与 MoE 差 "
          f"{mini['moe_minus_eqtotal']:,} = 路由门 E·d）")
    print(f"  激活参（单份）：dense {mini['dense_act_single']:,} / MoE {mini['moe_act_single']:,}"
          f"（含路由门；不含则 {mini['moe_act_single_no_router']:,}，与 dense 逐位相同）")
    print("== 验证二：GShard MoE(2048E,36L)（论文级） ==")
    g = res["gshard"]
    print(f"  专家账 = 36,684 x 2·1024·8192 = {g['experts_total']:,}（≈{g['experts_total']/B:.1f}B）"
          f"  vs 官方 {g['official_str']} —— 差 {(g['self_total']/g['official']-1)*100:+.1f}%，无公开 config，登记口径差")
    print("== 验证三：Switch-C（论文级，槽位反解） ==")
    s = res["switch_c"]
    print(f"  反解槽位 = 1571B / (2048 x 2·2080·6144) = {s['slots_solved']}；"
          f"自算总量 {s['self_total']:,}（≈{s['self_total']/B:.1f}B）vs 官方 {s['official_str']}")
    print("== 验证四：Mixtral 8x7B（config 级） ==")
    m = res["mixtral"]
    print(f"  总参 {m['total']:,}（{m['total']/B:.3f}B）/ 激活单份 {m['active_single']:,}"
          f"（{m['active_single']/B:.3f}B）/ 双份 {m['active_double']:,}（{m['active_double']/B:.3f}B）")
    print(f"  官方 {m['official_str']}")
    print("== 验证五：gpt-oss-20b（config 级，逐位 assert 官方 Table 1） ==")
    o = res["gptoss_20b"]
    print(f"  总参 {o['total']:,} / 激活单份 {o['active_single']:,}（{o['active_single']/B:.3f}B）"
          f" / 双份 {o['active_double']:,}（{o['active_double']/B:.3f}B）")
    print(f"  官方 {o['official_str']} —— 单份精确吻合（跨厂商口径不齐的实证）")
    print(f"  每 token 前向 FLOPs ≈ 2·N_act：Mixtral {2*m['active_single']/B:.1f}G / "
          f"gpt-oss {2*o['active_single']/B:.1f}G")

    out = os.path.join(LOG_DIR, f"account_{args.out_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
