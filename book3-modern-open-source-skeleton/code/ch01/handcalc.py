# handcalc.py —— ch1 算力账三笔手算：6ND 账、GPU-h 账、吞吐账，三方互证
# 用途：表 1.1 全部自算数字（6ND/PF-days/每参数 token 数/嵌入占比）+ 65B 三角互证；产物 log/book3-ch01/{out-name}.json
# 所属章节：Book3 第 1 章 1.4 节（原理章；纯 CPU 算术、秒级、无随机性）
# 运行方式：激活既有环境后，在本书目录python code/ch01/handcalc.py；外层工作区调用须加书目录前缀。
import argparse, json, math
from pathlib import Path

PF_DAY = 8.64e19        # 1 PF-day = 1e15 FLOP/s × 86400 s（Book2 式 (9.4) 换算盒）
A100_TFLOPS = 312.0     # A100-80GB FP16/BF16密集峰值；624是稀疏口径（NVIDIA官方规格）
V = 32_000              # LLaMA1 词表（untied：wte 与 lm_head 各一份 → 嵌入参数 = 2·V·dim）
# 档位：N=全参数（Table 2 口径 6.7/13.0/32.5/65.2B；GPT-3=174.6B 沿 Book2 第 9 章手算）；D=tokens；gpu_h=Table 15；dim=嵌入宽度
MODELS = [
    dict(name="GPT-3", N=174.6e9, D=300e9, gpu_h=None, dim=None),
    dict(name="LLaMA-7B", N=6.7e9, D=1.0e12, gpu_h=82_432, dim=4096),
    dict(name="LLaMA-13B", N=13.0e9, D=1.0e12, gpu_h=135_168, dim=5120),
    dict(name="LLaMA-33B", N=32.5e9, D=1.4e12, gpu_h=530_432, dim=6656),
    dict(name="LLaMA-65B", N=65.2e9, D=1.4e12, gpu_h=1_022_362, dim=8192),
]

def row(m):
    """一行账：6ND（全参数口径）→ PF-days 换算 → 每参数 token 数 → untied 嵌入占比。"""
    if not all(math.isfinite(m[k]) and m[k]>0 for k in ("N","D")):
        raise ValueError("N/D须为有限正数")
    flops = 6.0 * m["N"] * m["D"]                                   # 第一笔：6ND 账
    return dict(name=m["name"], N=m["N"], D=m["D"], flops=flops, pf_days=flops / PF_DAY,
                tok_per_param=m["D"] / m["N"], emb_share=(2.0 * V * m["dim"] / m["N"]) if m["dim"] else None,
                gpu_h=m["gpu_h"])

def triangle_65b():
    """第三笔：65B 三角互证——GPU-h 账（§2.4：2048 张 A100、380 tok/s/GPU、约 21 天）。"""
    m = MODELS[4]
    days = m["gpu_h"] / 2048 / 24                                   # GPU-h 账 → 天（2048 卡口径）
    tokens = 2048 * 380 * days * 86400                              # 吞吐账：反推全程吃掉的 token
    per_gpu = 6.0 * m["N"] * m["D"] / (m["gpu_h"] * 3600)           # 6ND ÷ GPU-秒 = 每卡有效算力
    return dict(gpu_h=m["gpu_h"], days_2048=days, tokens_via_380=tokens,
                flops_per_gpu_tflops=per_gpu / 1e12, from_380=380 * 6.0 * m["N"] / 1e12,
                mfu_vs_312=per_gpu / 1e12 / A100_TFLOPS,
                interpretation="6ND模型主项吞吐相对密集峰值；不是硬件全部执行FLOPs利用率，三输入出自同报告")

def lifetime_compute(n, d, usage):
    """6ND+2NU的模型主项；U计实际前向位置并假定复用历史投影，非只数输出词元。"""
    if not all(math.isfinite(x) for x in (n,d,usage)) or n<=0 or d<=0 or usage<0:
        raise ValueError("N/D须正、U须非负且全部有限")
    return 6*n*d+2*n*usage


def crossing_usage(n_a,d_a,n_b,d_b):
    """仅N_b<N_a时返回阈值u0；B严格更省需U>u0，阈值可非正。"""
    lifetime_compute(n_a,d_a,0);lifetime_compute(n_b,d_b,0)
    if n_b>=n_a:raise ValueError("此分界式要求B更小")
    return 3*(n_b*d_b-n_a*d_a)/(n_a-n_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="handcalc", help="产物文件名（防覆写）")
    ap.add_argument("--out-dir",type=Path,default=Path.cwd()/"log/book3-ch01")
    args = ap.parse_args()
    name=args.out_name[:-5] if args.out_name.endswith('.json') else args.out_name
    if Path(name).name!=name or name in ('','.','..'):raise ValueError('out-name须为单个文件名')
    target=args.out_dir/(name+'.json')
    if target.exists():raise FileExistsError('不覆盖已有结果：'+str(target))
    rows = [row(m) for m in MODELS]
    for r in rows[1:]:                                              # 对 GPT-3 锚（3.14e23，Book2 第 9 章同款）的倍率
        r["ratio_vs_gpt3"] = r["flops"] / rows[0]["flops"]
    tri = triangle_65b()
    print(f'{"model":<10}{"N":>9}{"D":>8}{"6ND/FLOPs":>12}{"PF-days":>10}{"tok/param":>10}{"xGPT-3":>8}{"emb%":>7}{"GPU-h":>10}')
    for r in rows:
        emb = f'{r["emb_share"] * 100:.2f}' if r["emb_share"] else "-"      # untied 嵌入占全参数比例
        gpu = f'{r["gpu_h"]:,}' if r["gpu_h"] else "n/a"
        print(f'{r["name"]:<10}{r["N"] / 1e9:>8.1f}B{r["D"] / 1e12:>7.1f}T{r["flops"]:>12.3e}{r["pf_days"]:>10.1f}'
              f'{r["tok_per_param"]:>10.1f}{r.get("ratio_vs_gpt3", 1.0):>8.3f}{emb:>7}{gpu:>10}')
    print(f'\n[65B 三角互证] {tri["gpu_h"]:,} GPU-h / 2048 / 24 = {tri["days_2048"]:.1f} 天（论文 §2.4 约 21 天）; '
          f'2048x380 tok/s x {tri["days_2048"]:.1f} 天 = {tri["tokens_via_380"]:.3e} tokens（累计训练量约1.4T）'
          f'\n[每卡有效算力] 6ND/GPU-秒 = {tri["flops_per_gpu_tflops"]:.1f} TFLOP/s; 380x6N = {tri["from_380"]:.1f} TFLOP/s '
          f'（两路口径咬合；标称 {A100_TFLOPS:.0f} 的 {tri["mfu_vs_312"] * 100:.0f}%；峰值为厂商标称，比率为本书估算）')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    toy=dict(n_a=10e9,d_a=200e9,n_b=5e9,d_b=600e9)
    threshold=crossing_usage(**toy)
    with target.open("x",encoding="utf-8") as f:
        json.dump(dict(rows=rows,triangle_65b=tri,pf_day=PF_DAY,a100_dense_tflops=A100_TFLOPS,
                       lifetime_toy={**toy,"crossing_usage":threshold,"kind":"教学构造，未验证等质量"},
                       scope="纸面近似计算；未训练或测量设备"),f,ensure_ascii=False,indent=2,allow_nan=False)
    print(f"[saved] {target}")

if __name__ == "__main__":
    main()
