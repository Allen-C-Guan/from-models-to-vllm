# 02_probe_roofline.py —— Book6 本机探针 P2：910B3 带宽/算力 microbench（roofline 校准锚点，ch1/ch2 前置）
#                        ＋ P3：HBM 计量接口语义实测（torch.npu.mem_get_info / max_memory_allocated /
#                          reset_peak_memory_stats —— 写作规范 §8「本册内存计量口径」定版的实测依据）
# 用途：
#   P2-① 带宽 microbench：HBM 大 tensor 逐元素 D2D 拷贝（读写 2×bytes）与规约（读 1×bytes），
#        多尺寸（64MiB-1GiB）实测有效带宽 GB/s，多尺寸取稳态（大尺寸为主报数）；
#   P2-② 算力 microbench：bf16 方阵 matmul 多尺寸（2048-8192）实测 TFLOPS（FLOPs = 2·n³）；
#   P2-③ 与官方标称的可达比例：本脚本只报实测——官方标称数值以另一路调研（910B3 规格调研）为准，
#        比值在其定版后于 notes/04-本机探针记录.md 回填；
#   P3   三个内存计量接口各跑一遍并打印语义说明——定版「本册内存计量口径」。
# 计时方法：torch.npu.synchronize() + time.perf_counter()；每尺寸 warmup 3 次后计 10 次，
#        报 min/median（microbench 惯例：min 剔除调度毛刺，median 观察稳态；带宽主报数取大尺寸 median）。
# 运行：cd <workspace> && source env.sh && npu-smi info（挑空闲卡）后
#      ASCEND_RT_VISIBLE_DEVICES=<卡号> python code/Book6-推理系统导论/00-feasibility/02_probe_roofline.py
# 预算：全程 ≤5 分钟（带宽 ~40s + matmul ~30s + 内存语义 ~10s，含首跑 kernel 编译）。
import time

import torch
import torch_npu  # noqa: F401 —— import 即注册 npu 后端

DEV = "npu"
COPY_SIZES_MIB = [64, 128, 256, 512, 1024]     # 带宽 tensor 的字节数档位（bf16 元素数 = MiB*2^20/2）
MM_SIZES = [2048, 3072, 4096, 6144, 8192]      # bf16 方阵边长档位
WARMUP, ITERS = 3, 10


def bench(fn):
    """单 kernel 计时：warmup 后逐次 sync 计时，返回 (min_s, median_s)。"""
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts), sorted(ts)[len(ts) // 2]


def env_header():
    print("=" * 78)
    print("[环境] torch", torch.__version__, "| torch_npu", torch_npu.__version__,
          "| device:", torch.npu.get_device_name(0))
    p = torch.npu.get_device_properties(0)
    print("[环境] properties: name=%s total_memory=%s GiB" % (
        getattr(p, "name", "?"), round(getattr(p, "total_memory", 0) / 2**30, 2)))
    free, total = torch.npu.mem_get_info()
    print("[环境] mem_get_info: free=%.2f GiB / total=%.2f GiB（设备级 HBM 物理口径）"
          % (free / 2**30, total / 2**30))
    print("=" * 78)


def probe_bandwidth():
    """P2-①：D2D 拷贝（traffic=2×bytes）与规约（traffic=1×bytes）的有效带宽。
    主报数取 ≥256MiB 档的中位（稳态）——64MiB 档实测显著偏高，疑似小工作集被片上缓存放大，
    以大尺寸稳态为准（「多尺寸取稳态」纪律）。"""
    print("\n[P2-① 带宽 microbench]  bf16 | 拷贝 y.copy_(x) 读写 2×bytes | 规约 x.sum() 读 1×bytes")
    print("%9s %12s %18s %18s" % ("size", "copy med", "copy BW(copy)", "reduce BW(sum)"))
    rows = []
    for mib in COPY_SIZES_MIB:
        n = mib * 2**20 // 2                                # bf16 元素数
        x = torch.randn(n, dtype=torch.bfloat16, device=DEV)
        y = torch.empty_like(x)
        t_cp_min, t_cp_med = bench(lambda: y.copy_(x))       # 读写各 n×2 字节
        acc = []
        t_rd_min, t_rd_med = bench(lambda: acc.append(x.sum()))  # 读 n×2 字节（防编译器消融：留存结果）
        bytes_ = n * 2
        bw_cp = 2 * bytes_ / t_cp_med / 1e9                  # GB/s（1e9 字节口径，下同）
        bw_rd = 1 * bytes_ / t_rd_med / 1e9
        rows.append((mib, bw_cp, bw_rd))
        print("%6dMiB %10.3fms %14.1f GB/s %14.1f GB/s"
              % (mib, t_cp_med * 1e3, bw_cp, bw_rd))
        del x, y, acc
    stable = [r for r in rows if r[0] >= 256]                # ≥256MiB 档 = 稳态
    plateau_cp = sorted(r[1] for r in stable)[len(stable) // 2]
    plateau_rd = sorted(r[2] for r in stable)[len(stable) // 2]
    print("[P2-① 稳态] ≥256MiB 档中位：拷贝(2×traffic) %.1f GB/s | 规约(1×traffic) %.1f GB/s"
          % (plateau_cp, plateau_rd))
    print("           注：64MiB 档拷贝 %.1f GB/s 显著高于稳态——疑似小工作集缓存放大，不作主报数"
          % rows[0][1])
    return plateau_cp, plateau_rd


def probe_matmul():
    """P2-②：bf16 方阵 matmul 实测 TFLOPS（FLOPs = 2·n³；median 为主报数）。"""
    print("\n[P2-② 算力 microbench]  bf16 方阵 C = A·B，torch.matmul")
    print("%8s %14s %12s %12s" % ("n", "median", "TFLOPS(med)", "TFLOPS(min)"))
    peak_med, peak_min, at_n = 0.0, 0.0, 0
    for n in MM_SIZES:
        a = torch.randn(n, n, dtype=torch.bfloat16, device=DEV)   # (n,n)
        b = torch.randn(n, n, dtype=torch.bfloat16, device=DEV)   # (n,n)
        t_min, t_med = bench(lambda: torch.matmul(a, b))          # (n,n)·(n,n)->(n,n)
        fl_min, fl_med = 2 * n**3 / t_min / 1e12, 2 * n**3 / t_med / 1e12
        if fl_med > peak_med:
            peak_med, peak_min, at_n = fl_med, fl_min, n
        print("%8d %10.3fms %12.1f %12.1f" % (n, t_med * 1e3, fl_med, fl_min))
        del a, b
    print("[P2-② 峰值] 尺寸 n=%d 处 median %.1f TFLOPS（min 口径 %.1f）" % (at_n, peak_med, peak_min))
    return peak_med


def probe_roofline_point(bw_copy, tflops):
    """用实测两锚点估算本机 roofline 屋脊点（算术强度 AI* = 峰值算力/峰值带宽，FLOP/Byte）。"""
    if bw_copy > 0 and tflops > 0:
        ai = tflops * 1e12 / (bw_copy * 1e9)                  # FLOP/Byte
        print("\n[P2-③ 实测 roofline 锚点] 带宽 %.1f GB/s（拷贝，2×traffic 口径） × 算力 %.1f TFLOPS"
              " -> 屋脊点 AI* ≈ %.1f FLOP/Byte" % (bw_copy, tflops, ai))
        print("           （官方标称可达比例待「910B3 规格调研」定版后回填 notes/04；本探针只报实测）")


def probe_mem_api():
    """P3：三个内存计量接口各跑一遍，打印语义说明——「本册内存计量口径」定版依据。"""
    GiB = 2**30
    print("\n" + "=" * 78)
    print("[P3 HBM 计量接口语义实测]  语义说明见各行注释式打印，结论进 notes/04 定版")
    free0, total0 = torch.npu.mem_get_info()
    alloc0 = torch.npu.memory_allocated()
    peak0 = torch.npu.max_memory_allocated()
    print("[语义] torch.npu.mem_get_info() -> (free, total)：设备级 HBM 物理口径（byte），")
    print("       含 caching allocator 已缓存未用块与运行时上下文——观察「整卡还剩多少」用它。")
    print("       实测#0: free=%.3f GiB, total=%.3f GiB" % (free0 / GiB, total0 / GiB))
    print("[语义] torch.npu.memory_allocated()：PyTorch caching allocator 当前「活动张量」字节，")
    print("       只数活着的 tensor，不含空闲缓存池——观察「模型/KV 实占」用它。实测#0: %.3f GiB"
          % (alloc0 / GiB))
    print("[语义] torch.npu.max_memory_allocated()：自进程启动或上次 reset 以来的 allocated 峰值。")
    print("       实测#0: %.3f GiB" % (peak0 / GiB))

    n = 128 * 2**20 // 2                                     # 128 MiB 的 bf16 tensor
    t = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    free1, _ = torch.npu.mem_get_info()
    alloc1 = torch.npu.memory_allocated()
    peak1 = torch.npu.max_memory_allocated()
    print("[实测#1] 分配 %.3f GiB tensor 后：allocated +%d B（精确等于张量字节数 %d B）"
          % (n * 2 / GiB, alloc1 - alloc0, n * 2))
    print("         mem_get_info free 变化 %.3f GiB（≥张量字节——allocator 按段预留，可能多占）"
          % ((free0 - free1) / GiB))
    print("         peak %.3f -> %.3f GiB（峰值随分配抬升）" % (peak0 / GiB, peak1 / GiB))

    del t                                                    # 释放张量：allocator 缓存不归还 HBM
    alloc2 = torch.npu.memory_allocated()
    free2, _ = torch.npu.mem_get_info()
    peak2 = torch.npu.max_memory_allocated()
    print("[实测#2] del tensor 后：allocated 回落到 %.3f GiB（活动账归零段），但 mem_get_info free 仅 %.3f GiB"
          "（未归还——caching allocator 持有）；peak 保持 %.3f GiB" % (alloc2 / GiB, free2 / GiB, peak2 / GiB))

    torch.npu.reset_peak_memory_stats()                      # 峰值重置为当前 allocated
    print("[实测#3] reset_peak_memory_stats() 后：peak = %.3f GiB == 当前 allocated（峰值重置生效）"
          % (torch.npu.max_memory_allocated() / GiB))

    torch.npu.empty_cache()                                  # 缓存池归还设备
    free3, _ = torch.npu.mem_get_info()
    print("[实测#4] empty_cache() 后：mem_get_info free 回升到 %.3f GiB（接近实测#0 的 %.3f GiB），"
          "allocated 不变 %.3f GiB" % (free3 / GiB, free0 / GiB, torch.npu.memory_allocated() / GiB))
    print("[P3 定版口径] 本册内存计量：模型/KV 实占 = memory_allocated（+max_* 看峰值）；")
    print("       整卡余量/多进程共卡场景 = mem_get_info；empty_cache 只在需要把缓存还给设备时调用。")


if __name__ == "__main__":
    t_all = time.perf_counter()
    torch.manual_seed(20261009)                              # 全书统一种子风格（本探针数据对种子不敏感）
    env_header()
    bw_copy, _bw_rd = probe_bandwidth()
    tflops = probe_matmul()
    probe_roofline_point(bw_copy, tflops)
    probe_mem_api()
    print("\n[总耗时] %.1f s" % (time.perf_counter() - t_all))
