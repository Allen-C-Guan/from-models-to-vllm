# 01_train215_npu.py —— Book6 消费线重建：train_215.py 的 NPU 薄启动器（Book3 原件零改动）
# 用途：monkey-patch 两处 MPS 探测/autocast 分支，使 train_215.run() 在昇腾 NPU 上以 bf16 autocast 运行——
#   patch ① torch.backends.mps.is_available → True：train_215 L170 的
#     `device = torch.device(device_pref if torch.backends.mps.is_available() else "cpu")`
#     在无 mps 的服务器上恒落 CPU；直通后 device_pref="npu" 生效；
#   patch ② torch.autocast 拦截重定向：train_215 L221-222 的
#     `autocast(device_type=device.type, bf16) if device.type == "mps" else autocast("cpu", enabled=False)`
#     对 device.type=="npu" 走 else 分支=关 autocast（训练降为 fp32，与 Mac bf16 口径漂移）；
#     本启动器把 `autocast("cpu", enabled=False)` 重定向为 `autocast("npu", bf16)`——
#     该调用点是 train_215 关闭 autocast 的唯一途径，eval 段不经 autocast（fp32 由脚本自身保证），
#     故本 patch 不影响 eval 口径。
# 适配 diff 就这两处（详见 research/Book6-推理系统导论/notes/01-消费线重建.md）；
#   断点续跑/产物命名/eval 协议全部沿用 train_215 原文（DESIGN-215M.md §5「eval 协议勿改」）。
# 所属章节：Book6 ch0 前置（消费线重建；plan/00-环境迁移-冲突审视报告.md C-3/D6 路线）
# 运行：source env.sh && ASCEND_RT_VISIBLE_DEVICES=<空闲卡> \
#       python code/Book6-推理系统导论/00-feasibility/01_train215_npu.py --task integrated --out-name npu1
import argparse
import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401 —— import 即注册 npu 后端

HERE = os.path.dirname(os.path.abspath(__file__))
T215 = os.path.abspath(os.path.join(HERE, "..", "..", "Book3-现代开源骨架", "ch10", "train_215.py"))
sys.path.insert(0, os.path.dirname(T215))          # train_215 同目录 import（llama215 等）


def _apply_patches() -> None:
    # patch ①：mps 探测直通
    torch.backends.mps.is_available = lambda: True
    # patch ②：训练段 autocast 重定向（npu bf16，对齐 Mac bf16 口径）
    _autocast = torch.autocast

    def _npu_autocast(device_type="cpu", dtype=None, enabled=True, **kw):
        if device_type == "cpu" and enabled is False and torch.npu.is_available():
            return _autocast(device_type="npu", dtype=torch.bfloat16)
        return _autocast(device_type=device_type, dtype=dtype, enabled=enabled, **kw)

    torch.autocast = _npu_autocast


def _load_train215():
    spec = importlib.util.spec_from_file_location("train_215", T215)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="train_215 NPU 启动器（Book6 消费线重建；task/steps/out-name 语义同原文）")
    ap.add_argument("--task", choices=["smoke", "integrated", "full"], default="smoke")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--device", default="npu", help="npu（默认；透传给 train_215.run 的 device_pref）")
    a = ap.parse_args()
    _apply_patches()
    t215 = _load_train215()
    t215.run(a.task, a.out_name, a.steps, device_pref=a.device)


if __name__ == "__main__":
    main()
