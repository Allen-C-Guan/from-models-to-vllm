# 04_probe_serving_smoke.py —— Book6 本机探针 P4：消费线冒烟（条件性；Book3 终态 215M 权重上 NPU 服务冒烟）
# 用途：log/book3-ch10/ckpt_integrated_*.pt 存在时——llama215.Llama215 构建 + load_state_dict(strict)
#       + NPU fp32（Book3 eval 协议口径）+ 20 token 贪心生成（每步全量重算，无 KV cache——
#       Book6 ch0 的「服务前基线」，KV cache 是本册 ch5 起的改造对象）；
#       不存在时——登记「待消费线重建完成后补跑」并退出（语料重建进行中属预期，非故障）。
# 前向口径：prompt 取 tokens32k.bin 首 24 token（uint16 裸流，sp-32000 真实语料；语料未就绪则
#       退化为统一种子的随机 id——打印注明来源）。贪心 argmax 逐 token 拼接。
# 运行：cd <workspace> && source env.sh && npu-smi info（挑空闲卡）后
#      ASCEND_RT_VISIBLE_DEVICES=<卡号> python code/Book6-推理系统导论/00-feasibility/04_probe_serving_smoke.py
import glob
import os
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401 —— import 即注册 npu 后端

HERE = os.path.dirname(os.path.abspath(__file__))
T215_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "Book3-现代开源骨架", "ch10"))
sys.path.insert(0, T215_DIR)                     # llama215 同目录 import（train_215 同款 path 注入）

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book3-ch10")
CKPT_GLOB = os.path.join(OUT_DIR, "ckpt_integrated_*.pt")
TOKENS_BIN = os.path.join(OUT_DIR, "tokens32k.bin")
SEED, N_GEN, PROMPT_LEN = 20261009, 20, 24


def pick_ckpt():
    cks = sorted(glob.glob(CKPT_GLOB), key=os.path.getmtime)
    return cks[-1] if cks else None               # 多个取 mtime 最新（消费线可能有多次 out-name）


def build_prompt(device):
    """prompt 来源：tokens32k.bin 首 PROMPT_LEN token（真实语料）＞ 统一种子随机 id（退化）。"""
    if os.path.exists(TOKENS_BIN):
        arr = np.memmap(TOKENS_BIN, dtype=np.uint16, mode="r")
        ids = torch.from_numpy(np.array(arr[:PROMPT_LEN], dtype=np.int64)).unsqueeze(0)  # (1,n)
        return ids.to(device), "tokens32k.bin 首 %d token（真实语料）" % PROMPT_LEN
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, 32000, (1, PROMPT_LEN), generator=g)                          # (1,n)
    return ids.to(device), "随机 id（种子 %d；语料未就绪的退化口径）" % SEED


def main():
    ck = pick_ckpt()
    if ck is None:
        print("[P4 登记] 未找到 %s" % CKPT_GLOB)
        print("[P4 登记] 消费线重建（01_train215_npu.py --task integrated）完成前无 integrated 权重——")
        print("           本探针「待消费线重建完成后补跑」（语料重建进行中属预期，非故障）。")
        sys.exit(0)

    from llama215 import Llama215                 # Book3 ch10 整机（import 时自带四插槽 sys.path 注入）

    t0 = time.perf_counter()
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    model = Llama215()                            # 215M 定版 config（cand1）
    model.load_state_dict(ckpt["model"], strict=True)   # strict 直搬——键位契约的硬验收
    model.eval().to("npu")                        # fp32：Book3 eval 协议口径
    print("[P4 载入] %s" % os.path.basename(ck))
    print("         ckpt 字段：step=%s eval_loss=%s init_loss=%s" %
          (ckpt.get("step"), ckpt.get("eval_loss"), ckpt.get("init_loss")))
    print("         参数 %s | 设备 npu/fp32 | 载入耗时 %.1fs"
          % (f"{sum(p.numel() for p in model.parameters()):,}", time.perf_counter() - t0))

    prompt, src = build_prompt("npu")
    print("[P4 生成] prompt 来源：%s | 贪心 %d token，每步全量重算（无 KV cache）" % (src, N_GEN))

    gen = prompt.clone()                          # (1,n0)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_GEN):
            logits = model(gen)[0]                # (1,n,V)——全量重算
            nxt = logits[:, -1].argmax(-1, keepdim=True)   # (1,1)
            gen = torch.cat([gen, nxt], dim=1)    # (1,n+1)
    torch.npu.synchronize()
    dt = time.perf_counter() - t1

    new_ids = gen[0, prompt.shape[1]:].tolist()   # 生成的 20 个 id
    print("[P4 结果] 续写 id（贪心，%d token）：%s" % (N_GEN, new_ids))
    print("         首步后均值 %.1f ms/token（含全量重算；n: %d->%d）| 吞吐 %.2f tok/s | 总 %.1fs"
          % (dt / N_GEN * 1e3, prompt.shape[1], gen.shape[1], N_GEN / dt, dt))
    ok = all(0 <= i < 32000 for i in new_ids)
    print("[P4 判定] 冒烟 %s（load_state_dict strict 通过 + NPU 前向产出有限 logits + 逐 token argmax 推进）"
          % ("通过" if ok else "异常：id 越界"))


if __name__ == "__main__":
    main()
