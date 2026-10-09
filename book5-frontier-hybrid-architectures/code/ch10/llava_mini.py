# llava_mini.py —— Book5 ch10 正式件：LLaVA mini（冻结 CLIP-base + 冻结 Qwen3-0.6B + 一层 projector 微调）
# 用途：ch10（能力扩展专题）多模态投影的最小可跑正身——LLaVA 三件套的玩具级复刻：
#   ① 视觉塔（冻结）：openai/clip-vit-base-patch32 的 CLIPVisionModel——末层 patch 隐状态均值池化
#      → 768 维图像特征（真实 LLaVA 用 ViT-L/14@336 的 256-576 token 序列；本件单 token 玩具简化，
#      偏离声明预登记——单 token 是「投影对齐两个嵌入空间」的最小形态）；
#   ② 语言塔（冻结）：Qwen/Qwen3-0.6B（d=1024，bf16）——文本侧零训练；
#   ③ projector（唯一可学件）：一层线性 768→1024（无 bias，786,432 参）——把图像特征推进语言塔的
#      嵌入空间，替换 <image> 占位 token 的嵌入（LLaVA-1 机制的最小实现）。
# 数据：Visual Genome 子集（CC BY 4.0）——数百图，instruction 配对（模板问句 × 区域描述应答）；
#   图像一律落 log/book5-ch10/vg/（缓存红线），零图像入书仓（零图像红线）。
# 【判据预注册（跑前写定 2026-10-08；notes/08 同文留档）】
#   - prep 判据：≥300 图成功落地 + manifest 配对完整（图/问/答三字段齐全）；
#   - smoke 判据：20 步 loss 有限且下降、projector 梯度有限；
#   - formal 判据：数百步 loss 曲线持续下降（过拟合口径如实报——数百图小样本必然过拟合，读数意义
#     在「投影学得动」而非泛化）；两条生成样例落 JSON（人读对照：应答是否开始与图像内容相关——
#     定性判读，不设自动判分）。
# 运行方式：cd 工作区根目录 && source env.sh &&
#   python "code/ch10/llava_mini.py" --task prep            # VG 子集下载+配对（~5 min 网络）
#   python "code/ch10/llava_mini.py" --task smoke --out-name s1
#   python "code/ch10/llava_mini.py" --task formal --out-name run1   # ≈15-20 min
# 产物（log/book5-ch10/，不入库）：vg/（图像+manifest+原始 zip）+ llava_mini_{task}_{out}.json
#   （loss 曲线+生成样例+参数账）+ curve_llava_{out}.csv。
import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "log", "book5-ch10")
VG_DIR = os.path.join(OUT_DIR, "vg")
SEED = 20261002

CLIP_ID = "openai/clip-vit-base-patch32"
LM_ID = "Qwen/Qwen3-0.6B"
IMG_TOK = "<|image_pad|>"                                       # Qwen3 词表既有多模态占位符（不新增词）
N_IMAGES, N_HOLDOUT, BATCH, MAXLEN = 340, 20, 16, 96
INSTRUCTIONS = ["Describe this image.", "What is in this image?", "Write one sentence about this image."]

VG_HOSTS = [
    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset",
    "http://visualgenome.org/static/data/dataset",
]


# ---------------- task=prep：VG 子集下载 + instruction 配对 ----------------
def download(url, dst, tries=3):
    """curl 带 -f（HTTP 错误不落盘）+ 断点续传 -C - + 长超时——VG 镜像 404 回 HTML 错误页、
    Washington 主机 ~0.3MB/s 慢速（region_descriptions.zip 127MB 需 >5min，300s 超时必断）。"""
    for i in range(tries):
        r = subprocess.run(["curl", "-sfL", "-C", "-", "--max-time", "1800", "-o", dst, url])
        if r.returncode == 0 and os.path.getsize(dst) > 0:
            return True
        time.sleep(2)
    return False


VG_IMG_HOSTS = ["https://cs.stanford.edu/people/rak248/VG_100K",
                "https://cs.stanford.edu/people/rak248/VG_100K_2"]


def fetch_vg_image(image_id):
    """VG 图像双目录探测（无 image_metadata 时的 URL 重建：<id>.jpg 在 VG_100K 或 VG_100K_2）。"""
    fname = f"{image_id}.jpg"
    dst = os.path.join(VG_DIR, fname)
    if os.path.exists(dst):
        return fname, dst
    for host in VG_IMG_HOSTS:
        if download(f"{host}/{fname}", dst):
            return fname, dst
    return None, None


def prep(out_name):
    """VG 子集下载 + instruction 配对。

    标注源=region_descriptions.json.zip（Washington 镜像 2026-10-08 实测可达；image_metadata.json.zip
    各镜像 404——弃用，改「双目录探测 + PIL 尺寸复核」重建图源）。"""
    os.makedirs(VG_DIR, exist_ok=True)
    name = "region_descriptions.json"                # zip 内文件名已含 .json——解压目标与 zip 同名异后缀
    dst = os.path.join(VG_DIR, f"{name}.zip")
    jsn = os.path.join(VG_DIR, name)
    if not os.path.exists(jsn):
        if not os.path.exists(dst):
            got = any(download(f"{h}/{name}.zip", dst) for h in VG_HOSTS)
            assert got, f"标注下载失败：{name}（hosts={VG_HOSTS}）"
        subprocess.run(["unzip", "-o", dst, "-d", VG_DIR], check=True)
    regions_by_img = {}
    for row in json.load(open(jsn, encoding="utf-8")):
        for r in row.get("regions", []):
            regions_by_img.setdefault(r["image_id"], []).append(r)
    print(f"[prep] region_descriptions：{len(regions_by_img)} 图的区域描述", flush=True)

    # 候选=image_id 升序、区域 ≥3 条；逐图双目录探测 + PIL 尺寸复核（宽 ≥400 且高 ≥300）
    from PIL import Image
    manifest, failures, skipped = [], 0, 0
    for image_id in sorted(regions_by_img):
        if len(manifest) >= N_IMAGES:
            break
        rs = regions_by_img[image_id]
        if len(rs) < 3:
            continue
        fname, path = fetch_vg_image(image_id)
        if fname is None:
            failures += 1
            continue
        try:
            with Image.open(path) as im:
                if im.width < 400 or im.height < 300:
                    os.remove(path)
                    skipped += 1
                    continue
        except Exception:
            os.remove(path)
            failures += 1
            continue
        phrase = max(rs, key=lambda r: len(r.get("phrase", "")))["phrase"].strip()
        phrase = phrase[0].upper() + phrase[1:]
        if not phrase.endswith("."):
            phrase += "."
        manifest.append({"image_id": image_id, "file": fname,
                         "url": f"cs.stanford.edu/VG_100K*/{fname}",
                         "instruction": INSTRUCTIONS[len(manifest) % len(INSTRUCTIONS)],
                         "response": phrase,
                         "license": "Visual Genome image, CC BY 4.0 (http://visualgenome.org/license)"})
        if len(manifest) % 50 == 0:
            print(f"  [prep] {len(manifest)}/{N_IMAGES} 图落地（失败 {failures} 小图跳过 {skipped}）", flush=True)
    n_train = max(1, len(manifest) - N_HOLDOUT)
    for i, row in enumerate(manifest):
        row["split"] = "train" if i < n_train else "holdout"
    path = os.path.join(VG_DIR, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    bytes_ = sum(os.path.getsize(os.path.join(VG_DIR, r["file"])) for r in manifest)
    print(f"[prep 判读] {len(manifest)} 图配对落地（train {n_train}/holdout {len(manifest) - n_train}，"
          f"下载失败 {failures} 小图跳过 {skipped}，共 {bytes_ / 1e6:.1f} MB）| manifest：{path}", flush=True)
    assert len(manifest) >= 300, "prep 判据未达：≥300 图"
    with open(os.path.join(OUT_DIR, f"llava_mini_prep_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump({"n_images": len(manifest), "n_train": n_train, "n_holdout": len(manifest) - n_train,
                   "failures": failures, "skipped_small": skipped, "bytes": bytes_, "seed": SEED,
                   "license": "VG images CC BY 4.0——图像只落 log/，零图像入书仓"}, f, ensure_ascii=False, indent=2)


# ---------------- 模型三件套：冻结 CLIP + 冻结 Qwen3 + 一层 projector ----------------
def load_towers(device):
    from transformers import AutoProcessor, CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer
    try:
        clip = CLIPVisionModel.from_pretrained(CLIP_ID, dtype=torch.float32)
    except TypeError:                                            # 老接口回退（torch_dtype→dtype 改名期）
        clip = CLIPVisionModel.from_pretrained(CLIP_ID, torch_dtype=torch.float32)
    clip = clip.to(device).eval()
    for p in clip.parameters():
        p.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(CLIP_ID)
    tok = AutoTokenizer.from_pretrained(LM_ID)
    try:
        lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.bfloat16)
    except TypeError:
        lm = AutoModelForCausalLM.from_pretrained(LM_ID, torch_dtype=torch.bfloat16)
    lm = lm.to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    img_id = tok.convert_tokens_to_ids(IMG_TOK)
    assert isinstance(img_id, int) and img_id >= 0, f"词表无占位符 {IMG_TOK}"
    d = lm.config.hidden_size
    proj = nn.Linear(clip.config.hidden_size, d, bias=False).to(device)
    nn.init.normal_(proj.weight, mean=0.0, std=0.02)
    return clip, proc, lm, tok, proj, img_id


@torch.no_grad()
def image_feats(clip, proc, files, device):
    """(B,3,224,224) → 末层 patch 隐状态均值池化 (B,768)——CLIP 视觉塔冻结只前向。"""
    from PIL import Image
    imgs = [Image.open(os.path.join(VG_DIR, f)).convert("RGB") for f in files]
    px = proc(images=imgs, return_tensors="pt").to(device)       # pixel_values (B,3,224,224)
    h = clip(**px).last_hidden_state                             # (B,50,768)——1 CLS + 49 patch
    return h[:, 1:, :].mean(dim=1)                               # (B,768) patch 均值（单 token 玩具简化）


def build_batch(tok, rows, img_id, device):
    """instruction 配对批：prompt=用户问句（含图像占位），response=区域描述；labels 只监督应答段。"""
    ids_list, labels_list, n_img_tok = [], [], []
    for r in rows:
        prompt = (f"<|im_start|>user\n{IMG_TOK}{r['instruction']}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        resp = f"{r['response']}<|im_end|>"
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        r_ids = tok(resp, add_special_tokens=False).input_ids
        ids_list.append(p_ids + r_ids)
        labels_list.append([-100] * len(p_ids) + r_ids)
        n_img_tok.append(p_ids.count(img_id))
    L = max(len(x) for x in ids_list)
    assert L <= MAXLEN, L
    ids = torch.full((len(ids_list), L), tok.pad_token_id or img_id, dtype=torch.long)
    labels = torch.full((len(ids_list), L), -100, dtype=torch.long)
    att = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids_list, labels_list)):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids.to(device), labels.to(device), att.to(device)


def embed_with_image(lm, proj, ids, feats, img_id):
    """文本嵌入 + 图像 token 位替换：emb(占位) → proj(图像特征)——LLaVA 投影机制的最小实现。"""
    emb = lm.get_input_embeddings()(ids)                         # (B,L,d)
    mask = ids == img_id                                        # (B,L)
    assert mask.any(), "批内无图像占位 token"
    assert mask.sum() == mask.shape[0], "每序列应恰有 1 个图像 token"
    emb = emb.clone()
    emb[mask] = proj(feats).to(emb.dtype)                        # (B,d) → 占位位
    return emb                                                  # (B,L,d)


def run_train(task, out_name, steps, device_pref="mps"):
    device = torch.device(device_pref if torch.backends.mps.is_available() else "cpu")
    manifest = json.load(open(os.path.join(VG_DIR, "manifest.json"), encoding="utf-8"))
    train_rows = [r for r in manifest if r["split"] == "train"]
    hold_rows = [r for r in manifest if r["split"] == "holdout"][:2]
    clip, proc, lm, tok, proj, img_id = load_towers(device)
    n_proj = sum(p.numel() for p in proj.parameters())
    n_lm = sum(p.numel() for p in lm.parameters())
    n_clip = sum(p.numel() for p in clip.parameters())
    trainable = [p for p in proj.parameters() if p.requires_grad]
    print(f"[三件套] 冻结 CLIP-base {n_clip / 1e6:.1f}M + 冻结 Qwen3-0.6B {n_lm / 1e6:.1f}M + "
          f"可学 projector {n_proj:,}（768→{lm.config.hidden_size}，一层无 bias）| device {device}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    curve, t0 = [], time.perf_counter()
    for step in range(1, steps + 1):
        rows = [train_rows[i] for i in rng.integers(0, len(train_rows), size=BATCH)]
        ids, labels, att = build_batch(tok, rows, img_id, device)
        with torch.no_grad():
            feats = image_feats(clip, proc, [r["file"] for r in rows], device)   # (B,768)
        emb = embed_with_image(lm, proj, ids, feats, img_id)     # (B,L,d)
        logits = lm(inputs_embeds=emb, attention_mask=att).logits  # (B,L,V)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                               labels[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        curve.append({"step": step, "loss": round(loss.item(), 5)})
        if step % 5 == 0 or step == 1:
            gnorm = float(proj.weight.grad.norm()) if proj.weight.grad is not None else None
            print(f"  step {step}/{steps} loss {loss.item():.4f} grad|w| {gnorm}", flush=True)
            assert gnorm is None or np.isfinite(gnorm), "projector 梯度非有限"

    # ---- 生成样例（2 张 holdout 图，贪心 64 token；Qwen3 思考模式介入时取末次 </think> 之后为答） ----
    # 注意：样例必须走 prompt-only 路径（不含 VG 参考应答——smoke s1 首版误用 build_batch 把参考
    # 应答拼进了生成前缀，样例判读被污染；本版独立拼 prompt，参考只在 JSON 里作对照字段）
    samples = []
    for r in hold_rows:
        prompt = (f"<|im_start|>user\n{IMG_TOK}Describe this image.<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids], device=device)                   # (1,L_p)
        att = torch.ones_like(ids)
        with torch.no_grad():
            feats = image_feats(clip, proc, [r["file"]], device)
            emb = embed_with_image(lm, proj, ids, feats, img_id)
            out = lm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=64,
                              do_sample=False, pad_token_id=img_id)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        if "</think>" in text:                                       # 思考模式介入：取末次 </think> 之后
            text = text.rsplit("</think>", 1)[1].strip()
        samples.append({"file": r["file"], "instruction": "Describe this image.",
                        "vg_reference": r["response"], "generated": text})
        print(f"  [样例 {r['file']}] VG 参考：{r['response']}\n    生成：{text}", flush=True)

    wall = time.perf_counter() - t0
    losses = [c["loss"] for c in curve]
    meta = {"task": task, "seed": SEED, "device": str(device), "steps": steps, "batch": BATCH,
            "models": {"clip": CLIP_ID, "lm": LM_ID, "frozen": True},
            "params": {"clip": n_clip, "lm": n_lm, "projector": n_proj},
            "image_token": "单 token 均值池化（真实 LLaVA 为 256-576 token 序列——玩具简化预登记）",
            "data": {"n_train": len(train_rows), "n_holdout": len(hold_rows),
                     "license": "VG CC BY 4.0，图像只落 log/ 不入书仓"},
            "recipe": "AdamW(0.9,0.95,wd0)+clip1.0+lr1e-3（仅 projector 可学）bf16 语言塔",
            "loss_first": losses[0], "loss_last10_mean": round(float(np.mean(losses[-10:])), 5),
            "loss_curve": curve, "samples": samples,
            "wall_sec": round(wall, 1),
            "verdict": "PASS" if (all(np.isfinite(losses)) and losses[-1] < losses[0]) else "CHECK"}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"llava_mini_{task}_{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, f"curve_llava_{out_name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss"])
        w.writeheader()
        w.writerows(curve)
    torch.save({"proj": proj.state_dict(), "step": steps},
               os.path.join(OUT_DIR, f"ckpt_llava_{out_name}.pt"))
    print(f"[完成] loss {losses[0]:.4f} → {losses[-1]:.4f} | wall {wall / 60:.1f} min | "
          f"产物 {OUT_DIR}/llava_mini_{task}_{out_name}.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description="ch10 LLaVA mini：冻结双塔 + 一层 projector 微调（判据预注册见文件头）")
    ap.add_argument("--task", choices=["prep", "smoke", "formal"], default="prep")
    ap.add_argument("--steps", type=int, default=None,
                    help="smoke 20 / formal 默认 600（数百步档；贴窗以实测 s/step 调整留痕）")
    ap.add_argument("--out-name", default="run1")
    args = ap.parse_args()
    if args.task == "prep":
        prep(args.out_name)
        return
    steps = args.steps or (20 if args.task == "smoke" else 600)
    run_train(args.task, args.out_name, steps)


if __name__ == "__main__":
    main()
