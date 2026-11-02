"""第13章：从原始记录生成随书快照与表格；不重新测量、不填补失败值。

source ./env.sh; python code/Book6-推理系统导论/ch13/summarize.py --run matrix1 --probes probes
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from run_matrix import CASES


def compare_probe(actual, reference):
    assert actual["prompt_ids"] == reference["prompt_ids"]
    n = len(actual["prompt_ids"])
    a = actual["response"]["choices"][0]
    b = reference["response"]["choices"][0]
    deltas = [abs(x-y) for x, y in zip(a["logprobs"]["token_logprobs"][:n],
                                     b["logprobs"]["token_logprobs"][:n])
              if x is not None and y is not None]
    assert len(deltas) == n-1
    return {"prefix_tokens": n, "max_abs_logprob_delta": max(deltas),
            "mean_abs_logprob_delta": sum(deltas) / len(deltas),
            "generated_ids_equal": a["token_ids"] == b["token_ids"],
            "generated_ids": a["token_ids"], "reference_generated_ids": b["token_ids"]}


def hbm_peaks(path, start, end):
    peaks = {}
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        if not start <= sample["time"] <= end:
            continue
        card = None
        for row in sample["stdout"].splitlines():
            found = re.match(r"\|\s*(\d+)\s+910B", row)
            if found:
                card = found[1]
            usage = re.search(r"(\d+)\s*/\s*65536", row)
            if card is not None and usage:
                peaks[card] = max(peaks.get(card, 0), int(usage[1]))
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="matrix1")
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--repeat", default="repeat")
    ap.add_argument("--controls", default="sync")
    a = ap.parse_args()
    log = Path("log/book6-ch13")
    dst = Path(__file__).parent / "results"
    dst.mkdir(exist_ok=True)
    summary = {"run": a.run, "cases": {}, "repeats": {}, "controls": {}, "evidence_sha256": {}}
    summary["interpretation_notes"] = {
        "pass_scope": "Health check and prescribed requests completed; not accuracy certification",
        "pipeline_review": "Default-scheduling PP long-input generation repeats adjacent tokens. See repeat/sync controls; do not rank it as valid acceleration.",
        "precision_scope": "Synthetic fixed-prefix logprob and generation comparisons; no task-level quality evaluation"}
    for p in [log / "capacity_model.json", log / "mechanisms_verified.json", *log.glob("collectives_hccl*.json")]:
        shutil.copy2(p, dst / p.name)
    def evidence(path):
        summary["evidence_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence(log / "hardware.txt")
    evidence(log / "topology.txt")
    ref_probe_path = log / f"serving_{a.probes}_tp1.json"
    ref_probe = json.loads(ref_probe_path.read_text())["correctness_probe"] if ref_probe_path.exists() else None
    reference = json.loads((log / f"serving_{a.run}_tp1.json").read_text())
    for case, (tp, pp, dp, dcp, ep, sp) in CASES.items():
        folder = log / a.run / case
        status_path = folder / "status.json"
        if not status_path.exists():
            summary["cases"][case] = {"state": "not_completed"}
            continue
        evidence(status_path)
        status = json.loads(status_path.read_text())
        shutil.copy2(status_path, dst / f"status_{a.run}_{case}.json")
        card_count = 2 if case == "replicas2" else tp * pp * dp
        entry = {"state": status["state"], "cards": card_count,
                 "tp": tp, "pp": pp, "dp": dp, "dcp": dcp, "ep": ep, "sp_requested": sp,
                 "error": status.get("error")}
        texts = []
        for p in sorted(folder.glob("server-*.log")):
            evidence(p)
            texts.append(p.read_text(errors="replace"))
        text = "\n".join(texts)
        entry["runtime_evidence_lines"] = [line for line in text.splitlines() if any(
            phrase in line for phrase in ["Sequence-parallel MoE is", "GPU KV cache size:",
                                         "Actual usage:", "Using attention backend", "is assigned as DP rank"])]
        entry["kv_capacity_log_tokens"] = [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)]
        entry["sp_log_states"] = sorted(set(re.findall(r"Sequence-parallel MoE is (enabled|disabled)", text)))
        entry["weight_memory_log_GB"] = [float(x) for x in re.findall(r"Loading model weights took ([\d.]+) GB", text)]
        if status["state"] == "pass":
            evidence(folder / "hbm-samples.jsonl")
            peaks = hbm_peaks(folder / "hbm-samples.jsonl", status["benchmark_start_unix"], status["benchmark_end_unix"])
            used = {c for cmd in status["commands"] for c in cmd["devices"].split(",")}
            entry["sampled_driver_hbm_peak_mib"] = {c: v for c, v in peaks.items() if c in used}
        source = log / f"serving_{a.run}_{case}.json"
        if source.exists():
            evidence(source)
            data = json.loads(source.read_text())
            shutil.copy2(source, dst / source.name)
            entry["measurements"] = []
            for i, row in enumerate(data["rows"]):
                item = {k: v for k, v in row.items() if k != "records"}
                item["output_tokens_per_s_per_card"] = row["output_tokens_per_s"] / card_count
                expected = reference["rows"][i]["records"][0]["token_ids"]
                item["exact_greedy_sequence_matches_vs_tp1"] = sum(r["token_ids"] == expected for r in row["records"])
                entry["measurements"].append(item)
            probe_path = source
            if "correctness_probe" not in data:
                probe_path = log / f"serving_{a.probes}_{case}.json"
                if probe_path.exists():
                    data["correctness_probe"] = json.loads(probe_path.read_text())["correctness_probe"]
                    shutil.copy2(probe_path, dst / probe_path.name)
                    evidence(probe_path)
            if ref_probe and "correctness_probe" in data:
                probe = data["correctness_probe"]
                assert probe["prompt_ids"] == ref_probe["prompt_ids"]
                n = len(probe["prompt_ids"])
                actual = probe["response"]["choices"][0]["logprobs"]["token_logprobs"][:n]
                expected = ref_probe["response"]["choices"][0]["logprobs"]["token_logprobs"][:n]
                deltas = [abs(x-y) for x, y in zip(actual, expected) if x is not None and y is not None]
                assert len(deltas) == n-1
                entry["teacher_forced_logprob_max_abs_delta"] = max(deltas)
                entry["teacher_forced_logprob_mean_abs_delta"] = sum(deltas) / len(deltas)
        summary["cases"][case] = entry
    for run, section in [(a.repeat, "repeats"), (a.controls, "controls")]:
        base_path = log / f"serving_{run}_tp1.json"
        base = json.loads(base_path.read_text()) if base_path.exists() else None
        for source in sorted(log.glob(f"serving_{run}_*.json")):
            data = json.loads(source.read_text())
            case = source.stem.removeprefix(f"serving_{run}_")
            status_path = log / run / case / "status.json"
            if not status_path.exists():
                continue
            evidence(source)
            evidence(status_path)
            for p in (status_path.parent).glob("server-*.log"):
                evidence(p)
            shutil.copy2(source, dst / source.name)
            shutil.copy2(status_path, dst / f"status_{run}_{case}.json")
            entry = {"state": json.loads(status_path.read_text())["state"],
                     "measurements": [{k: v for k, v in row.items() if k != "records"}
                                      for row in data["rows"]]}
            if base and "long_correctness_probe" in data and "long_correctness_probe" in base:
                entry["long_prefix_comparison"] = compare_probe(data["long_correctness_probe"], base["long_correctness_probe"])
                expected = base["rows"][1]["records"][0]["token_ids"]
                entry["long_generation_exact_matches"] = sum(r["token_ids"] == expected for r in data["rows"][1]["records"])
            summary[section][case] = entry
    (dst / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ["| 配置 | 卡数 | 短输入TTFT p50(ms) | 短输入TPOT p50(ms) | 长输入TTFT p50(ms) | 长输入TPOT p50(ms) | 并发8输出token/s | 每卡token/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, value in summary["cases"].items():
        rows = value.get("measurements", [])
        if value["state"] != "pass" or len(rows) != 3:
            lines.append(f"| {case} | {value.get('cards', '—')} | 未完成 | — | — | — | — | — |")
            continue
        short, long, concurrent = rows
        values = [short['ttft_ms']['p50'], short['tpot_ms']['p50'], long['ttft_ms']['p50'], long['tpot_ms']['p50'], concurrent['output_tokens_per_s'], concurrent['output_tokens_per_s_per_card']]
        label = case + " †" if case in {"pp2", "tp2pp2"} else case
        lines.append(f"| {label} | {value['cards']} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "† 默认调度下PP长输入生成出现相邻token重复；原始读数保留，不作为正确推理的加速结论。见正文复核与summary中的repeats/controls。"]
    (dst / "measurements.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
