"""Compact diagnosis of v1/v2 kernel metrics: overfitting + generalization gaps."""
import json
import sys

KEYS = ("best_val_loss", "train_time_s", "s_per_step", "gpu_mem_gb",
        "teacher_forced_masked_recovery", "val_ppl", "n_params")


def load(p):
    with open(p) as f:
        return json.load(f)


def show(tag, m):
    print(f"\n===== {tag} =====")
    cfg = m.get("config", {})
    print("cfg:", {k: cfg.get(k) for k in
                   ("batch_size", "steps_per_model", "lr", "dropout", "weight_decay",
                    "d_model", "n_layer", "block_size")})
    for k in ("diffusion", "ar"):
        d = m.get(k, {})
        print(k, {kk: d.get(kk) for kk in KEYS if kk in d})
        h = d.get("history", [])
        if h:
            pts = [(p["step"], round(p.get("train_loss", -1), 3), round(p.get("val_loss", -1), 3))
                   for p in h]
            print(f"  {k} history (step, train, val): {pts}")
    inf = m.get("infill_comparison", {})
    if inf:
        print("infill:", {k: v for k, v in inf.items()})
    sp = m.get("speed_bench", {})
    if sp:
        print("speed:", {k: {kk: v.get(kk) for kk in ("tokens_per_s", "forward_passes")}
                         for k, v in sp.items()})
    for s in m.get("generation_samples", [])[:3]:
        print("sample:", repr(s.get("diffusion_completion", ""))[:110])


for tag, p in (("V1 (2500 steps, b16)", "/tmp/v1out/metrics.json"),
               ("V2 (12000 steps, b32)", "/tmp/v2out/metrics.json")):
    try:
        show(tag, load(p))
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: FAILED {e}", file=sys.stderr)
        raise
