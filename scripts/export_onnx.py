#!/usr/bin/env python3
"""Export the trained DenoiseLM checkpoint (from Kaggle) to ONNX + tokenizer assets.

Inputs : download/diffuchat_artifacts/checkpoints/diffusion_best.pt (or --ckpt)
         download/diffuchat_artifacts/tokenizer.json
Outputs: /home/z/my-project/model/diffusion_lm.onnx
         /home/z/my-project/model/tokenizer.json
         /home/z/my-project/model/model_meta.json  (params, val loss, step)

Fixed input [batch, 256] int64 -> logits [batch, 256, V] fp32.
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to diffusion_best.pt")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--metrics", default=None, help="optional metrics.json for meta")
    ap.add_argument("--outdir", default="/home/z/my-project/model")
    args = ap.parse_args()

    sys.path.insert(0, "/home/z/my-project/work/diffuchat/src")
    from diffuchat.model import DenoiseLM
    from diffuchat.config import DenoiseLMConfig

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_key = "mcfg" if "mcfg" in ck else "cfg"
    mcfg = DenoiseLMConfig(
        **{k: v for k, v in ck[cfg_key].items() if k in DenoiseLMConfig.__dataclass_fields__}
    )
    model = DenoiseLM(mcfg)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    dummy = torch.ones(1, mcfg.block_size, dtype=torch.long)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = outdir / "diffusion_lm.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,  # legacy TorchScript exporter: inline weights, native opset 17
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
        )
    # inline external weight data (torch >= 2.15 writes .onnx.data for big models)
    import onnx
    from onnx import TensorProto
    m = onnx.load(str(onnx_path), load_external_data=True)
    n_ext = sum(1 for i in m.graph.initializer if i.data_location == TensorProto.EXTERNAL)
    if n_ext:
        for init in m.graph.initializer:
            if init.data_location == TensorProto.EXTERNAL:
                arr = onnx.numpy_helper.to_array(init)
                fresh = onnx.numpy_helper.from_array(arr, init.name)
                init.CopyFrom(fresh)
        onnx.save(m, str(onnx_path))
    stray = onnx_path.with_suffix(".onnx.data")
    if stray.exists():
        stray.unlink()

    # copy tokenizer + meta
    tok_dst = outdir / "tokenizer.json"
    tok_dst.write_bytes(Path(args.tokenizer).read_bytes())

    meta = {"nParams": n_params, "step": ck.get("step"), "valLoss": ck.get("val_loss"),
            "exportedFrom": str(args.ckpt), "blockSize": mcfg.block_size,
            "vocabSize": mcfg.vocab_size}
    if args.metrics and Path(args.metrics).exists():
        m = json.loads(Path(args.metrics).read_text())
        meta["runConfig"] = m.get("config")
    (outdir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"meta: {json.dumps(meta)}")

    # quick parity check: torch vs onnxruntime (full 256 and short 40-token inputs)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for T in (mcfg.block_size, 40):
        test_ids = torch.randint(0, mcfg.vocab_size - 1, (1, T))
        with torch.no_grad():
            ref = model(test_ids).numpy()
        got = sess.run(["logits"], {"input_ids": test_ids.numpy()})[0]
        diff = float(abs(ref - got).max())
        print(f"parity check T={T}: max|torch-onnx| = {diff:.2e} (should be < 1e-3)")
        assert diff < 1e-3, "ONNX parity check failed"
    print("OK")


if __name__ == "__main__":
    main()
