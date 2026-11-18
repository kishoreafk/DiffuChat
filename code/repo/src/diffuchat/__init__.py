"""DiffuChat — a miniature masked-diffusion language model (LLaDA/MDLM style)
trained from scratch, with an editable-answer chat product on top.

Package layout (see README.md for the full spec):
    config.py      dataclasses + presets (smoke / cpu / gpu)
    data.py        CharTokenizer (+[MASK] id 65), TinyShakespeare loader,
                   window sampling, "you :/ai :" dialogue-pair mining
    model.py       DenoiseLM — bidirectional transformer, NO causal mask
    train.py       MDLM objective (CE on masked positions only)
    generate.py    Canvas + denoise_step + generate (3 strategies) + edit()
    interactive.py terminal chat product
    ar_baseline.py matched causal (AR) char-GPT for the evaluation
    evaluate.py    the 3 experiments -> artifacts/eval_report.json
"""

__version__ = "0.1.0"
