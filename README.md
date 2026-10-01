# Qwen2.5-3B Math Reasoning with LoRA

This project studies whether parameter-efficient fine-tuning improves mathematical reasoning accuracy on GSM8K.

## Research question

> Does LoRA fine-tuning of Qwen2.5-3B improve held-out GSM8K exact-match accuracy relative to the frozen base model?

## Historical experiment

The original notebook successfully trained a Qwen2.5-3B-Instruct model with 4-bit quantization and LoRA.

Recorded training facts:
- GSM8K: 7,473 training examples and 1,319 test examples
- one epoch / 468 steps
- effective batch size 16
- learning rate 5e-6
- 14,966,784 trainable parameters out of 3,100,905,472
- 1:19:27 training wall time

The notebook then generated answers for 10 hand-picked questions for the base and fine-tuned models.

### Important limitation

Those 10 generations are qualitative checks. The notebook does not calculate exact-match accuracy over the official 1,319-example GSM8K test split.

## Controlled evaluation

This branch adds `scripts/evaluate_gsm8k.py` to evaluate:

1. frozen Qwen2.5-3B-Instruct
2. the trained LoRA adapter

Both use the same deterministic inference protocol and the same GSM8K test split.

Primary metric:
- exact-match numerical accuracy

Secondary metrics:
- answer parse rate
- strict `####` format rate
- mean generation latency

## Run

For a smoke test:

```bash
python scripts/evaluate_gsm8k.py --mode base --limit 100 --output results/base_100.json
python scripts/evaluate_gsm8k.py --mode adapter --adapter ./qwen-gsm8k-final --limit 100 --output results/adapter_100.json
python scripts/analyze_results.py results/base_100.json results/adapter_100.json
```

For the full held-out evaluation, remove `--limit`.

## Hypothesis

> LoRA fine-tuning will increase held-out GSM8K exact-match accuracy at a small trainable-parameter fraction.

This is falsifiable.

## Current status

Implemented:
- LoRA training prototype
- historical run record
- automated held-out GSM8K evaluator
- base-vs-adapter comparison protocol
- answer extraction tests

Not yet demonstrated:
- a full 1,319-example base-vs-adapter accuracy comparison in this branch

## Research trajectory

`PEFT → DPO → knowledge distillation → reasoning under compute constraints`

This project provides the weight-side counterpart to the test-time reasoning experiments: instead of spending more inference compute, it asks how much reasoning capability can be gained by a very small trainable parameter budget.