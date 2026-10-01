# Reproducibility

## Evaluation

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_gsm8k.py --adapter ./qwen-gsm8k-final --limit 100
python scripts/evaluate_gsm8k.py --adapter ./qwen-gsm8k-final
python scripts/evaluate_gsm8k.py --base-only
```

The evaluator uses the same deterministic generation settings for both conditions.

For a publishable comparison, report model revision, adapter checkpoint, dataset split, decoding settings, task count and git SHA.