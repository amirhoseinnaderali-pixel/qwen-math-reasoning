# Research Audit — Qwen Math Reasoning

## Research question

> Does parameter-efficient supervised fine-tuning of Qwen2.5-3B improve objective mathematical problem-solving accuracy on held-out GSM8K examples?

## Historical experiment

The notebook uses Qwen2.5-3B-Instruct, 4-bit quantization, LoRA, GSM8K, 7,473 training examples and 1,319 test examples, one epoch, effective batch size 16, learning rate 5e-6, and 468 training steps.

The notebook logs 14,966,784 trainable parameters out of 3,100,905,472 total parameters, about 0.483% trainable.

The recorded training wall time is 1:19:27.

## Main methodological gap

The notebook does not calculate automated exact-match accuracy on the 1,319-example held-out split. It instead prints generations for 10 manually selected questions for the base model and the fine-tuned model.

That is useful qualitative evidence, but it is not a benchmark result.

## Controlled redesign

This branch adds a reproducible evaluator for:
- Base Qwen2.5-3B-Instruct
- the trained LoRA adapter

Both conditions use identical prompts, decoding, dataset split and answer normalization.

Primary metric: GSM8K exact-match numerical accuracy.

Secondary metrics: answer parse rate, strict #### format rate, and generation latency.

## Hypothesis

> LoRA fine-tuning on GSM8K will increase held-out exact-match accuracy relative to the frozen base model under the same inference protocol.

This is falsifiable.

Validation loss alone is not sufficient evidence of improved mathematical problem-solving accuracy.