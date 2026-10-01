# Results

No new full held-out GSM8K accuracy has been executed in this branch.

## Historical training result

- 7,473 training examples
- 1,319 held-out examples
- 468 steps
- 1 epoch
- 5e-6 learning rate
- 14,966,784 trainable parameters
- 1:19:27 training wall time

The notebook reports decreasing validation loss during training, but it does not compute held-out GSM8K exact-match accuracy from generated answers.

## Required comparison

| Model | Split | Primary metric |
|---|---|---|
| Base Qwen2.5-3B-Instruct | GSM8K test | exact-match accuracy |
| LoRA adapter | same GSM8K test | exact-match accuracy |

The 10-question qualitative sample must not be reported as test-set accuracy.