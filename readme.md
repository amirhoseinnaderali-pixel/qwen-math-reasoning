# 🧮 GSM8K Fine-tuning with Qwen2.5-3B

A complete pipeline for fine-tuning Qwen2.5-3B-Instruct on the GSM8K dataset to improve mathematical reasoning capabilities using chain-of-thought (CoT) prompting.

## 📋 Overview

This project demonstrates how to fine-tune a language model for math word problems using:
- **Model**: Qwen2.5-3B-Instruct (4-bit quantized)
- **Dataset**: GSM8K (7,473 train + 1,319 test = 8,792 total problems)
- **Technique**: LoRA (Low-Rank Adaptation)
- **Framework**: Unsloth + Hugging Face Transformers

## ✨ Features

- ✅ **Optimized Configuration**: Anti-overfitting hyperparameters
- ✅ **Chain-of-Thought**: Step-by-step reasoning format
- ✅ **Memory Efficient**: 4-bit quantization + gradient checkpointing
- ✅ **Early Stopping**: Automatic training termination
- ✅ **Comprehensive Validation**: Full error handling and data validation
- ✅ **Comparison Testing**: Evaluate both base and fine-tuned models

## 🚀 Quick Start

### 1. Setup

```python
# The notebook automatically installs dependencies
# Just run the first cell to install:
# - unsloth
# - transformers
# - trl
# - datasets
```

### 2. Mount Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Run Training

Simply execute all cells in order. The pipeline will:
1. Load and validate the GSM8K dataset
2. Format data for chain-of-thought reasoning
3. Load Qwen2.5-3B with LoRA adapters
4. Train with optimized hyperparameters
5. Save model locally and to Google Drive
6. Run evaluation tests

## 📊 Dataset Format

**GSM8K Format:**
```json
{
  "question": "Natalia sold clips to 48 of her friends...",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips... #### 72"
}
```

**Formatted Prompt:**
```
<|im_start|>system
You are an expert math tutor. Solve problems step-by-step with clear reasoning.<|im_end|>
<|im_start|>user
[QUESTION]<|im_end|>
<|im_start|>assistant
[STEP-BY-STEP REASONING]

#### [FINAL ANSWER]<|im_end|>
```

## ⚙️ Key Configuration

### Anti-Overfitting Setup

```python
# Reduced learning rate for stability
LEARNING_RATE = 5e-6  # Down from typical 2e-4

# Single epoch to prevent memorization
NUM_EPOCHS = 1

# Conservative LoRA parameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 3
```

### Training Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Batch Size | 2 per device | Memory efficiency |
| Gradient Accumulation | 8 steps | Effective batch size of 16 |
| Total Steps | ~468 steps | (7473 samples / 16 batch) × 1 epoch |
| Max Sequence Length | 1536 | Handle long reasoning chains |
| Warmup Ratio | 0.05 | ~23 steps warmup |
| Weight Decay | 0.01 | L2 regularization |
| LR Scheduler | Cosine | Smooth learning rate decay |
| Max Grad Norm | 1.0 | Gradient clipping |
| Optimizer | AdamW 8-bit | Memory-efficient optimization |

## 📈 Training Progress

Expected training metrics (468 total steps):
```
Step  50: train_loss=1.306, eval_loss=1.711
Step 100: train_loss=0.817, eval_loss=1.253
Step 150: train_loss=0.609, eval_loss=1.075
Step 200: train_loss=0.533, eval_loss=1.025
Step 250: train_loss=0.489, eval_loss=0.998
Step 300: train_loss=0.474, eval_loss=0.986
Step 350: train_loss=0.462, eval_loss=0.976
Step 400: train_loss=0.445, eval_loss=0.973
Step 450: train_loss=0.456, eval_loss=0.971
```

Training typically completes in **~75-80 minutes** on a T4 GPU (actual: 1:19:27).

## 🧪 Evaluation

The notebook includes two evaluation functions:

### 1. Fine-tuned Model Test
```python
test_model()  # Tests the fine-tuned model with LoRA
```

### 2. Base Model Comparison
```python
test_base_model_fresh()  # Loads pure base model without LoRA
```

**Sample Test Questions:**
- Multi-step arithmetic problems
- Percentage calculations
- Rate/time/distance problems
- Fraction operations

## 📁 Project Structure

```
qwen-gsm8k-math-reasoning/
├── model files (saved to Google Drive)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer files
│   └── training_info.json
│
└── local checkpoints/
    └── qwen-gsm8k-checkpoints/
        └── checkpoint-{steps}/
```

## 💾 Model Saving

Models are saved to two locations:

1. **Local**: `./qwen-gsm8k-final/`
2. **Google Drive**: `/content/drive/MyDrive/qwen-gsm8k-math-reasoning/`

## 🔬 Technical Details

### LoRA Configuration
```python
r = 8                    # Rank (parameter efficiency)
alpha = 16              # Scaling factor (alpha/r = 2)
dropout = 0.1           # Regularization
target_modules = [       # Attention + MLP layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
bias = "none"           # No bias training
use_rslora = False      # Standard LoRA
```

**Trainable Parameters**: 14,966,784 / 3,100,905,472 (0.48%)

### Memory Optimization
- 4-bit quantization (BitsAndBytes)
- Gradient checkpointing (Unsloth)
- Mixed precision training (BF16 if available)
- Gradient accumulation

### Data Processing
- Answer extraction with regex
- Validation of all samples
- ChatML formatting
- Automatic padding/truncation

## 📊 Expected Results

**Base Model (Pre-training):**
- Correctly formats responses
- Basic arithmetic capability
- May lack systematic step-by-step reasoning

**Fine-tuned Model:**
- Structured chain-of-thought reasoning
- Explicit step markers (Step 1, Step 2...)
- Clear final answer format (#### X)
- Improved accuracy on multi-step problems

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory:**
```python
# Reduce batch size
config.BATCH_SIZE = 1
config.GRADIENT_ACCUMULATION = 16
```

**Slow Training:**
```python
# Reduce max sequence length
config.MAX_SEQ_LENGTH = 1024
```

**Overfitting:**
```python
# Already optimized, but can further reduce:
config.LEARNING_RATE = 3e-6
config.LORA_DROPOUT = 0.15
```

## 📚 References

- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Add BLEU/ROUGE metrics
- [ ] Implement automatic answer extraction and accuracy calculation
- [ ] Support for other math datasets (MATH, MathQA)
- [ ] Multi-GPU training support
- [ ] Hyperparameter optimization with Optuna

## 📄 License

This project uses:
- **Qwen2.5**: Apache 2.0 License
- **GSM8K**: MIT License
- **Unsloth**: Apache 2.0 License

## 🙏 Acknowledgments

- Qwen team at Alibaba Cloud
- OpenAI for GSM8K dataset
- Unsloth team for optimization tools
- Hugging Face for infrastructure

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Check Unsloth Discord for community support

---

**Note**: This notebook is designed for Google Colab with GPU runtime. Local execution requires CUDA-compatible GPU with 12GB+ VRAM.