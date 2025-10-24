# Chain-of-Thought Hallucination Detection via Activation Dynamics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/hallucination-cot-detection/blob/main/notebooks/00_smoke_test.ipynb)

Proactive hallucination detection in multi-step reasoning chains using layer-wise activation dynamics.

## 🎯 Quick Start (Google Colab)

1. Click badge above or open `notebooks/00_smoke_test.ipynb`
2. Enable GPU (Runtime → T4 GPU)
3. Set OpenAI API key
4. Run all cells (~2 hours)
5. Check results!

## 📊 Expected Results
```
Baseline (logprob):       AUC = 0.67
Your model (activations): AUC = 0.76
Δ (improvement):          +0.09  ✅ PASS
```

## 🔧 Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/hallucination-cot-detection.git
cd hallucination-cot-detection
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python scripts/run_smoke_test.py
```

## 📂 Structure

- `src/` - Core modules (generator, verifier, features)
- `notebooks/` - Jupyter notebooks (start with 00_smoke_test.ipynb)
- `scripts/` - CLI scripts
- `configs/` - YAML configurations

## 📚 Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA
- 12GB+ GPU VRAM
- OpenAI API key
