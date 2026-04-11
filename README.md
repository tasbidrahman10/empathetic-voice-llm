# Empathetic Full-Duplex Speech Language Model (EFSM)

**CSE465 — Pattern Recognition and Neural Networks**
Tasbid Al Rahman | ID: 2232225642

Fine-tuning Qwen2.5-Omni-7B for empathetic therapeutic voice conversation in full-duplex mode.

---

## Research Thesis

Can a unified speech-to-speech language model, fine-tuned on EmpatheticDialogues, produce measurably more empathetic responses (evaluated via the EPITOME framework) compared to the same base model without fine-tuning, while maintaining real-time full-duplex conversational capability?

---

## Architecture

| Component | Role | Trained/Frozen |
|-----------|------|----------------|
| Audio Encoder (Whisper-large-v3 encoder) | Raw audio → audio tokens | Frozen |
| Thinker (Qwen2.5-7B backbone) | Audio tokens → empathetic text response | **Fine-tuned (QLoRA)** |
| Talker (dual-track speech decoder) | Text tokens → audio tokens, streaming | Frozen |
| Token2Wav | Audio tokens → waveform | Frozen |

---

## Project Structure

```
empathetic-voice-llm/
├── configs/config.yaml          # All hyperparameters
├── src/
│   ├── data/
│   │   ├── preprocess_empathetic.py   # EmpatheticDialogues → JSONL
│   │   └── dataset.py                 # EFSMDataset with label masking
│   ├── models/
│   │   └── qlora_setup.py             # 4-bit loading + LoRA injection
│   ├── training/
│   │   └── train.py                   # Training loop with W&B logging
│   └── eval/
│       ├── epitome_scorer.py          # EPITOME empathy scorer (Claude API)
│       └── evaluate.py               # Full base vs fine-tuned evaluation
├── notebooks/
│   ├── 00_verify_model.ipynb          # Phase 1: verify Qwen2.5-Omni loads
│   ├── 01_preprocess.ipynb            # Phase 2: process EmpatheticDialogues
│   ├── 02_training.ipynb              # Phase 3: QLoRA training launcher
│   └── 03_evaluate.ipynb             # Phase 4: evaluation + W&B logging
├── demo/
│   └── app.py                         # Gradio full-duplex demo
└── requirements.txt
```

---

## Setup

### Local (VS Code)

```bash
git clone https://github.com/tasbidrahman10/empathetic-voice-llm.git
cd empathetic-voice-llm
pip install -r requirements.txt
```

### Kaggle

Every notebook starts with:

```python
from kaggle_secrets import UserSecretsClient
import os
secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = secrets.get_secret('HF_TOKEN')
os.environ['WANDB_API_KEY'] = secrets.get_secret('WANDB_API_KEY')
```

Then:

```bash
!git clone https://github.com/tasbidrahman10/empathetic-voice-llm.git
%cd empathetic-voice-llm
!pip install -r requirements.txt -q
```

Required Kaggle Secrets: `HF_TOKEN`, `WANDB_API_KEY`, `ANTHROPIC_API_KEY` (Phase 4 only).

---

## Running Each Phase

| Phase | Notebook | GPU needed | Approx. time |
|-------|----------|------------|--------------|
| 0 — Setup | (local, no notebook) | No | — |
| 1 — Model Verification | `notebooks/00_verify_model.ipynb` | T4 x2 | ~10 min |
| 2 — Dataset Prep | `notebooks/01_preprocess.ipynb` | CPU ok | ~20 min |
| 3 — QLoRA Training | `notebooks/02_training.ipynb` | T4 x2 | 3–5 hrs/epoch |
| 4 — Evaluation | `notebooks/03_evaluate.ipynb` | T4 x2 | 1–2 hrs |
| 5 — Demo | `python demo/app.py` | GPU recommended | — |

---

## Key Hyperparameters

See [configs/config.yaml](configs/config.yaml) for full settings. Key values:

- Base model: `Qwen/Qwen2.5-Omni-7B-Instruct`
- Quantisation: 4-bit NF4 (BitsAndBytes)
- LoRA rank: 16, alpha: 32
- Learning rate: 2×10⁻⁴ (cosine schedule, 100-step warmup)
- Effective batch size: 16 (batch=1 × grad_accum=16)
- Epochs: 3

---

## External Services

| Service | Purpose | Setup |
|---------|---------|-------|
| HuggingFace Hub | Checkpoint storage (`efsm-checkpoints`) | huggingface.co/settings/tokens |
| Weights & Biases | Training metrics (`efsm-cse465`) | wandb.ai |
| Kaggle | GPU compute (T4 x2) | kaggle.com |
| Anthropic API | EPITOME scoring in Phase 4 | console.anthropic.com |

---

## Evaluation Metrics

- **EPITOME** (Emotional Reaction, Interpretation, Exploration): 0–6 scale, LLM-graded
- **WER**: Word Error Rate on IEMOCAP/RAVDESS subset
- **Human evaluation**: 5-point Likert scale, 5–8 raters, 20 audio prompts
