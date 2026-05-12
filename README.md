# Empathetic Full-Duplex Speech Language Model (EFSM)

**CSE465 - Pattern Recognition and Neural Networks**  
**Student:** Tasbid Al Rahman  
**ID:** 2232225642

EFSM is a neural network course project on empathetic speech conversation. The original research goal was to adapt a unified speech-to-speech model, Qwen2.5-Omni, for therapeutic empathetic dialogue. During implementation, full Omni fine-tuning was not feasible on free T4 GPU environments, so the final trained artifact fine-tunes the Qwen2.5 language "Thinker" equivalent:

```text
Qwen/Qwen2.5-7B-Instruct + QLoRA adapter trained on EmpatheticDialogues
```

The final demo provides a practical speech-in/speech-out system:

```text
microphone audio -> Whisper ASR -> fine-tuned Qwen2.5-7B + LoRA -> TTS voice reply
```

The Gradio UI keeps session memory and supports app-level interruption, so the user can stop the current assistant turn and immediately speak again.

---

## Final Demo Checkpoint

| Item | Value |
|---|---|
| Base LLM | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuned adapter repo | `tasbid001/efsm-checkpoints-fixed` |
| Default adapter subfolder | `checkpoint-2667` |
| ASR model | `openai/whisper-tiny.en` by default in `demo/app.py` |
| TTS | `edge-tts` by default, with `gTTS`, `espeak`, or `pyttsx3` fallback |
| Demo UI | Gradio |

If the Hugging Face adapter repository is private, a Hugging Face token with read access is required. Do not commit the token into GitHub. Set it as an environment variable named `HF_TOKEN`.

---

## Repository Structure

```text
empathetic-voice-llm/
├── configs/
│   └── config.yaml
├── demo/
│   └── app.py                         # Final Gradio speech demo
├── notebooks/
│   ├── 00_verify_model.ipynb
│   ├── 01_preprocess.ipynb
│   ├── 02_training.ipynb
│   ├── 02_training_colab.ipynb
│   ├── 03_quick_evaluate.ipynb
│   ├── 03_quick_evaluate_colab.ipynb
│   ├── 04_kaggle_full_demo.ipynb       # Kaggle demo launcher
│   └── 04_local_full_demo.ipynb        # Local PC demo launcher
├── src/
│   ├── data/
│   │   ├── preprocess_empathetic.py
│   │   ├── dataset.py
│   │   └── validate_jsonl.py
│   ├── eval/
│   │   └── quick_compare.py
│   ├── models/
│   │   └── qlora_setup.py
│   └── training/
│       └── train.py
├── phase1_report.txt
├── phase2_report.txt
├── phase_3report.txt
├── EFSM_Project_Plan.md
├── requirements.txt
└── README.md
```

---

## Hardware Requirements

For the real demo:

- NVIDIA GPU strongly recommended
- About 12-16 GB VRAM for 4-bit Qwen2.5-7B inference
- CUDA-enabled PyTorch installed in the environment
- Microphone and speakers
- Internet access for first-time model/checkpoint download

For UI-only testing:

- No GPU required
- Use mock mode:

```bash
python demo/app.py --mock
```

Mock mode proves that the Gradio interface works, but it does not load the trained model.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/tasbidrahman10/empathetic-voice-llm.git
cd empathetic-voice-llm
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Important: `requirements.txt` intentionally does not install `torch`, because Kaggle/Colab usually provide CUDA-enabled PyTorch already. For local GPU use, install the correct CUDA PyTorch build from the official PyTorch instructions for the target machine.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Hugging Face Token

If the adapter repo is private, set `HF_TOKEN` before running the real model.

Windows PowerShell:

```powershell
$env:HF_TOKEN="paste_your_huggingface_token_here"
```

Linux/macOS:

```bash
export HF_TOKEN="paste_your_huggingface_token_here"
```

In Kaggle, add it as a notebook secret named:

```text
HF_TOKEN
```

Security note: never write a real token into `README.md`, notebooks, code files, or Git commits. If a token is accidentally shared, revoke it from Hugging Face settings and create a new one.

---

## Run the Final Demo Locally

First test the UI without loading the model:

```bash
python demo/app.py --mock
```

Open:

```text
http://127.0.0.1:7860
```

Then run the real demo:

```bash
python demo/app.py
```

Useful options:

```bash
python demo/app.py --share
python demo/app.py --adapter-subfolder checkpoint-2667
python demo/app.py --max-new-tokens 100
python demo/app.py --tts-provider edge
python demo/app.py --tts-provider gtts
```

If the current checkpoint is too heavy or an earlier checkpoint is requested:

```bash
python demo/app.py --adapter-subfolder checkpoint-1800 --max-new-tokens 100
```

---

## Run with Local Notebook

Use this notebook if the evaluator wants to run the demo from a local Jupyter environment:

```text
notebooks/04_local_full_demo.ipynb
```

The notebook includes:

- dependency installation
- `HF_TOKEN` setup via secure prompt
- CUDA/GPU check
- mock UI launch
- real Gradio demo launch

---

## Run on Kaggle

Use:

```text
notebooks/04_kaggle_full_demo.ipynb
```

Kaggle settings:

1. Turn Internet on.
2. Select GPU accelerator, preferably T4 x2.
3. Add a Kaggle secret named `HF_TOKEN` if the adapter repo is private.
4. Run the cells in order.
5. For the real demo cell, open the public Gradio URL printed by Kaggle.

The Kaggle notebook launches:

```bash
python demo/app.py --share
```

---

## Quick Evaluation

To compare base vs fine-tuned text responses without loading two separate 7B models, use:

```text
notebooks/03_quick_evaluate_colab.ipynb
```

or run directly on Windows PowerShell:

```powershell
python src/eval/quick_compare.py `
  --config configs/config.yaml `
  --output results/quick_eval_results.csv `
  --limit 6 `
  --max-new-tokens 80 `
  --device-strategy single_gpu_4bit `
  --compare-mode single_peft `
  --system-prompt-style therapeutic
```

On Linux/macOS, replace PowerShell backticks with `\`.

This loads one PEFT model and generates:

- base response with LoRA adapter disabled
- fine-tuned response with LoRA adapter enabled

The output is saved to:

```text
results/quick_eval_results.csv
```

---

## Training Summary

The final training target was the Qwen2.5-7B Thinker-equivalent rather than full Qwen2.5-Omni.

| Setting | Value |
|---|---|
| Dataset | `facebook/empathetic_dialogues` |
| Text field used after bug fix | `utterance` |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Max sequence length | 512 |
| Final checkpoint | `checkpoint-2667` |

The preprocessing bug from the first long run was fixed before final training. The corrected preprocessing uses actual dialogue utterances and validates that assistant turns are not simply copied from user turns.

---

## Reproduce Data Preparation

```bash
python src/data/preprocess_empathetic.py --config configs/config.yaml
python src/data/validate_jsonl.py data/train.jsonl data/val.jsonl data/test.jsonl
```

This creates:

```text
data/train.jsonl
data/val.jsonl
data/test.jsonl
```

The `data/` directory is not meant to be committed.

---

## Reproduce Training

Training was designed for a free single-T4 style environment and uploads checkpoints to Hugging Face when `HF_TOKEN` is available.

```bash
python src/training/train.py --config configs/config.yaml
```

To resume:

```bash
python src/training/train.py --config configs/config.yaml --resume_from_checkpoint checkpoints/checkpoint-889
```

Recommended notebook:

```text
notebooks/02_training.ipynb
```

---

## Project Reports

The phase reports document the implementation path:

- `phase1_report.txt` - model verification
- `phase2_report.txt` - dataset preparation and early QLoRA training issues
- `phase_3report.txt` - fixed-data training, evaluation debugging, and final state
- `EFSM_Project_Plan.md` - revised research plan

---

## Notes for Evaluators

The submitted demo is the practical working version of the EFSM idea. The original plan targeted direct Qwen2.5-Omni speech-to-speech fine-tuning, but free GPU constraints required training the Qwen2.5-7B language backbone instead. The final system still demonstrates the core project objective: an empathetic, fine-tuned neural conversation model accessible through a speech interface.
