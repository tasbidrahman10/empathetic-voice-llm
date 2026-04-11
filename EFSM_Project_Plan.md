CSE465 --- Pattern Recognition and Neural Networks

**Empathetic Full-Duplex Speech Language Model**

EFSM --- Revised Project Plan

Tasbid Al Rahman \| ID: 2232225642 \| April 2026

*Fine-tuning Qwen2.5-Omni-7B for therapeutic empathetic voice
conversation*

**1. Project Overview**

The Emotion-Aware Voice Therapist (EAVT) project has been revised
following supervisor feedback. The original design --- a six-component
pipeline connecting Whisper, WavLM, RoBERTa, a cross-attention fusion
module, LLaMA, and StyleTTS2 --- has been replaced with a fundamentally
different architectural paradigm: one unified end-to-end
speech-to-speech language model.

The new project, titled the Empathetic Full-Duplex Speech Language Model
(EFSM), fine-tunes Qwen2.5-Omni-7B-Instruct --- a state-of-the-art
unified multimodal model from Alibaba\'s Qwen team --- to engage in
empathetic therapeutic conversation in full-duplex mode. The user
speaks; the model listens and responds in speech, with both processes
capable of running simultaneously.

This approach addresses the central weakness of pipeline architectures:
text-as-intermediate strips paralinguistic information (emotion, tone,
hesitation) from the signal before it reaches the language model. In
EFSM, audio flows directly into a unified model that understands speech
holistically, responds with empathy, and synthesises emotionally
appropriate speech output --- all in a single forward pass through one
set of model weights.

**2. Research Thesis**

Can a unified speech-to-speech language model, fine-tuned specifically
on empathetic therapeutic dialogue data, produce measurably more
empathetic responses --- as evaluated by the EPITOME framework and human
raters --- compared to the same base model without empathetic
fine-tuning, while maintaining real-time full-duplex conversational
capability?

Two sub-questions drive the evaluation:

-   Does fine-tuning on EmpatheticDialogues improve EPITOME empathy
    scores (Emotional Reaction, Interpretation, Exploration) compared to
    the base Qwen2.5-Omni model?

-   Does the fine-tuned model maintain acceptable speech quality (WER,
    naturalness) relative to the base model?

**3. Architecture --- Why One Unified Model**

**3.1 The Pipeline Problem**

The original EAVT design worked as follows:

> Audio → Whisper (ASR) → WavLM (SER) → RoBERTa (lexical) → Fusion →
> LLaMA (LLM) → StyleTTS2 (TTS) → Audio

Each arrow is a modality boundary where information is lost. Whisper\'s
ASR output is a text string --- all prosodic, tonal, and paralinguistic
features are discarded. The LLaMA LLM never hears the user\'s voice; it
reads a transcript. The fusion module tries to partially recover
emotion, but the damage is done. Furthermore, the cumulative latency
across six sequential models is several seconds --- fundamentally
incompatible with natural conversation.

**3.2 The Unified Model Approach**

EFSM uses one model --- Qwen2.5-Omni-7B --- whose internal architecture
already solves these problems:

  --------------------- ------------------------------ -------------------
  **Component**         **What it Does**               **Trained/Frozen in
                                                       EFSM**

  Audio Encoder         Whisper-large-v3 encoder only. Frozen
                        Converts raw audio waveform to 
                        a sequence of high-dimensional 
                        audio tokens. Crucially, this  
                        is the encoder only --- the    
                        Whisper decoder that produces  
                        text is removed.               

  Thinker               Qwen2.5-7B language model      Fine-tuned with
                        backbone. Receives audio       QLoRA
                        tokens directly (no ASR text   
                        intermediate) and generates    
                        empathetic response text       
                        tokens. This is where the LLM  
                        \'thinks\'.                    

  Talker                Dual-track autoregressive      Frozen
                        speech decoder. Takes          
                        Thinker\'s hidden              
                        representations and text       
                        tokens as input and produces   
                        audio tokens in streaming      
                        fashion. The model starts      
                        speaking before finishing      
                        generating.                    

  Token2Wav             Converts discrete audio tokens Frozen
                        back into a waveform via a     
                        diffusion transformer decoder  
                        and BigVGAN vocoder.           
  --------------------- ------------------------------ -------------------

The entire path from user audio in to model audio out is one forward
pass. There is no text bottleneck. Emotion in the user\'s voice reaches
the Thinker as audio tokens directly.

**3.3 Full-Duplex Behaviour**

Full-duplex means the model does not wait for the user to finish
speaking before beginning to generate. In EFSM this is implemented in
two ways:

-   Streaming inference: The Talker generates and plays audio tokens as
    the Thinker produces them --- output begins before generation is
    complete.

-   Interruption detection: A Voice Activity Detector (silero-VAD) runs
    continuously. When the user begins speaking during the model\'s
    output, the generation loop is halted and the new input is
    processed. This is demonstrated live in the Gradio demo.

**4. Base Model --- Qwen2.5-Omni-7B**

Qwen2.5-Omni-7B-Instruct is an open-source end-to-end multimodal model
released by Alibaba\'s Qwen team (March 2025). It is freely available on
HuggingFace without gated access.

  -------------------------- --------------------------------------------
  **Property**               **Detail**

  HuggingFace Model ID       Qwen/Qwen2.5-Omni-7B-Instruct

  Total parameters           \~9--10 billion (7B Thinker + encoders +
                             Talker)

  Audio encoder              Modified Whisper-large-v3 encoder (encoder
                             only, no ASR decoder)

  Thinker backbone           Qwen2.5-7B language model

  Talker                     Dual-track autoregressive speech decoder
                             with streaming

  Speech output quality      Outperforms most streaming and non-streaming
                             TTS alternatives on naturalness benchmarks

  Input modalities           Text, audio, image, video

  Output modalities          Text and streaming speech

  License                    Qwen License (open source, research use
                             permitted)

  Estimated VRAM (4-bit      \~8--10 GB
  inference)                 

  Estimated VRAM (QLoRA      \~12--14 GB on T4 with batch_size=1
  training)                  
  -------------------------- --------------------------------------------

**5. Fine-tuning Strategy**

**5.1 What We Fine-tune and Why**

Only the Thinker (the Qwen2.5-7B language model backbone) is fine-tuned.
The audio encoder and Talker are frozen. This decision is made for three
reasons:

-   The audio encoder\'s representations of speech are already excellent
    and do not need updating for empathy.

-   The Talker\'s speech synthesis quality is already state-of-the-art
    and would degrade with further training on limited data.

-   Fine-tuning only the Thinker dramatically reduces memory
    requirements and makes training feasible on Kaggle T4 GPUs.

**5.2 Parameter-Efficient Fine-tuning --- QLoRA**

Quantised Low-Rank Adaptation (QLoRA) is applied to the Thinker. The
Thinker is loaded in 4-bit NF4 quantisation. LoRA adapters are injected
into all attention projection layers and trained in bf16. This results
in approximately 0.5% of the Thinker\'s parameters being trainable.

  ------------------------ ------------------ ----------------------------
  **QLoRA Hyperparameter** **Value**          **Rationale**

  Quantisation             4-bit NF4          Reduces Thinker memory from
                           (BitsAndBytes)     \~14GB to \~3.5GB

  LoRA rank (r)            16                 Higher than Phase 1\'s r=8
                                              --- Thinker is larger and
                                              task is more complex

  LoRA alpha (α)           32                 Standard α = 2r rule

  LoRA target modules      q_proj, v_proj,    All attention projections in
                           k_proj, o_proj     Thinker layers

  LoRA dropout             0.05               Light regularisation

  Compute dtype            bfloat16           Numerical stability on
                                              T4/A100

  Optimizer                AdamW 8-bit        Memory-efficient optimizer
                           (paged)            from bitsandbytes

  Learning rate            2 × 10⁻⁴           Standard for LoRA
                                              fine-tuning

  LR scheduler             Cosine with warmup Stable convergence
                           (100 steps)        

  Batch size               1                  Maximum feasible on T4 16GB

  Gradient accumulation    16                 Effective batch size = 16

  Epochs                   3                  EmpatheticDialogues is 25k
                                              samples --- 3 epochs is
                                              sufficient

  Max sequence length      2048 tokens        Covers multi-turn
                                              conversation context

  Mixed precision          bf16               Required for QLoRA

  Gradient clipping        max norm = 1.0     Training stability
  ------------------------ ------------------ ----------------------------

**5.3 Training Data Format**

Each training example is a full conversation from EmpatheticDialogues
formatted as a ChatML message list:

> \[System\]: \"You are an empathetic voice therapist\...\"
>
> \[User (emotion: sadness)\]: \"I just found out I didn\'t get the
> job\...\"
>
> \[Assistant\]: \"I\'m so sorry to hear that. That must feel really
> discouraging\...\"

Loss is computed only on assistant turn tokens. User and system tokens
are masked with -100 so they do not contribute to the gradient.

**6. Datasets**

  --------------------- --------------------------- --------------- -------------------------------
  **Dataset**           **Purpose**                 **Size**        **Access**

  EmpatheticDialogues   Primary fine-tuning dataset 25,000          Free --- HuggingFace:
                        for Thinker. 25k            conversations   facebook/empathetic_dialogues
                        conversations across 32                     
                        emotional situations. Each                  
                        conversation has a context                  
                        description, an emotional                   
                        user utterance, and an                      
                        empathetic response.                        

  IEMOCAP               Evaluation only. Real       12,000          Free --- Requires USC
                        emotional speech utterances utterances      registration (already done in
                        used to test the fine-tuned                 Phase 1)
                        model on natural spoken                     
                        inputs not seen during                      
                        training.                                   

  RAVDESS               Supplementary evaluation.   1,248           Already downloaded in Phase 1
                        Acted emotional speech for  utterances      
                        cross-dataset               (from Phase 1)  
                        generalisation check.                       
  --------------------- --------------------------- --------------- -------------------------------

Note: EmpatheticDialogues is text-only. The fine-tuning of the Thinker
uses text input/output. The model\'s speech I/O capabilities come from
the base Qwen2.5-Omni pre-training. For Phase 1 the student already
worked with RAVDESS and EMODB --- those checkpoints and insights carry
over as background context.

**7. Technology Stack**

  ---------------- --------------------- ---------------------------------
  **Category**     **Tool / Library**    **Role**

  DL Framework     PyTorch 2.x           All model operations

  Model Hub        HuggingFace           Qwen2.5-Omni loading, processor,
                   Transformers ≥ 4.52   generation

  Efficient        PEFT + BitsAndBytes   QLoRA --- 4-bit quantisation and
  Fine-tuning                            LoRA adapters

  Training Utility HuggingFace Trainer   Training loop, evaluation,
                                         checkpointing

  Audio Processing torchaudio,           Audio I/O, resampling, VAD
                   soundfile, librosa    preprocessing

  VAD              silero-vad            Voice activity detection for
                                         full-duplex interruption

  Experiment       Weights & Biases      Loss curves, EPITOME scores,
  Tracking                               ablation comparisons

  Demo Interface   Gradio 4.x            Live streaming voice interface

  Compute          Kaggle T4 GPU (15GB   Fine-tuning and inference
                   VRAM)                 

  Version Control  GitHub                Code bridge between local and
                                         Kaggle

  Checkpoint       HuggingFace Hub       Persistent checkpoint storage
  Storage                                across Kaggle sessions
  ---------------- --------------------- ---------------------------------

**8. Repository Structure (for Claude Code)**

The new repository is separate from the Phase 1 EAVT repo. Create a new
GitHub repository named empathetic-voice-llm.

> empathetic-voice-llm/
>
> ├── README.md
>
> ├── requirements.txt
>
> ├── .gitignore
>
> ├── configs/
>
> │ └── config.yaml \# All hyperparameters and paths
>
> ├── src/
>
> │ ├── data/
>
> │ │ ├── \_\_init\_\_.py
>
> │ │ ├── preprocess_empathetic.py \# Download + process
> EmpatheticDialogues
>
> │ │ └── dataset.py \# EFSMDataset --- tokenisation + label masking
>
> │ ├── models/
>
> │ │ ├── \_\_init\_\_.py
>
> │ │ └── qlora_setup.py \# QLoRA config, model loading, freezing
>
> │ ├── training/
>
> │ │ ├── \_\_init\_\_.py
>
> │ │ └── train.py \# Training loop with W&B logging
>
> │ └── eval/
>
> │ ├── \_\_init\_\_.py
>
> │ ├── epitome_scorer.py \# EPITOME automated empathy scoring
>
> │ └── evaluate.py \# Full evaluation pipeline (base vs fine-tuned)
>
> ├── notebooks/
>
> │ ├── 00_verify_model.ipynb \# Kaggle: verify Qwen2.5-Omni loads +
> infers
>
> │ ├── 01_preprocess.ipynb \# Kaggle: run EmpatheticDialogues
> preprocessing
>
> │ ├── 02_training.ipynb \# Kaggle: QLoRA training launcher
>
> │ └── 03_evaluate.ipynb \# Kaggle: run evaluation, log to W&B
>
> ├── demo/
>
> │ └── app.py \# Gradio streaming full-duplex demo
>
> └── data/ \# (gitignored) processed JSONL files
>
> ├── train.jsonl
>
> ├── val.jsonl
>
> └── test.jsonl

**9. Implementation Phases**

Each phase lists what Claude Code creates and what the student does
manually. Entries marked ⚙ CLAUDE CODE are implemented by Claude Code in
VS Code and pushed to GitHub. Entries marked ▶ YOU DO THIS are actions
taken by the student on Kaggle, HuggingFace, or locally.

**Phase 0 --- Setup and Environment (April 11--12)**

**Goal: New repository exists, dependencies defined, configs in place.**

> **⚙ CLAUDE CODE:** Create empathetic-voice-llm GitHub repository
> structure with all directories and \_\_init\_\_.py files.
>
> **⚙ CLAUDE CODE:** Write requirements.txt: torch, transformers\>=4.52,
> peft, bitsandbytes, accelerate, datasets, wandb, gradio, torchaudio,
> soundfile, librosa, silero-vad, huggingface_hub.
>
> **⚙ CLAUDE CODE:** Write configs/config.yaml with all hyperparameters:
> model_id, LoRA rank/alpha/dropout/targets, batch_size,
> gradient_accumulation, learning_rate, epochs, max_seq_len, dataset
> path, HF Hub repo name, W&B project name.
>
> **⚙ CLAUDE CODE:** Write .gitignore: data/, checkpoints/, \*.pt,
> \*.safetensors, \_\_pycache\_\_, .env, wandb/.
>
> **⚙ CLAUDE CODE:** Write README.md with project overview, setup
> instructions, and how to run each phase.
>
> **▶ YOU DO THIS:** Create a new GitHub repository named
> empathetic-voice-llm at github.com. Push the code Claude Code
> generates.
>
> **▶ YOU DO THIS:** Create a new W&B project at wandb.ai named
> efsm-cse465.
>
> **▶ YOU DO THIS:** On HuggingFace: create a new model repository named
> efsm-checkpoints (private). This is where Kaggle will upload trained
> checkpoints.

**Phase 1 --- Model Verification (April 12--13)**

**Goal: Confirm Qwen2.5-Omni-7B loads and runs inference on Kaggle T4
GPU.**

> **⚙ CLAUDE CODE:** Write notebooks/00_verify_model.ipynb. The notebook
> must: (1) clone the GitHub repo and install requirements, (2) load
> Qwen2.5-Omni-7B-Instruct in 4-bit via BitsAndBytesConfig, (3) run a
> simple text-mode chat inference (no audio needed), (4) run a simple
> speech generation to confirm Talker works, (5) print total and
> trainable parameter counts, (6) print VRAM usage before and after
> loading using torch.cuda.memory_allocated().
>
> **▶ YOU DO THIS:** Create a new Kaggle notebook. Enable GPU T4 x2. Add
> HF_TOKEN and WANDB_API_KEY as Kaggle Secrets (see Section 10 ---
> Manual Operations Guide). Run notebook cell by cell. If model loads
> and generates text + audio successfully, Phase 1 is done. Note any
> VRAM warnings and report back.

**Phase 2 --- Dataset Preparation (April 13--15)**

**Goal: EmpatheticDialogues processed into train/val/test JSONL files
ready for training.**

> **⚙ CLAUDE CODE:** Write src/data/preprocess_empathetic.py. This
> script must: (1) load the facebook/empathetic_dialogues dataset from
> HuggingFace datasets library, (2) group utterances by conversation ID
> to reconstruct multi-turn conversations, (3) format each conversation
> as a list of ChatML messages with system prompt, user turns labelled
> with emotion tag, and assistant turns, (4) apply the Qwen2.5-Omni
> processor\'s apply_chat_template to get the final text string, (5)
> split 80/10/10 into train/val/test, stratified by emotion category,
> (6) save to data/train.jsonl, data/val.jsonl, data/test.jsonl, (7)
> print dataset statistics: total conversations, emotion distribution,
> average turns per conversation, average token length.
>
> **⚙ CLAUDE CODE:** Write the system prompt string as a constant in
> preprocess_empathetic.py: \'You are an empathetic voice therapist.
> Your role is to listen carefully to how the person feels, acknowledge
> their emotions, and respond with genuine warmth and understanding.
> Validate their feelings. Do not offer unsolicited advice. Make them
> feel truly heard.\'
>
> **⚙ CLAUDE CODE:** Write src/data/dataset.py. EFSMDataset class must:
> (1) load JSONL file, (2) tokenise each example using
> Qwen2.5OmniProcessor with truncation to max_seq_len=2048, (3) create
> labels tensor identical to input_ids but with -100 masking applied to
> all non-assistant tokens (system and user tokens are masked so loss is
> only computed on assistant responses), (4) return dict of input_ids,
> attention_mask, labels.
>
> **⚙ CLAUDE CODE:** Write notebooks/01_preprocess.ipynb: clone repo,
> install requirements, run preprocess_empathetic.py, display sample
> formatted conversations, upload processed JSONL files to HuggingFace
> Hub dataset repo as backup.
>
> **▶ YOU DO THIS:** Run 01_preprocess.ipynb on Kaggle. This does not
> need GPU --- use CPU. Verify the JSONL files look correct (spot check
> 5 conversations). The data/train.jsonl should have roughly 17,000
> rows. Upload to HuggingFace if the script does so automatically.

**Phase 3 --- QLoRA Fine-tuning (April 15--20)**

**Goal: Thinker fine-tuned on EmpatheticDialogues. Best checkpoint saved
to HuggingFace Hub.**

> **⚙ CLAUDE CODE:** Write src/models/qlora_setup.py. Function
> load_model_for_training() must: (1) create BitsAndBytesConfig with
> load_in_4bit=True, bnb_4bit_quant_type=\'nf4\',
> bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
> (2) load Qwen2_5OmniForConditionalGeneration with the BnB config and
> device_map=\'auto\', (3) freeze all audio_encoder and talker and
> token2wav parameters by iterating named_parameters and setting
> requires_grad=False for any whose name contains those strings, (4)
> create LoraConfig with r=16, lora_alpha=32,
> target_modules=\[\'q_proj\',\'v_proj\',\'k_proj\',\'o_proj\'\],
> lora_dropout=0.05, bias=\'none\', task_type=None (omit task_type to
> avoid WavLM-style NotImplementedError --- same fix as Phase 1), (5)
> apply get_peft_model(model, lora_config), (6) call
> model.print_trainable_parameters() to log counts, (7) return model and
> processor.
>
> **⚙ CLAUDE CODE:** Write src/training/train.py. This script must: (1)
> load config from configs/config.yaml, (2) call
> load_model_for_training(), (3) load EFSMDataset for train and val
> splits, (4) create HuggingFace TrainingArguments:
> output_dir=checkpoints/, per_device_train_batch_size=1,
> gradient_accumulation_steps=16, learning_rate=2e-4,
> num_train_epochs=3, warmup_steps=100, lr_scheduler_type=\'cosine\',
> fp16=False, bf16=True, logging_steps=10, eval_steps=100,
> save_steps=100, save_total_limit=3, load_best_model_at_end=True,
> metric_for_best_model=\'eval_loss\', report_to=\'wandb\', run_name
> from config, (5) create Trainer with model, training_args,
> train_dataset, eval_dataset, data_collator=DataCollatorForSeq2Seq with
> padding, (6) call trainer.train(), (7) after training, save LoRA
> adapter weights only (model.save_pretrained()) and upload to
> HuggingFace Hub using huggingface_hub.HfApi.upload_folder().
>
> **⚙ CLAUDE CODE:** Write notebooks/02_training.ipynb: cell 1 sets up
> secrets and env vars, cell 2 clones repo and installs, cell 3
> downloads preprocessed data from HuggingFace Hub, cell 4 runs python
> src/training/train.py, cell 5 confirms checkpoint upload to HF Hub.
>
> **▶ YOU DO THIS:** Before running: open wandb.ai/settings and copy
> your API key. Open huggingface.co/settings/tokens and copy your write
> token. Add both as Kaggle Secrets (detailed steps in Section 10).
>
> **▶ YOU DO THIS:** Run 02_training.ipynb on Kaggle with GPU T4 x2
> selected. Expected training time: 3--5 hours per epoch on T4. Monitor
> loss in real time at wandb.ai --- open your efsm-cse465 project. If
> the session approaches the 9-hour Kaggle limit, the checkpoint save +
> HF Hub upload in the script will preserve progress. You can resume by
> loading the latest checkpoint in the next session.
>
> **▶ YOU DO THIS:** After training completes: verify the checkpoint
> appears at huggingface.co/YOUR_USERNAME/efsm-checkpoints. Download the
> adapter_config.json and adapter_model.safetensors to verify they are
> non-empty.

**Phase 4 --- Evaluation (April 20--23)**

**Goal: Quantitative comparison of base vs fine-tuned EFSM on empathy
and speech quality metrics.**

> **⚙ CLAUDE CODE:** Write src/eval/epitome_scorer.py. The EPITOME
> scorer evaluates model responses across three dimensions: Emotional
> Reaction (ER): Does the response show emotional acknowledgement?
> Interpretation (IP): Does it show understanding of the situation?
> Exploration (EX): Does it probe or explore further? Each dimension is
> scored 0--2, summing to a max of 6. Implement an LLM-based evaluator
> that sends (context, response) pairs to the Anthropic Claude API with
> a structured rubric prompt that returns JSON scores for ER, IP, EX.
> The function signature is: score_response(context: str, response: str)
> -\> dict with keys er, ip, ex, total.
>
> **⚙ CLAUDE CODE:** Write src/eval/evaluate.py. This script runs the
> full evaluation: (1) load the test.jsonl (last 10% of
> EmpatheticDialogues), (2) for each test conversation, run inference
> with (a) base Qwen2.5-Omni-7B-Instruct and (b) EFSM (base + fine-tuned
> LoRA adapters), (3) collect text responses from both, (4) run
> epitome_scorer.score_response() on all responses, (5) compute mean
> EPITOME scores and standard deviations for both systems, (6) run a
> simple t-test (scipy.stats.ttest_ind) on total EPITOME scores, (7)
> report WER on a 50-utterance IEMOCAP/RAVDESS subset if available, (8)
> log all results as a W&B summary table and as a CSV in
> results/evaluation_results.csv.
>
> **⚙ CLAUDE CODE:** Write notebooks/03_evaluate.ipynb: runs
> evaluate.py, displays comparison table, shows sample response pairs
> (base vs fine-tuned) for 5 emotional inputs.
>
> **▶ YOU DO THIS:** Run 03_evaluate.ipynb on Kaggle. You need your
> Anthropic API key added as a Kaggle Secret named ANTHROPIC_API_KEY for
> the EPITOME scorer. Get this from console.anthropic.com. Evaluation on
> 500 test samples takes approximately 1--2 hours. The W&B table will
> auto-populate at wandb.ai.
>
> **▶ YOU DO THIS:** For human evaluation: prepare 20 audio prompts from
> RAVDESS/IEMOCAP covering all 7 emotion classes. For each prompt,
> collect the base and fine-tuned model\'s text responses. Print these
> response pairs anonymised (A vs B, no labels). Ask 5--8 classmates or
> family members to rate each response pair on: (Q1) \'Did this response
> acknowledge how you were feeling? (1--5)\', (Q2) \'Did this response
> feel understanding of your situation? (1--5)\', (Q3) \'Did the
> response feel warm and caring? (1--5)\'. Compute mean Likert scores
> for A and B.

**Phase 5 --- Full-Duplex Demo (April 23--25)**

**Goal: Live Gradio demo showing speech-to-speech empathetic
conversation with interruption handling.**

> **⚙ CLAUDE CODE:** Write demo/app.py. The Gradio application must: (1)
> load Qwen2.5-Omni-7B-Instruct in 4-bit and merge LoRA adapters from
> HuggingFace Hub on startup, (2) use Gradio gr.Audio with
> source=\'microphone\' and streaming=True to capture user voice, (3)
> run silero-vad on incoming audio chunks (80ms windows) to detect
> speech end, (4) when VAD detects end of speech, send audio to the
> model for streaming generation --- use Qwen2.5-Omni\'s streaming
> inference mode so audio output begins before generation completes, (5)
> during output streaming, continue running VAD on the microphone input
> --- if new speech is detected, stop the output generator
> (interruption), (6) display a live text transcript of both user speech
> and model response in a gr.Chatbot component, (7) play the generated
> audio response in a gr.Audio output component. Include a \'Reset
> Conversation\' button that clears history.
>
> **▶ YOU DO THIS:** Run the demo locally using: python demo/app.py.
> This requires a microphone and speakers. If running on Kaggle, use
> gr.Interface.launch(share=True) which creates a public URL --- use
> this to do a live demo for your faculty. Record a short video of the
> demo for submission.

**10. Manual Operations Guide**

This section documents every action the student must take directly ---
steps that cannot be done by Claude Code because they involve web
interfaces, credentials, or interactive GPU sessions.

**10.1 GitHub Repository**

1.  Go to github.com → click \'+\' → New repository.

2.  Name: empathetic-voice-llm. Set to Private initially.

3.  Do NOT initialise with README (Claude Code will create it).

4.  Copy the repo URL. In your VS Code terminal: git remote add origin
    \<URL\>.

5.  After Claude Code generates Phase 0 files: git add . && git commit
    -m \'Phase 0: project setup\' && git push.

6.  Before each Kaggle training run: always push the latest code from VS
    Code first.

**10.2 HuggingFace Setup**

7.  Go to huggingface.co → sign in → click your avatar → Settings →
    Access Tokens.

8.  Click \'New token\' → Name: kaggle-efsm → Role: Write → Generate →
    copy the token.

9.  Store this token securely. You will add it to Kaggle Secrets ---
    never put it in code.

10. Create checkpoint repo: huggingface.co/new → name: efsm-checkpoints
    → Private → Create.

11. The model Qwen/Qwen2.5-Omni-7B-Instruct does NOT require gated
    access --- you can download it without approval.

**10.3 Weights & Biases Setup**

12. Go to wandb.ai → sign in or create free account.

13. Click \'+ New Project\' → Name: efsm-cse465 → Private.

14. Click your avatar → User Settings → API Keys → copy the key.

15. You will add this key to Kaggle Secrets as WANDB_API_KEY.

16. During training, open wandb.ai/YOUR_USERNAME/efsm-cse465 in a
    browser tab to monitor live loss curves.

**10.4 Kaggle Notebook Setup --- Step by Step**

17. Go to kaggle.com → Code → click \'New Notebook\'.

18. Name the notebook (e.g. \'EFSM-00-Verify\', \'EFSM-02-Training\').

19. In the right panel: click \'Session options\' → Accelerator → select
    \'GPU T4 x2\'. Click Save.

20. Add secrets: right panel → \'Add-ons\' → \'Secrets\' → \'Add a new
    secret\'.

-   Name: HF_TOKEN --- Value: your HuggingFace write token

-   Name: WANDB_API_KEY --- Value: your W&B API key

-   Name: ANTHROPIC_API_KEY --- Value: your Anthropic API key (for
    evaluation phase only)

21. In the first cell of EVERY Kaggle notebook, always add this block:

> from kaggle_secrets import UserSecretsClient
>
> import os
>
> secrets = UserSecretsClient()
>
> os.environ\[\'HF_TOKEN\'\] = secrets.get_secret(\'HF_TOKEN\')
>
> os.environ\[\'WANDB_API_KEY\'\] =
> secrets.get_secret(\'WANDB_API_KEY\')

22. In the second cell, clone the repo and install:

> !git clone https://github.com/YOUR_USERNAME/empathetic-voice-llm.git
>
> %cd empathetic-voice-llm
>
> !pip install -r requirements.txt -q

23. After this, run the notebook-specific cells as defined in each
    phase\'s notebook file.

24. Important: Kaggle notebooks time out after 9 hours. The training
    script saves a checkpoint after every 100 steps and immediately
    uploads to HuggingFace Hub. If the session ends, you can resume from
    the last checkpoint in a new session.

**10.5 Running Training --- What to Watch For**

-   W&B: open wandb.ai/efsm-cse465 in a browser. You should see
    train/loss and eval/loss charts appear within the first 2--3 minutes
    of training. If nothing appears after 5 minutes, the WANDB_API_KEY
    secret may be wrong.

-   VRAM: the first cell of 02_training.ipynb prints VRAM usage. If you
    see CUDA out of memory, reduce gradient_accumulation_steps in
    configs/config.yaml from 16 to 8 and restart the kernel.

-   Loss range: expect initial loss around 2.0--2.5 (Thinker on new
    domain). By the end of epoch 1 it should drop to 1.5--1.8. If loss
    does not decrease at all after 200 steps, something is wrong with
    label masking (report back).

-   Checkpoint: after each 100 training steps the script should print
    \'Checkpoint uploaded to HuggingFace Hub\'. Verify this appears at
    least once in the first hour.

**10.6 Checkpoint Management**

-   After training: the LoRA adapter files (adapter_config.json,
    adapter_model.safetensors) at
    huggingface.co/YOUR_USERNAME/efsm-checkpoints are your trained
    model.

-   These files are small (\~130MB for r=16). Download them locally as a
    backup.

-   To load the fine-tuned model anywhere: load the base
    Qwen2.5-Omni-7B-Instruct and merge the LoRA adapters using
    PeftModel.from_pretrained(base_model,
    \'YOUR_USERNAME/efsm-checkpoints\').

**10.7 Human Evaluation Protocol**

25. Select 20 audio clips from RAVDESS covering all 7 emotion classes
    (neutral, calm, happy, sad, angry, fearful, disgust) --- 2--3 clips
    per emotion.

26. For each clip, generate text responses from both: (A) base
    Qwen2.5-Omni-7B-Instruct and (B) EFSM. Print them side by side in a
    Google Form or paper form without labelling which is A or B.

27. Ask 5--8 evaluators to rate each response on three 1--5 Likert
    questions per response.

28. Calculate mean and standard deviation per question for A vs B.
    Report these in the final report.

29. Target: EFSM (B) should score ≥ 0.5 higher on average than base (A).

**11. Evaluation Plan**

**11.1 Systems Under Comparison**

-   System A --- Baseline: Qwen2.5-Omni-7B-Instruct with no fine-tuning
    (base model as released).

-   System B --- EFSM: Qwen2.5-Omni-7B-Instruct with Thinker fine-tuned
    via QLoRA on EmpatheticDialogues.

**11.2 Automated Metrics**

  --------------- ---------------------------- ------------ ----------------
  **Metric**      **Description**              **System**   **Target**

  EPITOME --- ER  Emotional Reaction score     A vs B       B \> A by ≥ 0.3
                  (0--2): does the response                 
                  show emotional                            
                  acknowledgement?                          

  EPITOME --- IP  Interpretation score (0--2): A vs B       B \> A by ≥ 0.3
                  does the response show                    
                  situational understanding?                

  EPITOME --- EX  Exploration score (0--2):    A vs B       B \> A by ≥ 0.2
                  does the response probe                   
                  further?                                  

  EPITOME ---     Sum of ER + IP + EX (0--6)   A vs B       B \> 4.0
  Total                                                     

  Eval Loss       Cross-entropy on             B only       \< 1.5 after 3
                  EmpatheticDialogues test set              epochs

  WER             Word Error Rate on Talker    A vs B       \< 10% for both
                  speech output (reference =                (speech quality
                  transcription)                            maintained)

  Latency         Time from end of user speech B            \< 2 seconds
                  to first audio output token               
  --------------- ---------------------------- ------------ ----------------

**11.3 Human Evaluation**

  ------------------------------------ --------------- -------------------
  **Question**                         **Scale**       **Target**

  Q1: Did the response acknowledge how 1--5 Likert     EFSM mean \> 3.5
  you were feeling?                                    

  Q2: Did the response feel            1--5 Likert     EFSM mean \> 3.5
  understanding of your situation?                     

  Q3: Did the response feel warm and   1--5 Likert     EFSM mean \> 3.5
  caring?                                              
  ------------------------------------ --------------- -------------------

20 audio test prompts × 8 evaluators = 160 ratings per question per
system. Compute inter-rater reliability using Krippendorff\'s alpha.

**11.4 Ablation**

The ablation is simple and clean: base Qwen2.5-Omni vs EFSM
(fine-tuned). This directly answers the research thesis. If time
permits, a second ablation can compare: Thinker fine-tuning only
(current approach) vs. full fine-tuning of all unfrozen components.

**12. Project Timeline**

  ------------ --------------------- -------------------------------------
  **Date**     **Phase**             **Deliverable**

  Apr 11--12   Phase 0: Setup        New repo, requirements, config,
                                     .gitignore live on GitHub

  Apr 12--13   Phase 1: Verification Qwen2.5-Omni loads on Kaggle T4,
                                     sample inference confirmed

  Apr 13--15   Phase 2: Dataset      train/val/test.jsonl files on Kaggle
                                     and HuggingFace

  Apr 15--20   Phase 3: Fine-tuning  Best checkpoint at efsm-checkpoints
                                     on HuggingFace, W&B loss curves
                                     logged

  Apr 20--23   Phase 4: Evaluation   EPITOME scores (A vs B) and human
                                     evaluation data collected

  Apr 23--25   Phase 5: Demo         Gradio app running, demo video
                                     recorded for faculty

  Apr 25--30   Report Writing        Final project report submitted
  ------------ --------------------- -------------------------------------

**13. Novelty and Contributions**

30. Empathetic fine-tuning of a unified speech-to-speech LLM: Unlike
    existing work on empathetic text chatbots, EFSM operates entirely in
    the speech modality with no text bottleneck.

31. Therapeutic domain adaptation: The specific fine-tuning objective
    targets therapeutic empathy (validation, warmth, non-directive
    listening) --- a narrower and more clinically-motivated goal than
    general empathy.

32. EPITOME-guided evaluation: Applying the EPITOME framework to a
    speech-to-speech system provides a principled, multi-dimensional
    empathy measurement beyond simple human preference.

33. Full-duplex demonstration: The Gradio demo shows live
    interruption-aware speech conversation --- a capability impossible
    in the original pipeline design.

34. Quantitative ablation: The base vs. fine-tuned comparison provides
    direct empirical evidence for the value of the fine-tuning step,
    satisfying the research thesis.

**GitHub:** github.com/tasbid-rahman10/empathetic-voice-llm **· W&B
Project:** efsm-cse465 **· Checkpoint:** HuggingFace: efsm-checkpoints
