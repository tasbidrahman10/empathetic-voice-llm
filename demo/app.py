#!/usr/bin/env python3
"""EFSM speech demo.

Practical final demo path:
microphone audio -> Whisper ASR -> Qwen2.5-7B + EFSM LoRA -> WAV TTS reply.

The trained project artifact is a text Thinker-equivalent LoRA adapter, so this
app wraps it in a speech interface and implements interruption at the UI/event
layer. Use --mock to test the UI without loading the 7B model.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep the demo on one GPU. 4-bit PEFT inference is more stable this way on T4.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gradio as gr


DEFAULT_SYSTEM_PROMPT = """You are EFSM, an empathetic therapeutic voice conversation partner.
Respond in a warm, careful, emotionally validating way.
First acknowledge the user's feeling, then validate why it makes sense, then gently invite reflection.
Keep the reply conversational and concise, usually 3 to 5 sentences.
Do not diagnose, do not claim to be a licensed therapist, and do not give crisis instructions unless the user expresses danger.
If the user may be at immediate risk of self-harm or harm to others, encourage contacting local emergency services or a trusted person right now."""


@dataclass
class DemoConfig:
    model_id: str
    adapter_id: str
    adapter_subfolder: str | None
    asr_model_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
    tts_rate: int
    tts_voice_contains: str | None
    share: bool
    mock: bool


class EFSMEngine:
    def __init__(self, config: DemoConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._asr = None
        self._load_lock = threading.Lock()
        self._stop_event = threading.Event()

    def clear_interrupt(self) -> None:
        self._stop_event.clear()

    def interrupt(self) -> str:
        self._stop_event.set()
        return "Interrupted. You can speak again now."

    def load(self) -> str:
        if self.config.mock:
            return "Mock mode is active. UI can run without loading ASR or Qwen."

        with self._load_lock:
            if self._model is not None and self._asr is not None:
                return "Models are already loaded."

            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA GPU was not detected. The real EFSM demo needs a GPU for Qwen2.5-7B 4-bit inference. "
                    "Run with --mock for UI testing only."
                )

            self._asr = pipeline(
                "automatic-speech-recognition",
                model=self.config.asr_model_id,
                device=0,
                torch_dtype=torch.float16,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
                low_cpu_mem_usage=True,
            )
            peft_kwargs: dict[str, Any] = {"token": os.environ.get("HF_TOKEN")}
            if self.config.adapter_subfolder:
                peft_kwargs["subfolder"] = self.config.adapter_subfolder
            model = PeftModel.from_pretrained(model, self.config.adapter_id, **peft_kwargs)
            model.eval()
            model.config.use_cache = True

            self._tokenizer = tokenizer
            self._model = model
            return (
                f"Loaded ASR `{self.config.asr_model_id}` and EFSM adapter "
                f"`{self.config.adapter_id}/{self.config.adapter_subfolder or ''}`."
            )

    def transcribe(self, audio_path: str | None) -> str:
        if not audio_path:
            return ""
        if self.config.mock:
            return "I feel overwhelmed about finishing this project and presenting it well."

        self.load()
        assert self._asr is not None
        import librosa

        # Gradio public links can hand us browser-recorded audio files whose
        # container metadata confuses the Transformers ASR pipeline. Loading the
        # file ourselves gives Whisper a plain mono 16 kHz waveform instead.
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        result = self._asr({"array": waveform, "sampling_rate": sample_rate})
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return " ".join(text.strip().split())

    def respond(self, user_text: str, history: list[list[str | None]], system_prompt: str) -> str:
        if self.config.mock:
            return (
                "It makes sense that you feel overwhelmed with the deadline so close. "
                "You have already done the hardest research and training work, and now the pressure is about making it presentable. "
                "Let's keep the next step small: what part feels most urgent right now, the demo, the report, or the evaluation evidence?"
            )

        self.load()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        assert self._model is not None
        assert self._tokenizer is not None

        stop_event = self._stop_event

        class InterruptStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return stop_event.is_set()

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt.strip()}]
        for user_turn, assistant_turn in history[-4:]:
            if user_turn:
                messages.append({"role": "user", "content": user_turn})
            if assistant_turn:
                messages.append({"role": "assistant", "content": assistant_turn})
        messages.append({"role": "user", "content": user_text})

        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
        do_sample = self.config.temperature > 0
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                repetition_penalty=1.08,
                stopping_criteria=StoppingCriteriaList([InterruptStoppingCriteria()]),
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def synthesize(self, text: str) -> str | None:
        if not text:
            return None
        if self._stop_event.is_set():
            return None

        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "pyttsx3 is not installed. Install requirements, then rerun the demo."
            ) from exc

        out_path = Path(tempfile.gettempdir()) / "efsm_reply.wav"
        engine = pyttsx3.init()
        engine.setProperty("rate", self.config.tts_rate)
        if self.config.tts_voice_contains:
            needle = self.config.tts_voice_contains.lower()
            for voice in engine.getProperty("voices"):
                haystack = f"{voice.id} {voice.name}".lower()
                if needle in haystack:
                    engine.setProperty("voice", voice.id)
                    break
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        engine.stop()
        return str(out_path)


def build_demo(engine: EFSMEngine) -> gr.Blocks:
    css = """
    #status { min-height: 36px; }
    .compact textarea { font-size: 14px; }
    """

    with gr.Blocks(title="EFSM Speech Demo", css=css) as demo:
        gr.Markdown(
            "# EFSM Speech Demo\n"
            "Speak into the microphone, then EFSM transcribes, responds empathetically, and returns speech."
        )
        status = gr.Textbox(label="Status", value="Models are not loaded yet.", interactive=False, elem_id="status")

        with gr.Row():
            load_btn = gr.Button("Load Models", variant="secondary")
            interrupt_btn = gr.Button("Interrupt / Stop Current Turn", variant="stop")
            clear_btn = gr.Button("Clear Conversation")

        chatbot = gr.Chatbot(label="Conversation", height=360)
        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                microphone = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Microphone Input",
                )
                submit_audio = gr.Button("Send Voice", variant="primary")
            with gr.Column(scale=1):
                transcript = gr.Textbox(label="Transcript", lines=4, elem_classes=["compact"])
                reply_audio = gr.Audio(label="Assistant Voice Reply", type="filepath", autoplay=True)

        with gr.Accordion("Demo Controls", open=False):
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=8,
            )
            manual_text = gr.Textbox(
                label="Optional Text Input",
                placeholder="Type here if the microphone or ASR is unavailable.",
                lines=3,
            )
            submit_text = gr.Button("Send Text")

        def load_models() -> str:
            return engine.load()

        def handle_turn(
            audio_path: str | None,
            typed_text: str,
            history: list[list[str | None]],
            prompt: str,
        ):
            engine.clear_interrupt()
            user_text = typed_text.strip() if typed_text and typed_text.strip() else engine.transcribe(audio_path)
            if not user_text:
                return history, history, "", None, "No speech or text was detected."

            assistant_text = engine.respond(user_text, history, prompt)
            completed_history = history + [[user_text, assistant_text]]
            audio_out = engine.synthesize(assistant_text)
            return completed_history, completed_history, user_text, audio_out, "Turn complete."

        def clear_history():
            return [], [], "", None, "Conversation cleared."

        load_btn.click(load_models, outputs=status)
        audio_event = submit_audio.click(
            handle_turn,
            inputs=[microphone, manual_text, history_state, system_prompt],
            outputs=[chatbot, history_state, transcript, reply_audio, status],
            concurrency_limit=1,
        )
        text_event = submit_text.click(
            handle_turn,
            inputs=[microphone, manual_text, history_state, system_prompt],
            outputs=[chatbot, history_state, transcript, reply_audio, status],
            concurrency_limit=1,
        )
        interrupt_btn.click(
            engine.interrupt,
            outputs=status,
            cancels=[audio_event, text_event],
        )
        clear_btn.click(clear_history, outputs=[chatbot, history_state, transcript, reply_audio, status])

    return demo


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="Run the EFSM speech UI demo.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-id", default="tasbid001/efsm-checkpoints-fixed")
    parser.add_argument("--adapter-subfolder", default="checkpoint-2667")
    parser.add_argument("--asr-model-id", default="openai/whisper-base")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tts-rate", type=int, default=165)
    parser.add_argument("--tts-voice-contains", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    return DemoConfig(**vars(args))


def main() -> None:
    config = parse_args()
    engine = EFSMEngine(config)
    demo = build_demo(engine)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=config.share,
    )


if __name__ == "__main__":
    main()
