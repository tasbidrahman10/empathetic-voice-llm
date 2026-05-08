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
import asyncio
import html
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep the demo on one GPU. 4-bit PEFT inference is more stable this way on T4.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gradio as gr


DEFAULT_SYSTEM_PROMPT = """You are EFSM, an empathetic therapeutic voice conversation partner.
Respond like a warm conversational companion, not a formal therapist and not a generic chatbot.
Always reply with at least 3 meaningful sentences.
Make the reply specific to the user's actual situation.
If the user is sad, betrayed, anxious, ashamed, or hopeless: first validate the pain, then give a grounded logical reframe or motivation, then invite them to keep talking.
If the user is happy, proud, relieved, or excited: reflect the positive emotion, celebrate the specific win, and encourage them to enjoy or build on it.
For betrayal or breakup: be protective of the user's self-worth. You may gently say that being cheated on reflects the other person's choices, not the user's value.
Use the recent conversation history naturally. If the user is continuing a previous thought, do not reset the conversation.
Avoid robotic one-liners like "Oh no, sorry to hear that."
Avoid empty reassurance like "you will get through this" unless you explain why in a grounded way.
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
    tts_provider: str
    edge_voice: str
    edge_style: str
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

    def is_interrupted(self) -> bool:
        return self._stop_event.is_set()

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
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA GPU was not detected. The real EFSM demo needs a GPU for Qwen2.5-7B 4-bit inference. "
                    "Run with --mock for UI testing only."
                )

            asr_processor = AutoProcessor.from_pretrained(self.config.asr_model_id)
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.asr_model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to("cuda:0")
            asr_model.eval()
            self._asr = {"processor": asr_processor, "model": asr_model}

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

        audio_file = Path(audio_path)
        for _ in range(20):
            if self._stop_event.is_set():
                return ""
            if audio_file.exists() and audio_file.stat().st_size > 0:
                break
            time.sleep(0.15)
        if not audio_file.exists() or audio_file.stat().st_size == 0:
            return ""

        self.load()
        assert self._asr is not None
        import torch
        import librosa

        # Gradio public links can hand us browser-recorded audio files whose
        # container metadata confuses the Transformers ASR pipeline. We bypass
        # that pipeline entirely and call Whisper generate() on a plain mono
        # 16 kHz waveform.
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        processor = self._asr["processor"]
        asr_model = self._asr["model"]
        inputs = processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device="cuda:0", dtype=torch.float16)

        with torch.no_grad():
            predicted_ids = asr_model.generate(input_features)
        if self._stop_event.is_set():
            return ""
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return " ".join(text.strip().split())

    def respond(self, user_text: str, history: list[list[str | None]], system_prompt: str) -> str:
        if self.config.mock:
            return (
                "It makes sense that you feel overwhelmed with the deadline so close. "
                "You have already done the hardest research and training work, and now the pressure is about making it presentable. "
                "Let's keep the next step small: what part feels most urgent right now, the demo, the report, or the evaluation evidence?"
            )
        if self._stop_event.is_set():
            return ""

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
        for user_turn, assistant_turn in history[-8:]:
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
                min_new_tokens=min(70, self.config.max_new_tokens // 2),
                do_sample=do_sample,
                temperature=self.config.temperature if do_sample else None,
                top_p=self.config.top_p if do_sample else None,
                repetition_penalty=1.08,
                stopping_criteria=StoppingCriteriaList([InterruptStoppingCriteria()]),
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        if self._stop_event.is_set():
            return ""
        response = self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        if self.needs_companion_rewrite(response):
            response = self.rewrite_as_companion_response(user_text, response, system_prompt)
        return response

    def needs_companion_rewrite(self, response: str) -> bool:
        stripped = response.strip()
        if not stripped:
            return True
        sentence_count = sum(stripped.count(mark) for mark in [".", "!", "?"])
        lowered = stripped.lower()
        generic_phrases = [
            "sorry to hear that",
            "that's terrible news",
            "you will get through this",
            "oh no",
        ]
        return sentence_count < 3 or len(stripped.split()) < 35 or any(phrase in lowered for phrase in generic_phrases)

    def rewrite_as_companion_response(self, user_text: str, draft: str, system_prompt: str) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        assert self._model is not None
        assert self._tokenizer is not None
        if self._stop_event.is_set():
            return ""

        stop_event = self._stop_event

        class InterruptStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return stop_event.is_set()

        repair_prompt = (
            f"{system_prompt.strip()}\n\n"
            "The previous assistant reply was too short or generic. Rewrite it as a human conversational companion.\n"
            "Rules: write 3 to 5 natural sentences; be specific to the user's situation; include one grounded reframe or motivation; "
            "end with a gentle invitation to keep talking. Do not mention that you are rewriting.\n"
            f"User message: {user_text}\n"
            f"Bad draft: {draft}"
        )
        messages = [
            {"role": "system", "content": repair_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                min_new_tokens=min(80, self.config.max_new_tokens // 2),
                do_sample=True,
                temperature=max(self.config.temperature, 0.75),
                top_p=self.config.top_p,
                repetition_penalty=1.08,
                stopping_criteria=StoppingCriteriaList([InterruptStoppingCriteria()]),
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        if self._stop_event.is_set():
            return ""
        return self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def infer_emotion(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in ["depressed", "hopeless", "lost", "sad", "failed", "cry", "worthless"]):
            return "sad"
        if any(word in lowered for word in ["anxious", "panic", "scared", "afraid", "worry", "nervous"]):
            return "anxious"
        if any(word in lowered for word in ["angry", "mad", "furious", "betrayed", "annoyed"]):
            return "angry"
        if any(word in lowered for word in ["happy", "proud", "excited", "relieved", "grateful"]):
            return "positive"
        return "neutral"

    def edge_voice_controls(self, user_text: str) -> tuple[str, str]:
        emotion = self.infer_emotion(user_text)
        rate = "-4%"
        pitch = "-2Hz"
        if emotion in {"sad", "anxious"}:
            rate = "-10%"
            pitch = "-4Hz"
        elif emotion == "angry":
            rate = "-8%"
            pitch = "-3Hz"
        elif emotion == "positive":
            rate = "+0%"
            pitch = "+2Hz"
        return rate, pitch

    def clean_text_for_tts(self, text: str) -> str:
        text = html.unescape(text or "")
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\[(emotion|assistant|user|system):[^\]]+\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[_#>~|]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def make_temp_audio_path(self, suffix: str) -> Path:
        handle = tempfile.NamedTemporaryFile(prefix="efsm_reply_", suffix=suffix, delete=False)
        path = Path(handle.name)
        handle.close()
        return path

    def synthesize(self, text: str, user_text: str = "") -> str | None:
        text = self.clean_text_for_tts(text)
        if not text:
            return None
        if self._stop_event.is_set():
            return None

        if self.config.tts_provider in {"auto", "edge"}:
            edge_path = self.make_temp_audio_path(".mp3")
            try:
                import edge_tts

                if self._stop_event.is_set():
                    return None
                rate, pitch = self.edge_voice_controls(user_text)
                communicate = edge_tts.Communicate(
                    text,
                    voice=self.config.edge_voice,
                    rate=rate,
                    pitch=pitch,
                )
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(communicate.save(str(edge_path)))
                finally:
                    loop.close()
                if self._stop_event.is_set():
                    return None
                return str(edge_path)
            except Exception as exc:
                if self.config.tts_provider == "edge":
                    raise
                print(f"Edge TTS failed, trying fallback TTS: {exc}")

        if self.config.tts_provider in {"auto", "gtts"}:
            gtts_path = self.make_temp_audio_path(".mp3")
            try:
                from gtts import gTTS

                if self._stop_event.is_set():
                    return None
                gTTS(text=text, lang="en", slow=False).save(str(gtts_path))
                if self._stop_event.is_set():
                    return None
                return str(gtts_path)
            except Exception as exc:
                if self.config.tts_provider == "gtts":
                    raise
                print(f"gTTS failed, trying local fallback TTS: {exc}")

        out_path = self.make_temp_audio_path(".wav")
        if self.config.tts_provider in {"auto", "espeak"} and os.name != "nt" and shutil.which("espeak"):
            proc = subprocess.Popen(
                [
                    "espeak",
                    "-v",
                    "en",
                    "-s",
                    str(self.config.tts_rate),
                    "-w",
                    str(out_path),
                    text,
                ]
            )
            while proc.poll() is None:
                if self._stop_event.is_set():
                    proc.terminate()
                    return None
                time.sleep(0.1)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)
            if self._stop_event.is_set():
                return None
            return str(out_path)

        if self.config.tts_provider == "espeak":
            raise RuntimeError("espeak was requested, but the espeak command was not found.")

        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "No usable TTS provider was available. Use --tts-provider edge or install gTTS/pyttsx3."
            ) from exc

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
    #primary_voice_btn { min-height: 48px; }
    """

    with gr.Blocks(title="EFSM Speech Demo", css=css) as demo:
        gr.Markdown(
            "# EFSM Speech Demo\n"
            "Speak into the microphone, then EFSM transcribes, responds with session memory, and returns speech."
        )
        status = gr.Textbox(label="Status", value="Models are not loaded yet.", interactive=False, elem_id="status")

        with gr.Row():
            load_btn = gr.Button("Load Models", variant="secondary")
            interrupt_btn = gr.Button("Interrupt", variant="stop")
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
                submit_audio = gr.Button("Retry Last Recording", variant="secondary", elem_id="primary_voice_btn")
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
            history = history or []
            yield history, history, "", None, "Listening to your input..."

            user_text = typed_text.strip() if typed_text and typed_text.strip() else engine.transcribe(audio_path)
            if not user_text:
                if engine.is_interrupted():
                    yield history, history, "", None, "Interrupted. You can speak again now."
                else:
                    yield history, history, "", None, "No speech was detected. Wait a moment, then press Send / Retry Voice."
                return

            yield history + [[user_text, None]], history, user_text, None, "Transcribed. Generating EFSM response..."

            assistant_text = engine.respond(user_text, history, prompt)
            if engine.is_interrupted() or not assistant_text:
                yield history, history, user_text, None, "Interrupted. You can speak again now."
                return

            completed_history = history + [[user_text, assistant_text]]
            yield completed_history, completed_history, user_text, None, "Response ready. Creating voice reply..."

            audio_out = engine.synthesize(assistant_text, user_text)
            if engine.is_interrupted():
                yield completed_history, completed_history, user_text, None, "Interrupted. You can speak again now."
                return
            yield completed_history, completed_history, user_text, audio_out, "Turn complete. You can record the next message."

        def clear_history():
            return [], [], "", None, "Conversation cleared."

        def interrupt_turn():
            engine.interrupt()
            return None, "Interrupted. You can speak again now."

        def begin_recording():
            engine.interrupt()
            return None, "Listening. Tap stop when you finish speaking."

        stop_audio_js = """() => {
            document.querySelectorAll('audio').forEach((audio) => {
                audio.pause();
                audio.currentTime = 0;
                audio.src = "";
                audio.load();
            });
        }"""

        load_btn.click(load_models, outputs=status)
        audio_event = submit_audio.click(
            handle_turn,
            inputs=[microphone, manual_text, history_state, system_prompt],
            outputs=[chatbot, history_state, transcript, reply_audio, status],
            concurrency_limit=1,
            trigger_mode="always_last",
        )
        microphone.start_recording(
            begin_recording,
            outputs=[reply_audio, status],
            cancels=[audio_event],
            queue=False,
            js=stop_audio_js,
        )
        auto_audio_event = microphone.stop_recording(
            handle_turn,
            inputs=[microphone, manual_text, history_state, system_prompt],
            outputs=[chatbot, history_state, transcript, reply_audio, status],
            concurrency_limit=1,
            trigger_mode="always_last",
        )
        text_event = submit_text.click(
            handle_turn,
            inputs=[microphone, manual_text, history_state, system_prompt],
            outputs=[chatbot, history_state, transcript, reply_audio, status],
            concurrency_limit=1,
            trigger_mode="always_last",
        )
        interrupt_btn.click(
            interrupt_turn,
            outputs=[reply_audio, status],
            cancels=[audio_event, auto_audio_event, text_event],
            queue=False,
            js=stop_audio_js,
        )
        clear_btn.click(clear_history, outputs=[chatbot, history_state, transcript, reply_audio, status])

    return demo


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="Run the EFSM speech UI demo.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-id", default="tasbid001/efsm-checkpoints-fixed")
    parser.add_argument("--adapter-subfolder", default="checkpoint-2667")
    parser.add_argument("--asr-model-id", default="openai/whisper-base")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--tts-rate", type=int, default=165)
    parser.add_argument("--tts-provider", choices=["auto", "edge", "gtts", "espeak", "pyttsx3"], default="auto")
    parser.add_argument("--edge-voice", default="en-US-JennyNeural")
    parser.add_argument("--edge-style", default="auto")
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
