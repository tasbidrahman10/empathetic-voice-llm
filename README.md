# Empathetic Voice LLM Demo

**CSE465 - Pattern Recognition and Neural Networks**  
**Student:** Tasbid Al Rahman  
**ID:** 2232225642

This project is a simple speech-based empathetic chatbot demo.

The system takes speech or an uploaded audio file, understands the message, generates an empathetic reply, and speaks the reply back to the user.

```text
Speech / audio file -> Speech recognition -> Empathetic LLM -> Voice response
```

---

## Recommended Demo: Kaggle Notebook

This is the easiest way to run the full demo.

Notebook:

```text
notebooks/04_kaggle_full_demo.ipynb
```

### Step 1: Open the Notebook in Kaggle

1. Open Kaggle.
2. Create a new notebook or import this repository.
3. Open the notebook:

```text
notebooks/04_kaggle_full_demo.ipynb
```

### Step 2: Enable GPU and Internet

In the Kaggle notebook settings:

1. Turn on **Internet**.
2. Select **GPU** as the accelerator.

### Step 3: Add the Hugging Face Secret

The demo needs a Hugging Face token to load the model.

In Kaggle:

1. Go to **Add-ons**.
2. Open **Secrets**.
3. Add a new secret named:

```text
HF_TOKEN: paste_your_hugging_face_token_here
```

4. Paste the Hugging Face token/key as the value.
5. Enable the secret for the notebook.

The secret name must be exactly:

```text
HF_TOKEN
```

### Step 4: Run All Cells

Run the notebook cells from top to bottom.

The final cell will start the Gradio demo and show a public link like:

```text
https://xxxxx.gradio.live
```

Click the Gradio link.

### Step 5: Load the Models

After the Gradio page opens:

1. Click **Load Models**.
2. Wait until the models finish loading.

This may take a few minutes.

### Step 6: Use the Demo

After the models are loaded, the professor can test the system in two ways:

1. Speak using the microphone.
2. Upload a `.wav` or `.mp3` audio file.

After a few seconds, the system will reply with an empathetic spoken response.

---

## Sample Prompts to Try

These can be spoken through the microphone or recorded/uploaded as audio.

### Sad

```text
I have been feeling really low lately.
It feels like nobody understands what I am going through.
I try to stay strong, but some days feel too heavy.
I just need someone to listen.
```

### Happy

```text
I had a really good day today.
Something I worked hard for finally went well.
I feel proud and excited, and I wanted to share it.
It feels nice to have a moment like this.
```

### Angry

```text
I am really frustrated right now.
I feel like my effort is not being respected.
I tried to stay calm, but it keeps bothering me.
I do not know how to let this go.
```

### Anxious

```text
I am worried about what might happen next.
My mind keeps thinking about all the things that could go wrong.
I know I should calm down, but it feels difficult.
I just want to feel a little more in control.
```

---

## Alternative Demo: Local Notebook

If Kaggle is not used, the demo can also be opened from the local notebook.

Notebook:

```text
notebooks/04_local_full_demo.ipynb
```

Simple process:

1. Open Jupyter Notebook or JupyterLab.
2. Open:

```text
notebooks/04_local_full_demo.ipynb
```

3. Run the cells one by one.
4. When the notebook asks for `HF_TOKEN`, paste the Hugging Face token/key.
5. Open the Gradio link shown by the notebook.
6. Click **Load Models**.
7. Wait for the models to load.
8. Speak using the microphone or upload a `.wav`/`.mp3` file.
9. The system will reply with an empathetic voice response.

---

## Notes

- Kaggle is recommended because it provides GPU support.
- The first model loading step can take several minutes.
- Internet must be enabled in Kaggle.
- If the Gradio link does not appear, run the final notebook cell again.
- If the model does not load, check that the Kaggle secret is named exactly `HF_TOKEN`.

---

## Model Summary

The demo uses:

```text
Whisper speech recognition
Fine-tuned Qwen2.5-7B-Instruct language model
Text-to-speech voice output
```

The main goal is to demonstrate an empathetic speech-to-speech conversation system.
