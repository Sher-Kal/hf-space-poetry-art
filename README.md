# 🎭 Poetry & Painting Demo

A Hugging Face **Space project** that combines **text generation** and **image generation** in a simple Gradio web app.  
Built as a portfolio piece to demonstrate how to integrate LLMs and diffusion models with style-switching.

---

## ✨ Features

- **Image Generation (Stable Diffusion)**  
  Generates pictures from text prompts.  
  Extra style buttons:  
  - 🎨 Alexei **Savrasov** (19th-century Russian realism, early spring moods)  
  - 🎨 Édouard **Manet** (French impressionist realism)

- **Poetry Generation (small GPT-2 models)**  
  Generates short poems in English.  
  Available styles:  
  - Neutral (free verse)  
  - Lord **Byron** (romantic, dramatic tone)  
  - Edgar Allan **Poe** (dark, gothic imagery)

- **Clean Gradio UI**  
  Tabs for Text and Image, sliders for parameters, and simple style dropdowns/buttons.

---

## 🚀 Usage

Run locally:

```bash
git clone https://github.com/Sher-Kal/hf-space-poetry-art.git
cd hf-space-poetry-art
pip install -r requirements.txt
python app.py
```

Or open the public Hugging Face Space:  
👉 [Demo on Hugging Face Spaces](https://huggingface.co/spaces/LightFuture/OAC1)

---

## ⚠️ Limitations

- **Image generation** works well, but English prompts give the best results.  
- **Poetry generation** is unstable:  
  - outputs are short, sometimes prose-like;  
  - rhyme/structure quality is limited.  
- These issues are not bugs of the app itself but are due to:  
  - small open GPT-2 models used (to fit free tier constraints),  
  - free Hugging Face Space (CPU only, 1 GB disk).

---

## 🎯 Why this project matters for a portfolio

- Shows ability to combine **text + image pipelines**.  
- Demonstrates integration of **style-switching** in the UI.  
- Highlights awareness of model quality vs. infrastructure constraints.  
- Presents a polished, easy-to-use Gradio application with clear limitations.

---

## 📄 License

MIT
