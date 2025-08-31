import os
import re
import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline, AutoTokenizer

# ==== Конфиг моделей ====
LOCAL_TEXT_MODEL = os.getenv("LOCAL_TEXT_MODEL", "ismaelfaro/gpt2-poems.en")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

# Токен для Inference API
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError(
        "Отсутствует HF_API_TOKEN. Добавь токен в Settings → Repository secrets, или экспортируй локально."
    )

image_client = InferenceClient(token=HF_API_TOKEN, provider="hf-inference")


# ===== Текстовая генерация =====
SYSTEM_PROMPT = "You are a helpful assistant. Answer briefly and to the point."

# Подсказки по стилю
STYLE_PROMPTS = {
    "neutral": "",
    "byron": (
        "Style: Lord Byron — romantic, dramatic, melancholic tone; "
        "evocative imagery; lofty language.\n\n"
    ),
    "poe": (
        "Style: Edgar Allan Poe — dark, gothic, eerie imagery; "
        "rhythmic and ominous, with shadows and ravens.\n\n"
    ),
}

# ---- примеры поэзии ----
FEW_SHOT_EXAMPLE = (
    "Example:\n"
    "The moon is pale above the sea,\n"
    "Shadows fall, the night is free.\n"
    "End Example.\n\n"
)

SECTION_INST = "### Instruction"
SECTION_Q    = "### Topic"
SECTION_A    = "### Poem"

def render_poetry_prompt(user_prompt: str, style_key: str) -> str:
    prefix = STYLE_PROMPTS.get(style_key, "")
    return (
        f"{prefix}"
        "Write a short poem in English.\n"
        "- 6 to 10 lines\n"
        "- 3 to 10 words per line\n"
        "- no title, no explanations\n\n"
        f"{FEW_SHOT_EXAMPLE}"
        f"Topic: {user_prompt.strip()}\n\n"
        "Poem:\n"
    )

# --- ленивое создание локального пайплайна для текста ---
_text_pipeline = None
_pad_token_id = None

def get_text_pipeline():
    global _text_pipeline, _pad_token_id
    if _text_pipeline is None:
        tok = AutoTokenizer.from_pretrained(LOCAL_TEXT_MODEL)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _pad_token_id = tok.pad_token_id
        _text_pipeline = pipeline(
            task="text-generation",
            model=LOCAL_TEXT_MODEL,
            tokenizer=tok,
            device_map="auto"
        )
    return _text_pipeline

def postprocess_poem(generated: str) -> str:
    s = generated.split("Poem:", 1)[-1]
    s = s.split("Example", 1)[0]
    s = s.replace("\\n", "\n").replace("\\r", "\n")
    s = re.sub(r"\S+@\S+|\bhttps?://\S+|\bwww\.\S+", "", s).strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    norm = []
    for ln in lines:
        norm.append(" ".join(ln.split()[:12]))
        if len(norm) >= 10:
            break
    return "\n".join(norm) if norm else "…"


def generate_poem(prompt: str, max_new_tokens: int = 220, temperature: float = 1.05, style: str = "neutral"):
    if not prompt.strip():
        return "Enter a topic (in English)."
    try:
        pipe = get_text_pipeline()
        final_prompt = render_poetry_prompt(prompt, style)
        out = pipe(
            final_prompt,
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=90,
            do_sample=True,
            temperature=float(temperature),
            top_p=0.95,
            top_k=50,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            pad_token_id=_pad_token_id,
            return_full_text=False,
        )
        raw = out[0]["generated_text"]
        return postprocess_poem(raw)
    except Exception as e:
        return f"Poem generation error: {e}"


# ===== Генерация изображения =====

# Подсказки по стилю
PAINTER_STYLES = {
    "none": "",
    "savrasov": (
        " in the style of Alexei Savrasov; 19th-century Russian landscape, "
        "muted palette, overcast sky, early spring mood, delicate atmospheric perspective, "
        "wet ground, crows, birch trees, understated realism"
    ),
    "manet": (
        " in the style of Édouard Manet; 19th-century French impressionist realism, "
        "loose brushwork, natural light, subdued yet rich colors, salon-era composition, "
        "modern life scenes"
    ),
}

def build_image_prompt(prompt: str, painter_key: str) -> str:
    base = prompt.strip()
    style = PAINTER_STYLES.get(painter_key, "")
    return (base + style).strip()

def generate_image(prompt: str, width: int = 768, height: int = 512):
    if not prompt or not prompt.strip():
        return None
    try:
        # SDXL нормально ест простой text_to_image через HF Inference
        img = image_client.text_to_image(
            prompt=prompt.strip(),
            model=IMAGE_MODEL,
            # у разных провайдеров часть аргументов может игнорироваться
        )
        return img
    except Exception as e:
        return None


# ===== Gradio UI =====
with gr.Blocks(title="Text + Image AI Demo") as demo:
    gr.Markdown(
        """
        # Text + Image AI Demo
        Небольшое демо: генерация **текста** и **изображений** через Hugging Face Inference API.
        """
    )

    with gr.Tab("Text-to-Image"):
        with gr.Row():
            img_prompt = gr.Textbox(
                label="Prompt (EN gives best results)",
                placeholder="A quiet village road after rain, early spring..."
            )
        with gr.Row():
            width = gr.Slider(384, 1024, value=768, step=32, label="Width")
            height = gr.Slider(256, 1024, value=512, step=32, label="Height")
        with gr.Row():
            painter = gr.State("none")
            btn_savrasov = gr.Button("Саврасов")
            btn_manet    = gr.Button("Мане")
            btn_reset    = gr.Button("Сброс стиля")

        # показывает текущий стиль
        style_info = gr.Markdown("Стиль: без стиля")

        def set_style_savrasov(_):
            return "savrasov", "Стиль: Саврасов"
        def set_style_manet(_):
            return "manet", "Стиль: Мане"
        def reset_style(_):
            return "none", "Стиль: без стиля"

        btn_savrasov.click(set_style_savrasov, inputs=[painter], outputs=[painter, style_info])
        btn_manet.click(set_style_manet, inputs=[painter], outputs=[painter, style_info])
        btn_reset.click(reset_style, inputs=[painter], outputs=[painter, style_info])

        gen_img_btn = gr.Button("Сгенерировать изображение")
        img_out = gr.Image(label="Результат", type="pil")

        def _on_image(p, w, h, painter_key):
            styled = build_image_prompt(p, painter_key or "none")
            return generate_image(styled, w, h)

        gen_img_btn.click(_on_image, inputs=[img_prompt, width, height, painter], outputs=[img_out])

    with gr.Tab("Poetry"):
        with gr.Row():
            prompt_in = gr.Textbox(
                label="Topic (EN)",
                placeholder="moonlit sea and distant lighthouse"
            )
        with gr.Row():
            max_tokens = gr.Slider(64, 240, value=96, step=8, label="max_new_tokens")
            temperature = gr.Slider(0.5, 1.3, value=0.9, step=0.05, label="temperature")
            style_dd = gr.Dropdown(
                choices=[("Neutral", "neutral"), ("Byron", "byron"), ("Poe", "poe")],
                value="neutral",
                label="Style"
            )
        gen_btn = gr.Button("Generate Poem")
        text_out = gr.Markdown()

        # обёртка, чтобы передать стиль
        def _on_poem(topic, mx, temp, style):
            return generate_poem(topic, mx, temp, style)

        gen_btn.click(_on_poem, inputs=[prompt_in, max_tokens, temperature, style_dd], outputs=[text_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
