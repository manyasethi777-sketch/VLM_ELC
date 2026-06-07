"""
===========================================================
  ELC 2025-26 | Vision-Language Model for Image Understanding
  Topic: Image Captioning + Visual Question Answering (VQA)
  Model: BLIP (Bootstrapping Language-Image Pre-training)
  Framework: Python + HuggingFace Transformers + Gradio
  Works on: CPU (No GPU required)
===========================================================
"""

import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import torch
import time

# ─── Load Models (done once at startup) ───────────────────────────────────────
print("Loading BLIP Caption model...")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("Loading BLIP VQA model...")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model     = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

caption_model.eval()
vqa_model.eval()
print("✅ Models loaded successfully!")


# ─── Task 1: Image Captioning ──────────────────────────────────────────────────
def generate_caption(image, max_new_tokens=50):
    """Generate a natural-language description of the given image."""
    if image is None:
        return "⚠️ Please upload an image first."
    
    start = time.time()
    inputs = caption_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        output = caption_model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    caption = caption_processor.decode(output[0], skip_special_tokens=True)
    elapsed = time.time() - start
    
    return f"📝 Caption: {caption}\n\n⏱ Inference time: {elapsed:.2f}s"


# ─── Task 2: Visual Question Answering ────────────────────────────────────────
def answer_question(image, question):
    """Answer a natural-language question about the given image."""
    if image is None:
        return "⚠️ Please upload an image first."
    if not question.strip():
        return "⚠️ Please enter a question."
    
    start = time.time()
    inputs = vqa_processor(images=image, text=question, return_tensors="pt")
    
    with torch.no_grad():
        output = vqa_model.generate(**inputs, max_new_tokens=30)
    
    answer = vqa_processor.decode(output[0], skip_special_tokens=True)
    elapsed = time.time() - start
    
    return f"💬 Answer: {answer}\n\n⏱ Inference time: {elapsed:.2f}s"


# ─── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="VLM: Image Captioning & VQA", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🔭 Vision-Language Model Demo
    ### ELC 2025-26 | Image Captioning + Visual Question Answering
    **Model**: BLIP (Salesforce) via HuggingFace Transformers  
    Upload any image, then either generate a caption or ask a question about it!
    """)

    # ── Tab 1: Captioning ──
    with gr.Tab("📷 Image Captioning"):
        gr.Markdown("Upload an image and the model will automatically describe it.")
        with gr.Row():
            cap_img = gr.Image(type="pil", label="Upload Image")
            cap_out = gr.Textbox(label="Generated Caption", lines=4)
        cap_btn = gr.Button("✨ Generate Caption", variant="primary")
        cap_btn.click(fn=generate_caption, inputs=[cap_img], outputs=[cap_out])
        
        gr.Examples(
            examples=[],
            inputs=[cap_img],
        )

    # ── Tab 2: VQA ──
    with gr.Tab("❓ Visual Question Answering"):
        gr.Markdown("Upload an image, type a question, and get an answer based on the image content.")
        with gr.Row():
            vqa_img = gr.Image(type="pil", label="Upload Image")
            with gr.Column():
                vqa_q   = gr.Textbox(label="Your Question", placeholder="e.g. What color is the car?")
                vqa_out = gr.Textbox(label="Model Answer", lines=4)
        vqa_btn = gr.Button("🔍 Get Answer", variant="primary")
        vqa_btn.click(fn=answer_question, inputs=[vqa_img, vqa_q], outputs=[vqa_out])
        
        gr.Markdown("""
        **Sample questions to try:**
        - What is in the image?
        - How many people are there?
        - What color is the sky?
        - Is it daytime or nighttime?
        - What is the person doing?
        """)

    # ── About ──
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About This Project

        ### Model Architecture
        **BLIP** (Bootstrapping Language-Image Pre-Training) uses:
        - A **Vision Encoder** (ViT — Vision Transformer) to extract image features
        - A **Text Decoder** (BERT-based) for caption generation
        - A **VQA Head** fine-tuned on VQA v2 dataset for question answering

        ### How It Works
        ```
        Image → ViT Encoder → Visual Features ─┐
                                                ├→ Cross-Attention → Text Output
        Text  → Text Encoder → Text Features  ─┘
        ```

        ### Datasets Used (Pre-training)
        | Task | Dataset |
        |------|---------|
        | Captioning | MS-COCO Captions |
        | VQA | VQA v2 |

        ### Performance Metrics (from paper)
        | Task | Metric | Score |
        |------|--------|-------|
        | Captioning | BLEU-4 | 36.7 |
        | Captioning | CIDEr | 121.7 |
        | VQA | Accuracy | 78.25% |
        
        ### References
        - Li et al., "BLIP: Bootstrapping Language-Image Pre-training", ICML 2022
        - HuggingFace: https://huggingface.co/Salesforce/blip-image-captioning-base
        """)

if __name__ == "__main__":
    demo.launch(share=False)  # set share=True for a public link
    # Note: uses Gradio 5.x (compatible with numpy 2.x)
