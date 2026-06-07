"""
ELC 2025-26 | Vision-Language Model
Image Captioning + Visual Question Answering
Model: BLIP (Salesforce) | UI: Tkinter
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import torch, threading, time

# ── Global state ──────────────────────────────────────────────────────────────
current_image   = None
models_ready    = False
cap_processor = cap_model = vqa_processor = vqa_model = None

def load_models():
    global cap_processor, cap_model, vqa_processor, vqa_model, models_ready
    set_status("Loading caption model…  (first run downloads ~500 MB, please wait)")
    cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model.eval()
    set_status("Loading VQA model…")
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model     = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model.eval()
    models_ready = True
    set_status("✅  Models ready — upload an image to begin")
    # Enable buttons
    btn_caption.config(state="normal")
    btn_vqa.config(state="normal")

def set_status(msg):
    root.after(0, lambda: status_var.set(msg))

def set_output(msg):
    root.after(0, lambda: _set_output(msg))

def _set_output(msg):
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, msg)
    result_box.config(state="disabled")

# ── Actions ───────────────────────────────────────────────────────────────────
def upload_image():
    global current_image
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if not path: return
    current_image = Image.open(path).convert("RGB")

    # Show thumbnail — fit inside 300x240, add subtle border
    thumb = current_image.copy()
    thumb.thumbnail((300, 240), Image.LANCZOS)
    thumb = ImageOps.expand(thumb, border=2, fill="#00A8CC")
    img_tk = ImageTk.PhotoImage(thumb)
    img_label.config(image=img_tk, text="")
    img_label.image = img_tk

    fname = path.split("/")[-1]
    set_status(f"📁  {fname}  ({current_image.width}×{current_image.height}px)")
    set_output("")

def do_caption():
    if current_image is None:
        return set_output("⚠️  Please upload an image first.")
    btn_caption.config(state="disabled")
    set_output("Generating caption…")
    def run():
        t = time.time()
        inp = cap_processor(images=current_image, return_tensors="pt")
        with torch.no_grad():
            out = cap_model.generate(**inp, max_new_tokens=60)
        cap = cap_processor.decode(out[0], skip_special_tokens=True)
        elapsed = time.time() - t
        set_output(f"📝  Caption\n\n{cap.capitalize()}\n\n⏱  {elapsed:.2f}s")
        root.after(0, lambda: btn_caption.config(state="normal"))
    threading.Thread(target=run, daemon=True).start()

def do_vqa():
    if current_image is None:
        return set_output("⚠️  Please upload an image first.")
    q = question_var.get().strip()
    if not q or q == PLACEHOLDER:
        return set_output("⚠️  Please type a question.")
    btn_vqa.config(state="disabled")
    set_output(f'Answering: "{q}"...')
    def run():
        t = time.time()
        inp = vqa_processor(images=current_image, text=q, return_tensors="pt")
        with torch.no_grad():
            out = vqa_model.generate(**inp, max_new_tokens=40)
        ans = vqa_processor.decode(out[0], skip_special_tokens=True)
        elapsed = time.time() - t
        set_output(f"❓  Question\n{q}\n\n💬  Answer\n{ans.capitalize()}\n\n⏱  {elapsed:.2f}s")
        root.after(0, lambda: btn_vqa.config(state="normal"))
    threading.Thread(target=run, daemon=True).start()

# ── Colours & fonts ───────────────────────────────────────────────────────────
BG       = "#0F1923"
PANEL    = "#172030"
ACCENT   = "#00A8CC"
ACCENT2  = "#00BFA5"
TEXT     = "#E8EDF2"
MUTED    = "#7A8FA6"
ENTRY_BG = "#1C2B3A"
BTN_DIS  = "#2A3A4A"

F_TITLE  = ("Helvetica", 17, "bold")
F_LABEL  = ("Helvetica", 11, "bold")
F_BODY   = ("Helvetica", 10)
F_MONO   = ("Courier", 11)
F_STATUS = ("Helvetica", 9)

PLACEHOLDER = "e.g.  What color is the tennis ball?"

# ── Root window ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("VLM Demo — ELC 2025-26")
root.geometry("820x580")
root.configure(bg=BG)
root.resizable(False, False)

status_var = tk.StringVar(value="⏳  Loading models in background…")

# ── Header ────────────────────────────────────────────────────────────────────
hdr = tk.Frame(root, bg=PANEL, pady=14)
hdr.pack(fill="x")
tk.Label(hdr, text="Vision-Language Model", font=F_TITLE,
         fg=ACCENT, bg=PANEL).pack()
tk.Label(hdr, text="Image Captioning  ·  Visual Question Answering  ·  ELC 2025-26",
         font=F_STATUS, fg=MUTED, bg=PANEL).pack()

# ── Main two-column layout ────────────────────────────────────────────────────
body = tk.Frame(root, bg=BG)
body.pack(fill="both", expand=True, padx=22, pady=14)

# LEFT — image panel
left = tk.Frame(body, bg=PANEL, width=310, padx=14, pady=14)
left.pack(side="left", fill="y")
left.pack_propagate(False)

tk.Label(left, text="INPUT IMAGE", font=("Helvetica", 9, "bold"),
         fg=MUTED, bg=PANEL).pack(anchor="w")

img_label = tk.Label(left, text="No image loaded\n\nClick Upload Image\nto begin",
                     width=28, height=13, bg="#0C1520", fg=MUTED,
                     font=F_BODY, relief="flat")
img_label.pack(pady=(8, 12))

tk.Button(left, text="⬆   Upload Image", command=upload_image,
          bg=ACCENT, fg="white", font=("Helvetica", 11, "bold"),
          relief="flat", padx=0, pady=8, cursor="hand2",
          activebackground="#0090AA", activeforeground="white",
          bd=0).pack(fill="x")

tk.Label(left, text="Supported: JPG · PNG · BMP · WEBP",
         font=("Helvetica", 8), fg=MUTED, bg=PANEL).pack(pady=(6, 0))

# RIGHT — controls + output
right = tk.Frame(body, bg=BG, padx=18)
right.pack(side="right", fill="both", expand=True)

# — Caption section —
tk.Label(right, text="IMAGE CAPTIONING", font=("Helvetica", 9, "bold"),
         fg=MUTED, bg=BG).pack(anchor="w")
tk.Frame(right, bg=ACCENT, height=1).pack(fill="x", pady=(2, 8))

btn_caption = tk.Button(right, text="✦   Generate Caption",
                        command=do_caption,
                        bg=ACCENT2, fg="white",
                        font=("Helvetica", 11, "bold"),
                        relief="flat", pady=8, cursor="hand2",
                        activebackground="#009E87", activeforeground="white",
                        disabledforeground="#888", bd=0, state="disabled")
btn_caption.pack(fill="x")

# — VQA section —
tk.Label(right, text="VISUAL QUESTION ANSWERING", font=("Helvetica", 9, "bold"),
         fg=MUTED, bg=BG).pack(anchor="w", pady=(18, 0))
tk.Frame(right, bg=ACCENT, height=1).pack(fill="x", pady=(2, 8))

question_var = tk.StringVar()
q_entry = tk.Entry(right, textvariable=question_var,
                   font=("Helvetica", 11),
                   bg=ENTRY_BG, fg=MUTED,
                   insertbackground=ACCENT,
                   relief="flat", bd=0)
q_entry.pack(fill="x", ipady=9)
q_entry.insert(0, PLACEHOLDER)

def on_focus_in(e):
    if q_entry.get() == PLACEHOLDER:
        q_entry.delete(0, tk.END)
        q_entry.config(fg=TEXT)
def on_focus_out(e):
    if not q_entry.get().strip():
        q_entry.insert(0, PLACEHOLDER)
        q_entry.config(fg=MUTED)

q_entry.bind("<FocusIn>",  on_focus_in)
q_entry.bind("<FocusOut>", on_focus_out)
q_entry.bind("<Return>",   lambda e: do_vqa())

btn_vqa = tk.Button(right, text="⌕   Get Answer",
                    command=do_vqa,
                    bg="#065A82", fg="white",
                    font=("Helvetica", 11, "bold"),
                    relief="flat", pady=8, cursor="hand2",
                    activebackground="#054D6E", activeforeground="white",
                    disabledforeground="#888", bd=0, state="disabled")
btn_vqa.pack(fill="x", pady=(8, 0))

# — Output section —
tk.Label(right, text="OUTPUT", font=("Helvetica", 9, "bold"),
         fg=MUTED, bg=BG).pack(anchor="w", pady=(18, 0))
tk.Frame(right, bg=ACCENT, height=1).pack(fill="x", pady=(2, 6))

result_box = scrolledtext.ScrolledText(right, height=8,
                                        font=F_MONO,
                                        bg="#0C1520", fg=TEXT,
                                        relief="flat", state="disabled",
                                        wrap="word", padx=10, pady=10,
                                        insertbackground=ACCENT)
result_box.pack(fill="both", expand=True)

# ── Status bar ────────────────────────────────────────────────────────────────
tk.Frame(root, bg="#0A1218", height=1).pack(fill="x")
tk.Label(root, textvariable=status_var, font=F_STATUS,
         fg=MUTED, bg="#0A1218", anchor="w", padx=12, pady=5).pack(fill="x")

# ── Start model loading ───────────────────────────────────────────────────────
threading.Thread(target=load_models, daemon=True).start()

root.mainloop()
