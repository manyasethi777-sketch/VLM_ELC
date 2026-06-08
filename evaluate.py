from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests, torch, time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Sample test images ──
TEST_SAMPLES = [
    {
        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "references": ["two cats lying on a couch", "a couple of cats on a sofa", "two cats resting on a pink blanket"]
    },
    {
        "url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "references": ["a man is playing baseball", "a baseball player swings the bat", "batter swinging at a pitch"]
    },
    {
        "url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "references": ["a man riding a wave on a surfboard", "a surfer on a large wave", "a person surfing in the ocean"]
    },
]

def load_model():
    print("Loading BLIP captioning model (CPU)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

def get_caption(image, processor, model):
    start = time.time()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption, round(time.time() - start, 2)

def compute_bleu(hypothesis, references):
    """Compute sentence BLEU-4 score."""
    hyp_tokens  = hypothesis.lower().split()
    ref_tokens  = [r.lower().split() for r in references]
    smoother    = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoother)

def evaluate():
    processor, model = load_model()
    
    print("\n" + "="*60)
    print("  BLIP IMAGE CAPTIONING — EVALUATION RESULTS")
    print("="*60)
    
    bleu_scores  = []
    infer_times  = []
    
    for i, sample in enumerate(TEST_SAMPLES):
        print(f"\n[Sample {i+1}]")
        try:
            image = Image.open(requests.get(sample["url"], stream=True, timeout=10).raw).convert("RGB")
        except Exception as e:
            print(f"  ⚠️  Could not load image: {e}")
            continue
        
        caption, elapsed = get_caption(image, processor, model)
        bleu             = compute_bleu(caption, sample["references"])
        
        bleu_scores.append(bleu)
        infer_times.append(elapsed)
        
        print(f"  Generated : {caption}")
        print(f"  Reference : {sample['references'][0]}")
        print(f"  BLEU-4    : {bleu:.4f}")
        print(f"  Time      : {elapsed}s")
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    if bleu_scores:
        print(f"  Avg BLEU-4 Score  : {sum(bleu_scores)/len(bleu_scores):.4f}")
        print(f"  Avg Inference Time: {sum(infer_times)/len(infer_times):.2f}s (CPU)")
        print(f"  Samples Evaluated : {len(bleu_scores)}")
    print("="*60)

if __name__ == "__main__":
    evaluate()
