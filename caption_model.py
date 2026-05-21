import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st

@st.cache_resource
def load_blip_model():
    """
    Loads and caches the BLIP Model and Processor.
    Using st.cache_resource ensures the model is loaded into memory only once,
    drastically improving inference speed for subsequent requests.
    """
    # Automatically select GPU if available for 2026-level speed optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model ID from Hugging Face
    model_id = "Salesforce/blip-image-captioning-base"
    
    # Load processor (handles image resizing & tokenization) and the model itself
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    
    return processor, model, device

def generate_image_caption(image, strategy="conditional"):
    """
    Generates a natural language caption for a given PIL Image.
    Supports two strategies: 'conditional' (structured) and 'unconditional' (creative).
    """
    try:
        processor, model, device = load_blip_model()
        
        # Preprocess the image and convert to PyTorch tensors
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        captions = []
        
        # Strategy 1: Deterministic/Conditional Captioning
        if strategy == "conditional" or strategy == "both":
            text = "a photography of"
            conditional_inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            out_cond = model.generate(**conditional_inputs, max_new_tokens=50)
            caption_cond = processor.decode(out_cond[0], skip_special_tokens=True)
            captions.append(caption_cond.capitalize())
            
        # Strategy 2: Purely Generative / Unconditional (Nucleus Sampling)
        if strategy == "unconditional" or strategy == "both":
            # Using sampling parameters for a more creative, descriptive generation
            out_uncond = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True, 
                top_k=50, 
                top_p=0.92
            )
            caption_uncond = processor.decode(out_uncond[0], skip_special_tokens=True)
            captions.append(caption_uncond.capitalize())
            
        return captions
    except Exception as e:
        raise RuntimeWarning(f"Error during caption generation: {str(e)}")