# models/mistral_loader.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def load_mistral_pipeline(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)