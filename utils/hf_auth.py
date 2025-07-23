# utils/hf_auth.py

import os
from huggingface_hub import login
from dotenv import load_dotenv

def hf_login_from_env():
    load_dotenv() 
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token is None:
        raise ValueError("Hugging Face token not found in .env file.")
    login(token=token)
    print("Hugging Face login successful")