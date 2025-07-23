def main():
    from models.mistral_loader import load_mistral_pipeline
    from utils.hf_auth import hf_login_from_env
    hf_login_from_env()
    print("Hello from course-advisor!")
    llm_pipe = load_mistral_pipeline("mistralai/Mistral-7B-Instruct-v0.2")

if __name__ == "__main__":
    main()
