import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "")
API_KEY = os.getenv("API_KEY", "")

MODELS = {
    # "step-3.5": "openrouter/stepfun/step-3.5-flash:free",
    # "Llama-3.2-1B": "huggingface/meta-llama/Llama-3.2-1B-Instruct",
    # "Llama-3.1-8B": "huggingface/meta-llama/Llama-3.1-8B-Instruct",
    # "Llama-4-Scout-17B": "huggingface/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "GigaChat-2-Lite": "gigachat/GigaChat-2",
    "GigaChat-2-Pro": "gigachat/GigaChat-2-Pro",
    "GigaChat-2-Max": "gigachat/GigaChat-2-Max",
}
