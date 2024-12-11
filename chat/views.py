from transformers import LlamaTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from accelerate import infer_auto_device_map, dispatch_model
import gc

# Hugging Face 토큰과 디바이스 설정
HUGGINGFACE_AUTH_TOKEN = "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"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
reinforce_model = None
unlearn_model = None

def load_models():
    """Load the models and tokenizer with optimized device_map."""
    global tokenizer, reinforce_model, unlearn_model
    try:
        print("Loading models...")
        
        # Load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN
        )

        # Load base model with device map
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN,
            torch_dtype=torch.float16  # Use half-precision to save memory
        )

        # Automatically infer device map
        device_map = infer_auto_device_map(base_model, max_memory={"cpu": "24GB", "cuda": "16GB"})

        # Dispatch model to devices
        base_model = dispatch_model(base_model, device_map=device_map)

        # Load reinforcement model with PEFT adapter
        reinforce_model = PeftModel.from_pretrained(
            base_model,
            "gaheeBang/peft-adapter-harrypotter-4bit"
        )

        # Load unlearn model
        unlearn_model = AutoModelForCausalLM.from_pretrained(
            "paul02/unlearned_HP_8bit",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN,
            device_map="auto",  # Automatically assign to available devices
            torch_dtype=torch.float16
        )

        print("Models loaded successfully with optimized device maps!")
    except Exception as e:
        print(f"Failed to load models: {e}")
        raise e
