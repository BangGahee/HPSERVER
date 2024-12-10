from transformers import LlamaTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from django.http import JsonResponse
from django.shortcuts import render
import gc  # For garbage collection

# Hugging Face 토큰과 디바이스 설정
HUGGINGFACE_AUTH_TOKEN = "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"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델과 토크나이저 전역 변수
tokenizer = None
reinforce_model = None
unlearn_model = None

def index(request):
    """Render the main page."""
    return render(request, 'chat/index.html')

def load_models():
    """Load the models and tokenizer."""
    global tokenizer, reinforce_model, unlearn_model
    try:
        print("Loading models...")
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN
        )
        
        # 강화 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        reinforce_model = PeftModel.from_pretrained(
            base_model,
            "gaheeBang/peft-adapter-harrypotter-4bit"
        ).to(DEVICE)

        # 언러닝 모델 로드
        base_model_unlearn = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        unlearn_model = PeftModel.from_pretrained(
            base_model_unlearn,
            "paul02/unlearned_HP_8bit"
        ).to(DEVICE)

        print("Models loaded successfully!")
    except Exception as e:
        print(f"Failed to load models: {e}")
        raise e

def chat(request):
    """Process chat messages."""
    global tokenizer, reinforce_model, unlearn_model

    if not tokenizer or not reinforce_model or not unlearn_model:
        return JsonResponse({"error": "Models not loaded."}, status=500)

    if request.method == "POST":
        user_input = request.POST.get("message")
        model_type = request.POST.get("model_type")  # "reinforce" or "unlearn"

        if not user_input:
            return JsonResponse({"error": "No message provided."}, status=400)

        if not user_input.strip():
            return JsonResponse({"error": "Message cannot be empty or whitespace."}, status=400)

        if model_type not in ["reinforce", "unlearn"]:
            return JsonResponse({"error": "Invalid model type. Use 'reinforce' or 'unlearn'."}, status=400)

        try:
            # 모델 선택
            model = reinforce_model if model_type == "reinforce" else unlearn_model

            # 입력 텍스트 토큰화
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs["input_ids"].to(DEVICE)  # Extract input IDs and move to the correct device

            # 응답 생성
            outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return JsonResponse({"message": response})

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return JsonResponse({"error": "Out of memory. Please try again later."}, status=500)
        except Exception as e:
            print(f"Exception occurred: {e}")
            return JsonResponse({"error": "An internal error occurred. Check server logs for details."}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# 서버 시작 시 모델 로드
try:
    load_models()
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print("Failed to load models on startup.")
