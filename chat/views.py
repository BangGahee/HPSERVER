from transformers import LlamaTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from django.http import JsonResponse
from django.shortcuts import render
import gc  # For garbage collection

HUGGINGFACE_AUTH_TOKEN = "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"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
reinforce_model = None
unlearn_model = None

def index(request):
    return render(request, 'chat/index.html')

def load_models():
    global tokenizer, reinforce_model, unlearn_model
    try:
        print("Loading models...")

        # Tokenizer 로드
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN
        )

        # Reinforce 모델 로드
        reinforce_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN,
            torch_dtype=torch.float16
        )
        reinforce_model = PeftModel.from_pretrained(
            reinforce_model,
            "gaheeBang/peft-adapter-harrypotter-4bit"
        ).to(DEVICE)

        # Unlearn 모델 로드
        unlearn_model = AutoModelForCausalLM.from_pretrained(
            "paul02/unlearned_HP_8bit",
            use_auth_token=HUGGINGFACE_AUTH_TOKEN,
            torch_dtype=torch.float16
        ).to(DEVICE)

        print("Models loaded successfully!")
    except Exception as e:
        print(f"Failed to load models: {e}")
        raise e

def chat(request):
    global tokenizer, reinforce_model, unlearn_model

    if not tokenizer or not reinforce_model or not unlearn_model:
        return JsonResponse({"error": "Models not loaded."}, status=500)

    if request.method == "POST":
        user_input = request.POST.get("message")
        model_type = request.POST.get("model_type")

        if not user_input or not user_input.strip():
            return JsonResponse({"error": "Invalid message."}, status=400)

        if model_type not in ["reinforce", "unlearn"]:
            return JsonResponse({"error": "Invalid model type."}, status=400)

        try:
            model = reinforce_model if model_type == "reinforce" else unlearn_model
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=100).to(DEVICE)
            
            outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return JsonResponse({"message": response})

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return JsonResponse({"error": "Out of memory."}, status=500)
        except Exception as e:
            print(f"Error during chat: {e}")
            return JsonResponse({"error": "An internal error occurred."}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)

try:
    load_models()
except Exception as e:
    print("Failed to load models on startup.")
