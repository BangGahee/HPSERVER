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

def index(request):
    """Render the main page."""
    return render(request, 'chat/index.html')
    
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
            # inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=100)

            inputs = tokenizer([user_input1, user_input2], return_tensors="pt", truncation=True, padding=True, max_length=100)
            inputs = inputs["input_ids"].to(DEVICE)  # Move input IDs to the correct device

            outputs = model.generate(inputs, max_length=100)

            # 응답 생성
            # Beam search 대신 sampling 사용
            # outputs = model.generate(
            #     inputs,
            #     max_length=100,
            #     num_return_sequences=1,
            #     do_sample=True,
            #     top_k=50,
            #     top_p=0.9,
            #     temperature=0.8,
            # )


            # outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
            # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
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
