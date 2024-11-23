from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# 로컬 모델 경로
LOCAL_MODEL_PATH = "/Users/bang-gahee/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model from local path...")
tokenizer = LlamaTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)
print("Model loaded successfully!")

def index(request):
    """메인 페이지"""
    return render(request, 'chat/index.html')

def chat(request):
    if not model or not tokenizer:
        return JsonResponse({"error": "Model not loaded yet."}, status=500)

    if request.method == "POST":
        user_input = request.POST.get("message")
        if not user_input:
            return JsonResponse({"error": "No message provided."}, status=400)

        try:
            inputs = tokenizer(user_input, return_tensors="pt").to(DEVICE)
            outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return JsonResponse({"message": response})
        except Exception as e:
            print(f"Error during model processing: {e}")
            return JsonResponse({"error": "Failed to generate response."}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)
