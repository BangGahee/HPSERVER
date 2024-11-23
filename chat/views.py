from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
from django.http import JsonResponse
from django.shortcuts import render
import gc  # For garbage collection

HUGGINGFACE_MODEL_NAME = "reciprocate/tiny-llama"
HUGGINGFACE_AUTH_TOKEN = "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"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

def index(request):
    """Render the main page."""
    return render(request, 'chat/index.html')

def load_model():
    """Load the model and tokenizer."""
    global tokenizer, model
    try:
        print("Loading model...")
        tokenizer = LlamaTokenizer.from_pretrained(
            HUGGINGFACE_MODEL_NAME,
            token=HUGGINGFACE_AUTH_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(
            HUGGINGFACE_MODEL_NAME,
            token=HUGGINGFACE_AUTH_TOKEN
        ).to(DEVICE)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

def chat(request):
    """Process chat messages."""
    global tokenizer, model

    if not tokenizer or not model:
        return JsonResponse({"error": "Model not loaded."}, status=500)

    if request.method == "POST":
        user_input = request.POST.get("message")
        if not user_input:
            return JsonResponse({"error": "No message provided."}, status=400)

        if not user_input.strip():
            return JsonResponse({"error": "Message cannot be empty or whitespace."}, status=400)

        try:
            # Tokenize input and ensure proper formatting
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs["input_ids"].to(DEVICE)  # Extract input IDs and move to the correct device
            print(f"Tokenized inputs: {inputs}")

            # Generate response
            outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
            print(f"Model outputs: {outputs}")
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated response: {response}")
            return JsonResponse({"message": response})

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return JsonResponse({"error": "Out of memory. Please try again later."}, status=500)
        except Exception as e:
            error_message = f"Exception occurred: {str(e)}"
            detailed_traceback = traceback.format_exc()
            print(f"Error message: {error_message}")
            print(f"Traceback details:\n{detailed_traceback}")
            return JsonResponse({"error": "An internal error occurred. Check server logs for details."}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method. Use POST."}, status=405)


# Load the model when the server starts
try:
    load_model()
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model is on: {next(model.parameters()).device}")

except Exception as e:
    print("Failed to load model on startup.")
