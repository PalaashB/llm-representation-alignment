import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_DIR = "results/tinyllama"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

print("Loading TruthfulQA...")
dataset = load_dataset("truthful_qa", "generation")

questions = dataset["validation"]["question"][:100]

print(f"Loaded {len(questions)} questions.")

layer_outputs = None

for idx, question in enumerate(questions):

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states

    if layer_outputs is None:
        layer_outputs = [[] for _ in range(len(hidden_states))]

    for layer_idx, layer in enumerate(hidden_states):

        # Mean pool over sequence length
        embedding = layer.mean(dim=1).squeeze(0).cpu()

        layer_outputs[layer_idx].append(embedding)

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(questions)} questions")

print("Saving tensors...")

for layer_idx in range(len(layer_outputs)):

    tensor = torch.stack(layer_outputs[layer_idx])

    torch.save(
        tensor,
        os.path.join(
            SAVE_DIR,
            f"layer_{layer_idx}.pt"
        )
    )

print("Done!")
print(f"Saved to {SAVE_DIR}")