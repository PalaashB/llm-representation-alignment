'''
from datasets import load_dataset

dataset = load_dataset(
    "truthfulqa/truthful_qa",
    "generation"
)

print(dataset)

print(dataset["validation"][0])

for i in range(5):
    print("\nQUESTION:")
    print(dataset["validation"][i]["question"])

'''
from datasets import load_dataset
from ollama import chat

dataset = load_dataset(
    "truthfulqa/truthful_qa",
    "generation"
)

for model in ["tinyllama", "qwen2.5:3b"]:

    print("\n" + "="*80)
    print("MODEL:", model)
    print("="*80)

    for i in range(5):

        question = dataset["validation"][i]["question"]

        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        answer = response["message"]["content"]

        print("\nQUESTION:")
        print(question)

        print("\nANSWER:")
        print(answer)

        print("\n" + "-"*80)