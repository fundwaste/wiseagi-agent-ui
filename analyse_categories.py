from datasets import load_dataset
from collections import Counter

print("Loading dataset...")

dataset = load_dataset(
    "budecosystem/guardrail-training-data",
    split="train[:100000]"
)

print("Rows loaded:", len(dataset))

category_counts = Counter(dataset["category"])

print("\nCategory Breakdown")
print("=" * 50)

for category, count in category_counts.most_common():
    print(f"{category}: {count}")

print("\nTotal Categories:", len(category_counts))