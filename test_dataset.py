from datasets import load_dataset

dataset = load_dataset(
    "budecosystem/guardrail-training-data",
    split="train[:1000]"
)

print(dataset)

print("\nFirst example:")
print(dataset[0])

print("\nAvailable columns:")
print(dataset.column_names)

print("\nCategories:")
categories = sorted(set(dataset["category"]))

for category in categories:
    print(category)

print("\nNumber of categories:", len(categories))