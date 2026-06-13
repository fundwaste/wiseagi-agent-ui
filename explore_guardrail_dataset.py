from datasets import load_dataset
from collections import Counter

# Load a manageable sample first
dataset = load_dataset(
    "budecosystem/guardrail-training-data",
    split="train[:50000]"
)

print("Rows loaded:", len(dataset))

# --------------------------------------------------
# 1. Category counts
# --------------------------------------------------
category_counts = Counter(dataset["category"])

print("\nCategory counts:")
for category, count in category_counts.most_common():
    print(category, count)

# --------------------------------------------------
# 2. Safe vs unsafe counts
# --------------------------------------------------
safe_counts = Counter(dataset["is_safe"])

print("\nSafe vs Unsafe:")
for label, count in safe_counts.items():
    print(label, count)

# --------------------------------------------------
# 3. Terrorism / organised crime examples
# --------------------------------------------------
terrorism_rows = [
    row for row in dataset
    if row["category"] == "terrorism_organized_crime"
]

print("\nTerrorism / organised crime rows found:", len(terrorism_rows))

# --------------------------------------------------
# 4. Search for Islamist-extremism-related terms
# --------------------------------------------------
search_terms = [
    "isis",
    "isil",
    "daesh",
    "al qaeda",
    "al-qaeda",
    "alqaeda",
    "boko haram",
    "al shabaab",
    "al-shabaab",
    "taliban",
    "jihad",
    "mujahid",
    "mujahideen",
    "caliphate",
    "martyrdom",
    "shahid",
    "khilafah",
    "hijrah",
    "bayah",
    "pledge allegiance"
]

matched_rows = []

for row in terrorism_rows:
    text = row["text"].lower()

    if any(term in text for term in search_terms):
        matched_rows.append(row)

print("\nPotential Islamist-extremism-related matches:", len(matched_rows))

# --------------------------------------------------
# 5. Print examples for review
# --------------------------------------------------
print("\nSample matching examples:")

for i, row in enumerate(matched_rows[:50], start=1):
    print("\n" + "=" * 80)
    print("Example:", i)
    print("Category:", row["category"])
    print("Safe:", row["is_safe"])
    print("Source:", row["source"])
    print("Text:")
    print(row["text"][:1500])