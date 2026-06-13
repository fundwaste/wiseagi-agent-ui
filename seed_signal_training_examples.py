from datasets import load_dataset
from supabase import create_client
from dotenv import load_dotenv
import os
import json

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Load first 1,000 rows only
dataset = load_dataset(
    "budecosystem/guardrail-training-data",
    split="train[:100000]"
)

MODULE_SIGNAL_MAP = {
    "benign": ("Wellbeing & Resilience", "low_belonging", "NORMAL"),

    "self_harm": ("Wellbeing & Resilience", "emotional_distress", "CRITICAL_SAFETY"),
    "hate_speech_offensive_language": ("Anti-Bullying & Respect", "bullying", "ELEVATED_REVIEW"),
    "discrimination_stereotype_injustice": ("Anti-Bullying & Respect", "bullying", "ELEVATED_REVIEW"),

    "child_abuse": ("Child Protection & Safeguarding", "abuse", "CRITICAL_SAFETY"),

    "terrorism_organized_crime": ("Counter-Extremism Intelligence", "extremism", "CRITICAL_SAFETY"),
    "violence_aiding_and_abetting_incitement": ("Counter-Extremism Intelligence", "extremism", "CRITICAL_SAFETY"),

    "privacy_violation": ("Digital Safety Intelligence", "privacy_risk", "ELEVATED_REVIEW"),
    "malware_hacking_cyberattack": ("Digital Safety Intelligence", "cyber_risk", "CRITICAL_SAFETY"),
    "jailbreak_prompt_injection": ("Digital Safety Intelligence", "prompt_injection", "ELEVATED_REVIEW"),
    "non_violent_unethical_behavior": ("Digital Safety Intelligence", "unsafe_ai_use", "ELEVATED_REVIEW"),
}

modules = supabase.table("intelligence_modules").select("id,module_name").execute().data or []
signals = supabase.table("signals_catalogue").select("id,signal_name").execute().data or []

module_lookup = {m["module_name"]: m["id"] for m in modules}
signal_lookup = {s["signal_name"]: s["id"] for s in signals}

rows = []

for item in dataset:
    category = item.get("category") or "unknown"

    module_name, signal_name, route = MODULE_SIGNAL_MAP.get(
        category,
        ("Digital Safety Intelligence", "unsafe_ai_use", "ELEVATED_REVIEW")
    )

    module_id = module_lookup.get(module_name)
    signal_id = signal_lookup.get(signal_name)

    if not module_id:
        print(f"Skipping row. Missing module: {module_name}")
        continue

    if not signal_id:
        print(f"Skipping row. Missing signal: {signal_name}")
        continue

    rows.append({
        "text": (item.get("text") or "")[:5000],
        "dataset_category": category,
        "module_id": module_id,
        "signal_id": signal_id,
        "mapped_route": route,
        "is_safe": item.get("is_safe"),
        "source": item.get("source"),
        "language": "en",
        "review_status": "pending",
        "raw_row": json.loads(json.dumps(item)),
        "embedding_status": "pending",
    })

print(f"Prepared {len(rows)} rows for insert.")

for i in range(0, len(rows), 100):
    batch = rows[i:i + 100]
    supabase.table("signal_training_examples").insert(batch).execute()
    print(f"Inserted {i + len(batch)} rows")

print("Done.")