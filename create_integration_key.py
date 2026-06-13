# app.py - WiseAGI Complete Rewrite with Cross-Agent Reasoning ✅
import os, secrets, hashlib
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SERVICE_KEY)

def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

company_id = input("Company UUID: ").strip()

raw_key = secrets.token_urlsafe(32)
key_hash = hash_key(raw_key)

supabase.table("company_integrations").upsert({
    "company_id": company_id,
    "integration_type": "react_native_webview",
    "app_key_hash": key_hash,
    "is_active": True,
    "allowed_sources": ["school_app"],
    "allowed_subjects": ["Maths","English","Science","Vocational","General"],
    "default_tier": "free",
}).execute()

print("\nGive this key to the app developer (store securely):\n")
print(raw_key)
print("\n(You will NOT be able to view it later; only the hash is stored.)")







