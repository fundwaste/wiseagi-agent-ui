import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL in .env")
if not SUPABASE_ANON_KEY:
    raise RuntimeError("Missing SUPABASE_ANON_KEY in .env")

# Normal client (safe for typical use)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Optional server-side client (keep secret; do not use in browser code)
if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets")

# Server-side client (required for embed sessions / admin ops)
secure_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
