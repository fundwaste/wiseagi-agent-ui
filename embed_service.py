import os, time, hashlib
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import jwt
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
EMBED_SHARED_SECRET = os.getenv("EMBED_SHARED_SECRET")

sb = create_client(SUPABASE_URL, SERVICE_ROLE)
app = FastAPI()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class EmbedTokenRequest(BaseModel):
    external_user_id: str
    company_id: str
    subject: str
    year_group: str
    topic: str
    support_profile: Optional[List[str]] = None
    tier: str = "free"
    source: str = "school_app"

@app.post("/embed-token")
def embed_token(req: EmbedTokenRequest, x_app_key: str = Header(default="", convert_underscores=False)):
    if not x_app_key:
        raise HTTPException(401, "Missing x-app-key")

    # Look up the integration for this company
    row = (sb.table("company_integrations")
        .select("app_key_hash,is_active,allowed_sources,allowed_subjects,default_tier")
        .eq("company_id", req.company_id)
        .eq("integration_type", "react_native_webview")
        .maybe_single()
        .execute().data) or {}

    if not row or not row.get("is_active"):
        raise HTTPException(403, "Integration not active")

    if row.get("app_key_hash") != sha256_hex(x_app_key):
        raise HTTPException(403, "Invalid app key")

    if req.source not in (row.get("allowed_sources") or []):
        raise HTTPException(400, "Source not allowed")

    if req.subject not in (row.get("allowed_subjects") or []):
        raise HTTPException(400, "Subject not allowed")

    tier = req.tier or row.get("default_tier") or "free"

    now = int(time.time())
    exp = now + 3600  # 5 minutes

    payload = req.model_dump()
    payload["tier"] = tier
    payload.update({"iat": now, "exp": exp})

    token = jwt.encode(payload, EMBED_SHARED_SECRET, algorithm="HS256")
    return {"embed_token": token, "expires_in_seconds": exp - now}


@app.post("/api/v1/embed-url")
def embed_url(req: EmbedTokenRequest, x_app_key: str = Header(default=".", alias="x-app-key", convert_underscores=False)):
    if not x_app_key:
        raise HTTPException(401, "Missing x-app-key")

    # Look up the integration for this company
    try:
        row = (sb.table("company_integrations")
            .select("app_key_hash,is_active,allowed_sources,allowed_subjects,default_tier")
            .eq("company_id", req.company_id)
            .eq("integration_type", "react_native_webview")
            .maybe_single()
            .execute().data) or {}
    except Exception as e:
        print(e)
        raise HTTPException(401, "sad")

    if not row or not row.get("is_active"):
        raise HTTPException(403, "Integration not active")

    if row.get("app_key_hash") != sha256_hex(x_app_key):
        raise HTTPException(403, "Invalid app key")

    if req.source not in (row.get("allowed_sources") or []):
        raise HTTPException(400, "Source not allowed")

    if req.subject not in (row.get("allowed_subjects") or []):
        raise HTTPException(400, "Subject not allowed")

    tier = req.tier or row.get("default_tier") or "free"

    now = int(time.time())
    exp = now + 3600  # 5 minutes

    payload = req.model_dump()
    payload["tier"] = tier
    payload.update({"iat": now, "exp": exp})

    token = jwt.encode(payload, EMBED_SHARED_SECRET, algorithm="HS256")
    return {"url": f"http://localhost:8501/?embed_token={token}", "expires": exp}
    