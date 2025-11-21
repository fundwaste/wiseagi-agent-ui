# app.py - WiseAGI Complete Rewrite with Cross-Agent Reasoning ✅
import streamlit as st
from supabase_config import supabase
from supabase import create_client
from pymilvus import connections, Collection
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime
from openai import APIStatusError
from functools import lru_cache
import os, re, numpy as np
import tiktoken
import uuid

# --------- 1. Setup Connections ---------
connections.connect(
    alias="default",
    uri="https://in03-357b70cf3851670.serverless.gcp-us-west1.cloud.zilliz.com",
    token="ce5060c7939d564fa7ae65d5c85cad6462b6b6fe5b0a8afc6216c7e3bd80da0aeb3ed4688157c2af9a36ddd30bc5838f9f53d880"
)

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=api_key)

# --- Provider selection for AGENT TURNS (not consensus) ---
AGENT_PROVIDER = os.getenv("AGENT_PROVIDER", "deepseek").lower()  # "deepseek" | "grok"

# DeepSeek (OpenAI-compatible) client agent agent turns
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") 
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # override if you prefer

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_MODEL   = os.getenv("XAI_MODEL", "grok-4-fast-reasoning")
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1") if XAI_API_KEY else None

SUPABASE_URL = "https://bkxphaekghgcgfzwbnqc.supabase.co"
SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJreHBoYWVrZ2hnY2dmendibnFjIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDU3NTk1OCwiZXhwIjoyMDYwMTUxOTU4fQ.AgX9_99opI-UphOJVzLXM1huPz0adXiOmI34s7YwPuQ"  
secure_supabase = create_client(SUPABASE_URL, SERVICE_ROLE_KEY)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --------- 2. Helper Functions ---------
# ---- Create 768-dim Arabic collection once ----
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

if not utility.has_collection("all_agents_arabic"):
    fields = [
        FieldSchema(name="id",       dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source",   dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="vector",   dtype=DataType.FLOAT_VECTOR, dim=768),
    ]
    schema = CollectionSchema(fields, description="Arabic 768-dim")
    coll = Collection("all_agents_arabic", schema)
    coll.create_index("vector", {"index_type":"IVF_FLAT","metric_type":"COSINE","params":{"nlist":2048}})
    coll.load()

# ---- Milvus search helpers (Sprint 0) ----
def _safe_ent_get(ent: dict, key: str, default=""):
    # pymilvus returns entity as a dict-like object
    try:
        if isinstance(ent, dict):
            return ent.get(key, default)
        # some builds expose attributes
        return getattr(ent, key, default)
    except Exception:
        return default

def create_role_based_prompt(agent_role, default_summary, agent_speciality, sources):
    citation = f"\nSources: {sources}" if sources else ""
    if agent_role == "default":
        return f"{agent_speciality}: Summarise this question with supporting details:\n{default_summary}{citation}"
    elif agent_role == "devil_advocate":
        return (
            f"{agent_speciality} (Devil's Advocate): Provide a different perspective to make the user think laterally."
        )
    elif agent_role == "pragmatic":
        return (
            f"{agent_speciality} (Pragmatic): Review this summary practically and provide a grounded viewpoint:\n{default_summary}{citation}"
        )
    elif agent_role == "supportive":
        return (
            f"{agent_speciality} (Supportive): Review this summary neutrally and offer a balanced interpretation:\n{default_summary}{citation}"
        )
    else:
        return f"{agent_speciality}: Summarise or comment on:\n{default_summary}{citation}"

def fetch_hits_for_agent(col: Collection, query_vec, agent_id: str, top_k: int = 20):
    """
    Returns a list of dicts: {text, source, agent_id, distance}
    Assumes collection has output fields: Text, Source, agent_id
    """
    results = col.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["Text", "agent_id", "Source"],
    )
    if not results or not results[0]:
        return []

    out = []
    for hit in results[0]:
        ent = hit.entity  # dict-like
        if _safe_ent_get(ent, "agent_id") != agent_id:
            continue
        out.append({
            "text": _safe_ent_get(ent, "Text"),
            "source": _safe_ent_get(ent, "Source"),
            "agent_id": _safe_ent_get(ent, "agent_id"),
            "distance": getattr(hit, "distance", 0.0),
        })
    return out

def generate_embedding(text):
    return model.encode([text])[0].tolist()

def _is_arabic(s: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", s or ""))

AR_EMBED_MODEL = os.getenv(
    "AR_EMBED_MODEL",
    "Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka"
)

@lru_cache(maxsize=1)
def _load_ar_model():
    return SentenceTransformer(AR_EMBED_MODEL)

def embed_arabic_768(text: str) -> np.ndarray:
    t = (text or "").strip()
    if not t:
        return np.zeros((768,), dtype=np.float32)
    vec = _load_ar_model().encode([t], normalize_embeddings=True)[0]
    # Matryoshka models can be safely truncated; ensure exactly 768 dims
    vec = vec[:768] if len(vec) >= 768 else vec
    return np.asarray(vec, dtype=np.float32)

# ---- Model/collection registry (add more rows later) ----
EMBED_REGISTRY = [
    {
        "name": "multi384",
        "dim": 384,
        "collection": "all_agents",
        "embed": lambda txt: np.asarray(generate_embedding(txt), dtype=np.float32),
    },
    {
        "name": "arabic768",
        "dim": 768,
        "collection": "all_agents_arabic",
        "embed": embed_arabic_768,
    },
    # add more models here when needed…
]

def _to_sim(distance: float) -> float:
    try: return 1.0 - float(distance)
    except: return 0.0

def _norm(hits):
    if not hits: return []
    sims = [_to_sim(h.get("distance", 1.0)) for h in hits]
    lo, hi = min(sims), max(sims); span = (hi - lo) or 1e-9
    out=[]
    for h,s in zip(hits,sims):
        hh=dict(h); hh["_sim"]=s; hh["_norm"]=(s-lo)/span; out.append(hh)
    return out

def fetch_hits_multi(agent_id: str, query_text: str, top_k: int = 20, agent_lang: str | None = None):
    """
    Query ALL registered collections (mixed dims), normalise, lightly weight, merge, return top_k.
    """
    prefer_ar = _is_arabic(query_text) or (agent_lang == "ar")
    merged = []

    for m in EMBED_REGISTRY:
        vec = m["embed"](query_text)           # produces exactly m["dim"]
        col = Collection(m["collection"])
        hits = _norm(fetch_hits_for_agent(col, vec, agent_id, top_k=top_k))
        # light prior: nudge Arabic when query/agent is Arabic
        w = 1.10 if (prefer_ar and "arabic" in m["name"]) else 1.00
        for h in hits:
            hh = dict(h)
            hh["_rank_score"] = h["_norm"] * w
            hh["_collection"] = m["collection"]
            merged.append(hh)

    merged.sort(key=lambda x: x["_rank_score"], reverse=True)
    return merged[:top_k]

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def estimate_cost(prompt_tokens, completion_tokens, model="gpt-3.5-turbo"):
    if model == "gpt-3.5-turbo":
        return (prompt_tokens * 0.0015 + completion_tokens * 0.002) / 1000
    elif model == "gpt-4o":
        return (prompt_tokens * 0.005 + completion_tokens * 0.015) / 1000
    return 0.0

def model_router(purpose: str = "default") -> str:
    if purpose == "peer_review":
        return "gpt-3.5-turbo"
    elif purpose == "consensus":
        return "gpt-4o"
    return "gpt-3.5-turbo"

def query_openai_context(prompt, context, purpose="default", personality=None):
    model = model_router(purpose)
    full_input = f"{prompt}\n\n{context}"
    prompt_tokens = count_tokens(full_input, model)

    sys = "You are a helpful and knowledgeable assistant."
    if personality:
        sys = (
            f"You are a helpful assistant. Adopt a {personality.get('tone','Balanced')} tone "
            f"and a {personality.get('style','Concise').lower()} style. "
            f"Prefer a maximum of {personality.get('max_words',180)} words. "
            f"{'When possible, cite sources provided by the user.' if personality.get('citations', True) else ''}"
        )

    response = openai_client.chat.completions.create(
        model=model,
        max_tokens=350,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": full_input}
        ]
    )

    answer = response.choices[0].message.content.strip()
    completion_tokens = count_tokens(answer, model)
    total_tokens = prompt_tokens + completion_tokens
    cost = estimate_cost(prompt_tokens, completion_tokens, model)
    return answer, total_tokens, cost

def query_agent_context(prompt: str, context: str, personality=None):
    """
    Agent turns only. Uses provider set by AGENT_PROVIDER (DeepSeek or Grok).
    Falls back to OpenAI helper if provider is unavailable or errors.
    Returns (answer, tokens_est, cost_est).
    """
    # Allow runtime override (optional) via Streamlit session
    provider = (st.session_state.get("agent_provider") if "st" in globals() and hasattr(st, "session_state") else None) or AGENT_PROVIDER

    sys = "You are a helpful domain expert."
    if personality:
        sys = (
            f"You are a helpful assistant. Adopt a {personality.get('tone','Balanced')} tone "
            f"and a {personality.get('style','Concise').lower()} style. "
            f"Prefer a maximum of {personality.get('max_words',180)} words. "
            f"{'When possible, cite sources provided by the user.' if personality.get('citations', True) else ''}"
        )

    full_input = f"{prompt}\n\n{context}"
    prompt_tokens = count_tokens(full_input, model="gpt-3.5-turbo")

    def _call(client, model):
        resp = client.chat.completions.create(
            model=model,
            max_tokens=350,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": full_input},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        completion_tokens = count_tokens(answer, model="gpt-3.5-turbo")
        return answer, (prompt_tokens + completion_tokens), 0.0  # keep £0 for non-OpenAI in MVP

    try:
        if provider == "grok" and grok_client:
            return _call(grok_client, XAI_MODEL)
        if provider == "deepseek" and deepseek_client:
            return _call(deepseek_client, DEEPSEEK_MODEL)
        # If provider not configured, fall back to OpenAI
        return query_openai_context(prompt, context, purpose="peer_review", personality=personality)

    except APIStatusError:
        # e.g. 402 Insufficient Balance or other API errors → fallback
        return query_openai_context(prompt, context, purpose="peer_review", personality=personality)
    except Exception:
        return query_openai_context(prompt, context, purpose="peer_review", personality=personality)

def deduplicate_agents(agent_list):
    seen = set()
    unique = []
    for agent in agent_list:
        if agent['id'] not in seen:
            unique.append(agent)
            seen.add(agent['id'])
    return unique

# === Agent governance helpers (MVP, no schema changes) ===

def _resp_data(resp):
    # Works whether the client returns an object with .data or a plain dict
    try:
        return resp.data
    except AttributeError:
        return resp

def is_project_owner(supabase, project_id, user_id):
    row = (supabase.table("project_members")
           .select("role")
           .eq("project_id", project_id)
           .eq("user_id", user_id)
           .maybe_single()
           .execute().data)
    return (row or {}).get("role") == "owner"

def get_member_role(supabase, project_id, user_id):
    row = (supabase.table("project_members")
           .select("role")
           .eq("project_id", project_id)
           .eq("user_id", user_id)
           .maybe_single()
           .execute().data)
    return (row or {}).get("role")

def get_allowed_agent_ids(supabase, project_id):
    rows = (supabase.table("project_agents")
            .select("agent_id")
            .eq("project_id", project_id)
            .execute().data or [])
    return {r["agent_id"] for r in rows}

def _public_agent_ids():
    rows = supabase.table("agents").select("id").eq("is_public", True).execute().data or []
    return {r["id"] for r in rows}

def assert_agents_allowed(supabase, project_id, requested_ids, plan=None):
    requested = set(requested_ids or [])
    public_ids = _public_agent_ids()

    # If no project context: allow only public
    if not project_id:
        blocked = requested - public_ids
        if blocked:
            raise PermissionError("On this screen you can use public agents only.")
        return list(requested & public_ids)

    # In a project: allow project agents + public
    allowed = get_allowed_agent_ids(supabase, project_id) | public_ids
    blocked = requested - allowed
    if blocked:
        raise PermissionError("One or more agents are not enabled for this project.")
    return list(requested & allowed)

def get_or_create_project_log_conversation(supabase_client, project_id):
    title = "__PROJECT_LOG__"
    # Try to find existing
    resp = (supabase_client.table("mvp_conversations")
            .select("id")
            .eq("project_id", project_id)
            .eq("title", title)
            .maybe_single()
            .execute())
    row = getattr(resp, "data", None) or {}
    if isinstance(row, dict) and row.get("id"):
        return row["id"]
    if isinstance(row, list) and row:
        return row[0].get("id")

    # Create (no .select() chaining in Python)
    resp = (supabase_client.table("mvp_conversations")
            .insert({"project_id": project_id, "title": title, "created_by": "system"})
            .execute())
    created = getattr(resp, "data", None) or {}
    if isinstance(created, list) and created:
        return created[0].get("id")
    if isinstance(created, dict):
        return created.get("id")
    return None

def log_project_event(supabase_client, project_id, text, meta=None):
    conv_id = get_or_create_project_log_conversation(supabase_client, project_id)
    if not conv_id:
        return
    payload = {"conversation_id": conv_id, "user_id": "system", "content": text, "author_type": "system"}
    if meta is not None:
        payload["meta"] = meta
    try:
        supabase_client.table("mvp_messages").insert(payload).execute()
    except Exception:
        payload.pop("meta", None)
        supabase_client.table("mvp_messages").insert(payload).execute()

def set_project_agent_enabled(supabase, project_id, user_id, agent_id, enabled):
    if not is_project_owner(supabase, project_id, user_id):
        raise PermissionError("Only the project owner can modify project agents.")
    if enabled:
        (supabase.table("project_agents")
         .upsert({"project_id": project_id, "agent_id": agent_id}, on_conflict="project_id,agent_id")
         .execute())
        log_project_event(secure_supabase, project_id, f"Enabled agent {agent_id}", {"agent_id": agent_id, "enabled": True})
    else:
        (supabase.table("project_agents")
         .delete()
         .eq("project_id", project_id)
         .eq("agent_id", agent_id)
         .execute())
        log_project_event(secure_supabase, project_id, f"Disabled agent {agent_id}", {"agent_id": agent_id, "enabled": False})

def render_agent_admin_panel(supabase, project_id, user_id, all_agents):
    """
    Owner-only: enable/disable agents for this project.
    Others: read-only list of enabled agents.
    """
    if not project_id:
        return

    enabled_ids = get_allowed_agent_ids(supabase, project_id)

    if not is_project_owner(supabase, project_id, user_id):
        st.sidebar.markdown("**Agents enabled for this project**")
        for a in all_agents:
            if a["id"] in enabled_ids:
                st.sidebar.checkbox(a["agent_name"], value=True, disabled=True, key=f"show_agent_{a['id']}")
        return

    st.sidebar.markdown("**Manage agents (owner only)**")
    for a in all_agents:
        tick = st.sidebar.checkbox(a["agent_name"], value=(a["id"] in enabled_ids), key=f"admin_agent_{a['id']}")
        if tick and a["id"] not in enabled_ids:
            set_project_agent_enabled(supabase, project_id, user_id, a["id"], True)
            enabled_ids.add(a["id"])
        elif (not tick) and a["id"] in enabled_ids:
            set_project_agent_enabled(supabase, project_id, user_id, a["id"], False)
            enabled_ids.remove(a["id"])

def render_agent_picker(supabase, project_id, all_agents):
    """
    Participant-side picker: shows only agents enabled for the project.
    Returns the list of chosen agent dicts for this turn.
    """
    chosen = []
    if not project_id:
        return chosen
    allowed_ids = get_allowed_agent_ids(supabase, project_id)
    st.sidebar.subheader("Agents for this project")
    for a in all_agents:
        if a["id"] in allowed_ids:
            if st.sidebar.checkbox(f"{a['agent_name']} — {a.get('description','')}", key=f"use_agent_{a['id']}"):
                chosen.append(a)
        else:
            st.sidebar.checkbox(f"{a['agent_name']} (not enabled)", value=False, disabled=True, key=f"disabled_agent_{a['id']}")
    return chosen

def facilitator_gate(supabase, project_id, user_id, pending_agent_ids, question):
    role = get_member_role(supabase, project_id, user_id)
    if role not in {"owner", "facilitator"}:
        return question
    st.markdown("**Facilitator controls**")
    refined = st.text_area("Refine or guide the question (optional):", value=question, key="facilitator_q")
    approve = st.checkbox("Approve selected agents for this turn", value=True, key="facilitator_ok")
    if not approve:
        st.stop()
    return refined

# ---------- Sprint 3: conversations helpers ----------
from typing import List, Dict, Any

def ensure_membership(conversation_id: str, user_id: str):
    # MVP: skip membership while conversations live in mvp_conversations
    return

# use service client to avoid RLS
def _users_by_id(uids: list[str]) -> dict[str, str]:
    if not uids:
        return {}
    rows = (
        secure_supabase.table("Users")
        .select("id,email")
        .in_("id", uids)
        .execute()
        .data or []
    )
    return {r["id"]: r["email"] for r in rows}

def create_conversation(title: str, project_id: str, company_id: str, created_by: str) -> Dict[str, Any]:
    conv = secure_supabase.table("mvp_conversations").insert({
        "title": (title or "Conversation").strip()[:120],
        "project_id": project_id,
        "company_id": company_id,
        "created_by": created_by,
        # created_at is defaulted by DB
    }).execute().data[0]
    # keep membership in a separate table if you want, or skip for MVP
    return conv

def list_conversations_for_project(project_id: str, user_id: str) -> list[dict]:
    return (
        secure_supabase.table("mvp_conversations")
        .select("id,title,created_at,project_id,company_id,created_by")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .execute()
        .data or []
    )

def post_user_message(conversation_id: str, user_id: str, text: str, meta: dict | None = None):
    return secure_supabase.table("mvp_messages").insert({
        "conversation_id": conversation_id,
        "author_id": user_id,
        "author_type": "user",
        "content": text,
        "meta": meta or {},
        "created_at": datetime.utcnow().isoformat(),
    }).execute().data[0]

def post_agent_message(conversation_id: str, label: str, content: str, meta: dict | None = None):
    secure_supabase.table("mvp_messages").insert({
        "conversation_id": conversation_id,
        "author_id": None,
        "author_type": "agent",
        "content": f"[{label}] {content}",
        "meta": meta or {},
        "created_at": datetime.utcnow().isoformat(),
    }).execute()

def list_messages(conversation_id: str, limit: int = 300):
    return (
        secure_supabase.table("mvp_messages")
        .select("id, author_id, author_type, content, meta, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at")     # ascending is default; or use desc=True for newest first
        .limit(limit)
        .execute()
        .data or []
    )

# ---- Projects minimal (Sprint 2) ----
def list_projects(company_id: str):
    res = secure_supabase.table("projects").select("id,name").eq("company_id", company_id).execute()
    return res.data or []

def assert_project_limit(supabase, company_id, plan):
    if plan == "free":
        raise PermissionError("Projects require a paid plan.")
    if plan == "team":
        res = supabase.table("projects").select("id", count="exact").eq("company_id", company_id).execute()
        current = getattr(res, "count", None) or len(res.data or [])
        if current >= 1:
            raise PermissionError("Team plan includes 1 project. Upgrade for more.")

# At the top of create_project(...)
def create_project(company_id, user_id, name, desc=""):
    plan = get_user_plan(user_id)
    assert_project_limit(secure_supabase, company_id, plan)
    pr = (
    secure_supabase
    .table("projects")
    .insert({
        "company_id": company_id,
        "name": name,
        "description": desc,
        "created_by": user_id
    })
    .execute()
    .data[0]
)
    secure_supabase.table("project_members").insert({
        "project_id": pr["id"],
        "user_id": user_id,
        "role": "owner"
    }).execute()

    return pr

def get_project_agents(project_id: str):
    res = supabase.table("project_agents").select("agent_id, agents(id,agent_name,description,collection_name)") \
        .eq("project_id", project_id).execute()
    return [r["agents"] for r in res.data if r.get("agents")]

def get_all_agents():
    return supabase.table("agents").select("id, agent_name, description, collection_name").execute().data or []

def get_user_agents(user_id):
    join = supabase.table("user_agents").select("agent_id, agents(id, agent_name, description, collection_name)").eq("user_id", user_id).execute()
    return deduplicate_agents([r["agents"] for r in join.data if r.get("agents")])

def load_user_personality(user_id: str, company_id: str):
    try:
        res = supabase.table("user_personalities") \
            .select("personality") \
            .eq("user_id", user_id) \
            .maybe_single() \
            .execute()
        data = getattr(res, "data", None)
        if not data:
            raise ValueError("No personality row")
        # if personality column is NULL, return default
        personality = data.get("personality") or {}
    except Exception:
        personality = {}

    # defaults
    return {
        "tone": personality.get("tone", "Balanced"),
        "style": personality.get("style", "Concise"),
        "citations": personality.get("citations", True),
        "max_words": personality.get("max_words", 180),
    }

def save_user_personality(user_id: str, company_id: str, pers: dict):
    supabase.table("user_personalities").upsert({
        "user_id": user_id,
        "company_id": company_id,
        "personality": pers,
    }).execute()

def save_user_agent(user_id, agent_id):
    supabase.table("user_agents").insert({"user_id": user_id, "agent_id": agent_id}).execute()

def upload_document_to_agent(file, agent_id):
    fname = f"{agent_id}/{file.name}"
    supabase.storage.from_("agent_documents").upload(fname, file)
    supabase.table("agent_documents").insert({"agent_id": agent_id, "file_name": file.name, "uploaded_at": datetime.utcnow().isoformat()}).execute()

# --------- 3. Authentication ---------
def login_page():
    st.title("🔐 AI Platform Account")
    mode = st.radio("Action", ["Login", "Sign up"])
    email = st.text_input("Email")
    pwd   = st.text_input("Password", type="password")

    if mode == "Sign up":
        comps = supabase.table("companies").select("id,name,logo_url").execute().data or []
        names = [c["name"] for c in comps]
        choice = st.selectbox("Company", ["(new)"] + names)
        new_name = st.text_input("New company name") if choice == "(new)" else None
        new_logo = st.file_uploader("Company logo (jpg)", type="jpg") if choice == "(new)" else None

        if st.button("Create account"):
            if new_name:
                up = {"name": new_name}
                if new_logo:
                    key = f"logos/{uuid.uuid4()}_{new_logo.name}"
                    file_bytes = new_logo.read()
                    secure_supabase.storage.from_("company-logos").upload(key, file_bytes)
                    up["logo_url"] = secure_supabase.storage.from_("company-logos").get_public_url(key)

                existing_company = supabase.table("companies").select("*").eq("name", up["name"]).execute().data
                company = existing_company[0] if existing_company else supabase.table("companies").insert(up).execute().data[0]
            else:
                company = next(c for c in comps if c["name"] == choice)

            res = supabase.auth.sign_up({"email": email, "password": pwd})
            user_id = res.user.id

            supabase.table("Users").update({
                "company_id": company["id"],
                "email": email.lower(),
                "is_admin": False,
                "is_agent": False,
                "plan": "free",
            }).eq("id", user_id).execute()

            # (Optional) if sign_up returns a session for your config, persist it
            if getattr(res, "session", None):
                st.session_state["sb_access_token"]  = res.session.access_token
                st.session_state["sb_refresh_token"] = res.session.refresh_token

            st.session_state["user"] = {
                "id": user_id,
                "email": email.lower(),
                "company_id": company["id"],
                "is_admin": False
            }
            st.rerun()

    else:  # Login
        if st.button("Login"):
            res = supabase.auth.sign_in_with_password({"email": email, "password": pwd})
            if not res.session:
                st.error("Invalid credentials")
                st.stop()

            # ✅ persist tokens for future reruns
            st.session_state["sb_access_token"]  = res.session.access_token
            st.session_state["sb_refresh_token"] = res.session.refresh_token

            uid = res.user.id
            row = (supabase.table("Users")
                   .select("company_id,is_admin")
                   .eq("id", uid)
                   .maybe_single()
                   .execute().data or {})

            st.session_state["user"] = {
                "id": uid,
                "email": email.lower(),
                "company_id": row.get("company_id"),
                "is_admin": row.get("is_admin", False),
            }
            st.rerun()

def restore_supabase_session():
    at = st.session_state.get("sb_access_token")
    rt = st.session_state.get("sb_refresh_token")
    if at and rt:
        try:
            supabase.auth.set_session(at, rt)
        except Exception:
            # Tokens expired or invalid — clear them to force a fresh login
            st.session_state.pop("sb_access_token", None)
            st.session_state.pop("sb_refresh_token", None)

# --- Auth bootstrap (place this AFTER login_page() and restore_supabase_session() defs) ---
restore_supabase_session()
sess = supabase.auth.get_session()
if not (sess and sess.user):
    st.session_state.pop("user", None)
    login_page()
    st.stop()

# --------- 4. Sidebar UI ---------
# --- Auth guard (prevents KeyError after logout) ---
user = st.session_state.get("user")
if not user:
    # Show your existing login UI / function until a user signs in.
    # If you already have a login_page() that returns a user dict, use it here:
    try:
        user = login_page()  # <-- use your real login function
    except NameError:
        # If you don't have a function that returns the user, just stop here
        st.info("Please sign in to continue.")
        st.stop()

    if not user:
        st.stop()  # stop rendering until login_page sets/returns a user

    # Persist to session for next rerun
    st.session_state["user"] = user

comp = supabase.table("companies") \
               .select("name,logo_url") \
               .eq("id", user["company_id"]) \
               .maybe_single() \
               .execute().data or {}

# Pick logo (fallback to local image)
logo_url = comp.get("logo_url") or "RABIIT.jpg"

# Logo: half-width + black border
st.sidebar.markdown(
    f"""
    <div style="text-align:left; margin:8px 0;">
        <img src="{logo_url}" alt="Company logo"
             style="width:50%; border:2px solid #000; border-radius:6px;" />
    </div>
    """,
    unsafe_allow_html=True,
)

# Plain text (no link), Calibri 12pt
st.sidebar.markdown(
    f"""
    <div style="font-family: Calibri, Arial, sans-serif; font-size:12pt; line-height:1.35;">
        <div>👤 {user['email']}</div>
        <div><strong>Company:</strong> {comp.get('name','–')}</div>
        <div><strong>Admin:</strong> {str(user.get('is_admin', False))}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("🚪 Logout"):
    # Clear only the keys you actually set during a session
    for k in ("user", "company_id", "project_id", "conversation_id", "chosen_agents"):
        st.session_state.pop(k, None)
    st.rerun()

with st.sidebar.expander("LLM (agents) — developer toggle", expanded=False):
    current = st.session_state.get("agent_provider", AGENT_PROVIDER)
    choice = st.radio("Provider", ["deepseek", "grok"], index=0 if current=="deepseek" else 1, horizontal=True)
    if choice != current:
        st.session_state["agent_provider"] = choice
        st.toast(f"Agent provider set to: {choice}")
        st.rerun()
        
# ------------------ Personality Editor -------------------

st.sidebar.markdown("---")
st.sidebar.subheader("🧭 My Personality")

_curr = load_user_personality(user["id"], user["company_id"])
tone  = st.sidebar.selectbox("Tone", ["Balanced","Analytical","Strategic","Supportive","Challenger"],
                             index=["Balanced","Analytical","Strategic","Supportive","Challenger"].index(_curr.get("tone","Balanced")))
style = st.sidebar.selectbox("Style", ["Concise","Detailed"], index=0 if _curr.get("style","Concise")=="Concise" else 1)
cit   = st.sidebar.checkbox("Include sources/citations", value=bool(_curr.get("citations", True)))
mxw   = st.sidebar.slider("Max words", min_value=80, max_value=500, value=int(_curr.get("max_words", 180)), step=10)

if st.sidebar.button("💾 Save personality"):
    save_user_personality(user["id"], user["company_id"], {
        "tone": tone, "style": style, "citations": cit, "max_words": mxw
    })
    st.sidebar.success("Saved.")

# --------- Admin Panel Updated: Agent Role Prompts ---------

def admin_page():
    st.title("🛠️ Admin Panel")
    st.markdown("Manage agents, their descriptions, and role-based prompt templates.")

    tab1, tab2 = st.tabs(["Manage Agents", "Role Prompts"])

    # --- TAB 1: Agent Description ---
    with tab1:
        st.subheader("✍️ Edit Agent Descriptions")
        agents = get_all_agents()
        agent_dict = {a['agent_name']: a['id'] for a in agents}
        selected_name = st.selectbox("Select Agent", list(agent_dict.keys()))
        selected_id = agent_dict[selected_name]

        # Fetch agent detail
        agent_detail = supabase.table("agents").select("description").eq("id", selected_id).single().execute().data
        desc = st.text_area("Agent Description", value=agent_detail.get("description", ""))

        if st.button("Update Description"):
            supabase.table("agents").update({"description": desc}).eq("id", selected_id).execute()
            st.success("Agent description updated.")

    # --- TAB 2: Role Prompts ---
    with tab2:
        st.subheader("🎭 Edit Role Prompts per Agent")

        selected_name = st.selectbox("Select Agent for Roles", list(agent_dict.keys()), key="agent_roles")
        selected_id = agent_dict[selected_name]

        prompts = supabase.table("agent_role_prompts") \
            .select("id, role, prompt_template, is_active") \
            .eq("agent_id", selected_id).execute().data or []

        st.markdown("### 📋 Existing Role Prompts")
        if not prompts:
            st.info("No prompts defined for this agent yet.")
        for entry in prompts:
            with st.expander(f"🧩 Role: {entry['role']}"):
                new_template = st.text_area("Prompt Template", value=entry["prompt_template"], key=f"template_{entry['id']}")
                new_status = st.checkbox("Active", value=entry["is_active"], key=f"active_{entry['id']}")
                if st.button("Save", key=f"save_{entry['id']}"):
                    supabase.table("agent_role_prompts") \
                        .update({"prompt_template": new_template, "is_active": new_status}) \
                        .eq("id", entry["id"]).execute()
                    st.success("Prompt updated.")

        st.markdown("---")
        st.subheader("➕ Add New Role Prompt")
        role_name = st.text_input("Role Name")
        prompt_text = st.text_area("Prompt Template")
        if st.button("Add Prompt"):
            if role_name and prompt_text:
                supabase.table("agent_role_prompts").insert({
                    "agent_id": selected_id,
                    "role": role_name,
                    "prompt_template": prompt_text,
                    "is_active": True
                }).execute()
                st.success("Prompt added.")
                st.rerun()
            else:
                st.warning("Both fields are required.")

if st.session_state.get("user", {}).get("is_admin", False):
    if st.sidebar.checkbox("🔐 Admin Mode"):
        admin_page()
        st.stop()

st.sidebar.markdown("---")

# ---- Plan helper (top-level helpers area is fine) ----
def get_user_plan(user_id):
    row = (supabase.table("Users").select("plan").eq("id", user_id).maybe_single().execute().data) or {}
    return (row.get("plan") or "free").lower()

# --- Plan & upgrade ---
plan = get_user_plan(user["id"]).lower()
is_paid = plan in ("team", "enterprise")
st.sidebar.caption(f"Plan: {plan.title()}")

# Show upgrade CTAs only when on Free
if not is_paid:
    st.sidebar.info("Projects are available on paid plans.")
    st.sidebar.link_button("Upgrade to Team", os.getenv("STRIPE_TEAM_LINK", "#"))
    st.sidebar.link_button("Upgrade to Enterprise", os.getenv("STRIPE_ENT_LINK", "#"))

# ---- Developer test (temporary) ----
# Show plan toggles on ALL plans, but only for admins (or set DEV_MODE=1 to always show)
show_dev = (str(os.getenv("DEV_MODE", "1")) == "1") or bool(user.get("is_admin"))
if show_dev:
    with st.sidebar.expander("Developer test (temporary)"):
        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("Free", key="dev_plan_free"):
                # use secure_supabase if your RLS blocks writes for normal client
                (secure_supabase if 'secure_supabase' in globals() else supabase) \
                    .table("Users").update({"plan": "free"}).eq("id", user["id"]).execute()
                st.info("Plan set to Free."); st.rerun()
        with colB:
            if st.button("Team", key="dev_plan_team"):
                (secure_supabase if 'secure_supabase' in globals() else supabase) \
                    .table("Users").update({"plan": "team"}).eq("id", user["id"]).execute()
                st.success("Plan set to Team."); st.rerun()
        with colC:
            if st.button("Enterprise", key="dev_plan_ent"):
                (secure_supabase if 'secure_supabase' in globals() else supabase) \
                    .table("Users").update({"plan": "enterprise"}).eq("id", user["id"]).execute()
                st.success("Plan set to Enterprise."); st.rerun()

    # Optional: simple public-agent picker for Free users
    pubs = supabase.table("agents").select("id,agent_name,description").eq("is_public", True).execute().data or []
    st.sidebar.header("Public agents")
    free_selected = []
    for a in pubs:
        if st.sidebar.checkbox(f"{a['agent_name']} — {a.get('description','')}", key=f"pub_{a['id']}"):
            free_selected.append(a["id"])

if is_paid:
# ---- after computing plan/is_paid and (if free) showing the public agents list ----
    st.sidebar.header("📁 Project")

    projects = list_projects(user["company_id"])
    proj_options = ["(none)"] + [p["name"] for p in projects]
    sel = st.sidebar.selectbox("Select project", proj_options)
    project_id = None
    if sel != "(none)":
        project_id = next(p["id"] for p in projects if p["name"] == sel)

    # Reset the open conversation if the project changed
    if project_id and st.session_state.get("project_id") != project_id:
        st.session_state["project_id"] = project_id
        st.session_state.pop("conversation_id", None)

    with st.sidebar.expander("New project"):
        p_name = st.text_input("Project name", key="newproj_name")
        p_desc = st.text_area("Description", key="newproj_desc")
        if st.button("Create project"):
            if not p_name.strip():
                st.warning("Name required.")
            else:
                try:
                    pr = create_project(user["company_id"], user["id"], p_name.strip(), p_desc.strip())
                    st.success(f'Project “{pr["name"]}” created.')
                    st.session_state["project_id"] = pr["id"]
                    st.rerun()
                except PermissionError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Could not create project: {e}")

    # --------- Sidebar: Current Agents (paid) ---------
    proj_agents = get_project_agents(project_id) if project_id else []
    proj_agent_ids = {a["id"] for a in proj_agents}

    st.sidebar.markdown("---")
    st.sidebar.header("Your Current Agents")

    render_agent_admin_panel(supabase, project_id, user["id"], get_all_agents())
    chosen_agents = render_agent_picker(supabase, project_id, get_user_agents(user["id"]))
    selected_agent_ids = [a["id"] for a in chosen_agents]

    st.sidebar.markdown("---")
    st.sidebar.header("Add New Agents")
    existing_agents = get_all_agents()
    query = st.sidebar.text_input("Search for agent expertise:")
    filtered = [a for a in existing_agents if query.lower() in a['description'].lower()] if query else existing_agents
    available = {f"{a['agent_name']} - {a['description']}": a['id'] for a in filtered if a['id'] not in selected_agent_ids}
    labels = st.sidebar.multiselect("Add agents to your portfolio", options=list(available.keys()), max_selections=5)
    if labels:
        for label in labels:
            save_user_agent(st.session_state['user']['id'], available[label])
        st.sidebar.success("✅ Agents added.")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Create a New Agent")
    with st.sidebar.form("new_agent_form", clear_on_submit=True):
        new_name = st.text_input("Agent Name")
        new_desc = st.text_area("Agent Description")
        uploaded_files = st.file_uploader("Upload supporting documents (optional)", accept_multiple_files=True)
        submitted = st.form_submit_button("Save Agent")
        if submitted:
            if not new_name or not new_desc:
                st.warning("Please provide both an agent name and description.")
            else:
                res = supabase.table("agents").insert({
                    "agent_name": new_name,
                    "description": new_desc,
                    "user_id": st.session_state['user']['id'],
                    "collection_name": new_name.replace(" ", "_").lower(),
                    "is_public": True
                }).execute()
                if res.data:
                    agent_id = res.data[0]['id']
                    for f in uploaded_files or []:
                        upload_document_to_agent(f, agent_id)
                    save_user_agent(st.session_state['user']['id'], agent_id)
                    st.sidebar.success("🎉 New agent created and added to your portfolio!")
                    st.rerun()
                else:
                    st.sidebar.error("⚠️ Could not create agent.")
else:
    # FREE plan: selected public agents from earlier
    selected_agent_ids = free_selected
    project_id = None

# --------- Sprint 3: Conversations tab (PAID only) ---------
if is_paid:
    tab_proj, tab_convos = st.tabs(["Project", "Conversations"])

    # ---------------- Project tab ----------------
    with tab_proj:
        if not project_id:
            st.info("Select a project in the sidebar to see its overview.")
        else:
            # --- Basic project info ---
            pr = (
                secure_supabase
                .table("projects")
                .select("id,name,description,created_at")
                .eq("id", project_id)
                .maybe_single()
                .execute()
                .data or {}
            )

            st.subheader(pr.get("name", "Project"))
            st.caption(f"Created: {pr.get('created_at','—')}")
            if pr.get("description"):
                st.write(pr["description"])
            else:
                st.info("No project description yet.")

            st.markdown("---")

            # --- Quick stats ---
            convs = (
                secure_supabase
                .table("mvp_conversations")
                .select("id,title,created_at")
                .eq("project_id", project_id)
                .order("created_at", desc=True)
                .execute()
                .data or []
            )
            st.metric("Conversations", len(convs))

            proj_agents = get_project_agents(project_id)
            st.metric("Linked Agents", len(proj_agents))

            # --- Linked agents ---
            st.markdown("### Linked Agents")
            if proj_agents:
                for a in proj_agents:
                    st.write(f"- **{a['agent_name']}** — {a.get('description','')}")
            else:
                st.info("No agents linked to this project yet.")

            # --- Recent conversations ---
            st.markdown("### Recent Conversations")
            if convs:
                for c in convs[:5]:
                    st.write(f"• {c['title']}  —  {c['created_at'][:19]}")
            else:
                st.info("No conversations yet. Create one in the Conversations tab.")

            st.markdown("---")

            # --- Project Members ---
            st.markdown("### Project Members")

            # --- Add member ---
            st.markdown("### ➕ Add Member")

            # We already listed members above; if not, fetch them
            try:
                members
            except NameError:
                members = (
                    secure_supabase.table("project_members")
                    .select("user_id")
                    .eq("project_id", project_id)
                    .execute()
                    .data or []
                )

            # Build selectable users (exclude those already in the project)
            all_users = secure_supabase.table("Users").select("id,email").execute().data or []
            already = {m["user_id"] for m in members}
            user_options = {u["email"]: u["id"] for u in all_users if u["id"] not in already}

            # Team plan cap helper
            def assert_member_cap(supabase_client, project_id, plan):
                if plan != "team":  # Enterprise is unlimited
                    return
                res = (
                    supabase_client.table("project_members")
                    .select("user_id", count="exact")
                    .eq("project_id", project_id)
                    .execute()
                )
                current = getattr(res, "count", None) or len(getattr(res, "data", []) or [])
                if current >= 5:
                    raise PermissionError("Team plan allows up to 5 members per project.")

            if not user_options:
                st.info("All users are already in this project.")
            else:
                new_member_email = st.selectbox(
                    "Select user to add",
                    sorted(user_options.keys()),
                    key="add_member_email"
                )
                new_member_role = st.selectbox(
                    "Role for new member",
                    ["viewer", "editor", "owner"],
                    key="add_member_role"
                )

                if st.button("Add Member", key="btn_add_member"):
                    try:
                        assert_member_cap(secure_supabase, project_id, get_user_plan(user["id"]))
                        secure_supabase.table("project_members").insert({
                            "project_id": project_id,
                            "user_id": user_options[new_member_email],
                            "role": new_member_role
                        }).execute()
                        st.success(f"{new_member_email} added as {new_member_role}")
                        st.rerun()
                    except PermissionError as e:
                        st.warning(str(e))
                    except Exception as e:
                        st.error(f"Could not add member: {e}")

            # Show members with edit/remove
            members = (
                secure_supabase.table("project_members")
                .select("project_id,user_id,role")
                .eq("project_id", project_id)
                .execute()
                .data or []
            )
            user_ids = [m["user_id"] for m in members]
            user_emails = {}
            if user_ids:
                users = (
                    secure_supabase.table("Users")
                    .select("id,email")
                    .in_("id", user_ids)
                    .execute()
                    .data or []
                )
                user_emails = {u["id"]: u["email"] for u in users}

            if members:
                for m in members:
                    email = user_emails.get(m["user_id"], m["user_id"])
                    row_key = f"{m['user_id']}_{m['project_id']}"
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(email)
                    with col2:
                        new_role = st.selectbox(
                            "Role",
                            ["owner", "editor", "viewer"],
                            index=["owner", "editor", "viewer"].index(m["role"]),
                            key=f"role_{row_key}"
                        )
                    with col3:
                        if st.button("❌ Remove", key=f"remove_{row_key}"):
                            secure_supabase.table("project_members").delete().match({
                                "project_id": m["project_id"],
                                "user_id": m["user_id"],
                            }).execute()
                            st.rerun()

                    # update role if changed
                    if new_role != m["role"]:
                        secure_supabase.table("project_members").update(
                            {"role": new_role}
                        ).match({
                            "project_id": m["project_id"],
                            "user_id": m["user_id"],
                        }).execute()
            else:
                st.info("No members yet.")

            # --- Quick edit project details ---
            with st.expander("Quick edit project details"):
                new_name = st.text_input("Name", value=pr.get("name", ""))
                new_desc = st.text_area("Description", value=pr.get("description", "") or "")
                if st.button("Save project details"):
                    secure_supabase.table("projects").update({
                        "name": new_name.strip(),
                        "description": new_desc.strip()
                    }).eq("id", project_id).execute()
                    st.success("Project details updated.")
                    st.rerun()

    # ---------------- Conversations tab ----------------
    with tab_convos:
        if not project_id:
            st.info("Select a project to manage its conversations.")
        else:
            st.subheader("Conversations")

            # Create a new conversation
            with st.form("new_conv_form", clear_on_submit=True):
                new_title = st.text_input("Title", placeholder="e.g. Roadmap review")
                submitted = st.form_submit_button("Create")
                if submitted:
                    if not new_title.strip():
                        st.warning("Please enter a title.")
                    else:
                        conv = create_conversation(new_title, project_id, user["company_id"], user["id"])
                        st.session_state["conversation_id"] = conv["id"]
                        st.success("Conversation created.")
                        st.rerun()

            # List rooms for this project
            convs = list_conversations_for_project(project_id, user["id"])
            if not convs:
                st.info("No conversations yet. Create one above.")
            else:
                for c in convs:
                    if st.button(f"Open: {c['title']} · {c['created_at'][:19]}", key=f"open_{c['id']}"):
                        st.session_state["conversation_id"] = c["id"]
                        st.rerun()

            # Active room display + composer
            if "conversation_id" in st.session_state:
                conv_id = st.session_state["conversation_id"]
                ensure_membership(conv_id, user["id"])

                st.markdown("---")
                st.markdown("### Messages")

                # Light polling (3s)
                try:
                    st.autorefresh(interval=3000, key=f"poll_{conv_id}")
                except Exception:
                    pass

                msgs = list_messages(conv_id)
                user_ids = [m["author_id"] for m in msgs if m["author_type"] == "user" and m.get("author_id")]
                user_map = _users_by_id(list(set(user_ids)))

                for m in msgs:
                    badge = {"user":"👤","agent":"🤖","system":"⚙️","observer":"📝"}.get(m["author_type"], "💬")
                    who = user_map.get(m.get("author_id"), "User") if m["author_type"] == "user" else "Agent"
                    when = (m.get("created_at") or "")[:19]
                    st.markdown(f"{badge} **{who}** — *{when}*")
                    st.write(m["content"])
                    if m.get("meta") and m["meta"] != {}:
                        with st.expander("Details"):
                            st.json(m["meta"])
                    st.markdown("---")

                with st.form("composer", clear_on_submit=True):
                    text = st.text_area("Type a message", height=100, placeholder="Write your update or question…")
                    ask_agents = st.checkbox("Ask selected agents", value=True)
                    submitted = st.form_submit_button("Send")
                    if submitted and text.strip():
                        # Enforce allow-list + facilitator refinement
                        allowed_ids = assert_agents_allowed(supabase, project_id, selected_agent_ids)
                        refined_q = facilitator_gate(supabase, project_id, user["id"], allowed_ids, text.strip())

                        # Save the user's message with audit meta
                        post_user_message(conv_id, user["id"], refined_q,
                                          meta={"agents_requested": selected_agent_ids, "agents_used": list(allowed_ids)})

                        if ask_agents and allowed_ids:
                            # ↓↓↓ run agents only when allowed & requested ↓↓↓
                            pers = load_user_personality(user['id'], user['company_id'])
                            active_agents = [a for a in chosen_agents if a['id'] in allowed_ids]

                            # RANK/ASSIGN ROLES
                            agent_scores = []
                            results_by_agent = {}
                            for a in active_agents:
                                hits = fetch_hits_multi(
                                    agent_id=a['id'],
                                    query_text=refined_q,
                                    top_k=20,
                                    agent_lang=a.get("language")
                                )
                                best = hits[0]["_rank_score"] if hits else 0.0
                                agent_scores.append((a, best))
                                results_by_agent[a['id']] = hits
                            agent_scores.sort(key=lambda x: x[1], reverse=True)
                            role_order = ["default", "devil_advocate", "pragmatic", "supportive"]
                            role_results, cited_docs = {}, {}

                            # Default summary
                            if agent_scores:
                                default_agent = agent_scores[0][0]
                                default_hits = results_by_agent.get(default_agent['id'], [])
                                default_titles = set(h.get("source") or "" for h in default_hits)
                                default_sources = ", ".join(sorted(t for t in default_titles if t))
                                default_prompt = create_role_based_prompt("default", refined_q, default_agent["description"], default_sources)
                                default_summary, _, _ = query_agent_context(refined_q, default_prompt, personality=pers)
                                role_results[default_agent["agent_name"]] = default_summary
                                cited_docs[default_agent["agent_name"]] = default_sources

                                # Other roles
                                for i, (agent, _) in enumerate(agent_scores[1:], start=1):
                                    role = role_order[i] if i < len(role_order) else "supportive"
                                    hits = results_by_agent.get(agent['id'], [])
                                    sources = ", ".join(sorted({h.get("source") for h in hits if h.get("source")}))
                                    prompt = create_role_based_prompt(role, f"Question: {refined_q}\nSummary: {default_summary}", agent["description"], sources)
                                    agent_summary, _, _ = query_agent_context(f"Agent [{agent['agent_name']}] role: {role}",prompt,personality=pers)
                                    role_results[agent["agent_name"]] = agent_summary
                                    cited_docs[agent["agent_name"]] = sources

                                # Consensus
                                combined_context = "\n\n".join([f"[{n}] {s}" for n, s in role_results.items()])
                                final_sources = ", ".join(sorted(set(sum([(v.split(', ') if v else []) for v in cited_docs.values()], []))))
                                final_prompt = f"Final consensus answer based on all perspectives. Cite from: {final_sources}"
                                final_answer, _, _ = query_openai_context(final_prompt, combined_context, purpose="consensus", personality=pers)

                                post_agent_message(conv_id, "Consensus", final_answer, {
                                    "role": "consensus",
                                    "sources": final_sources.split(", ") if final_sources else []
                                })

# --------- 7. Ask Question with Role-Aware Collaboration ---------
if not is_paid:

    # FREE HOME
    st.title("AI Platform — Start a conversation")
    st.caption("Ask a question with your public experts. No project needed.")

    question = st.text_input("Ask your question:", placeholder="e.g. Why did Q2 margins dip?")

    if selected_agent_ids and st.button("Ask the question", type="primary"):
        # Only public agents are allowed on the free home
        allowed_ids = assert_agents_allowed(supabase, None, selected_agent_ids)
        if not allowed_ids:
            st.warning("Please select at least one public agent in the sidebar.")
        else:
            pers = load_user_personality(user['id'], user['company_id'])

            active_agents = [a for a in get_all_agents() if a['id'] in allowed_ids]

            role_order = ["default", "devil_advocate", "pragmatic", "supportive"]
            agent_scores, results_by_agent = [], {}

            # --- gather hits and score agents
            agent_scores = []
            results_by_agent = {}
            for agent in active_agents:
                hits = fetch_hits_multi(
                    agent_id=agent['id'],
                    query_text=question,
                    top_k=20,
                    agent_lang=agent.get("language")  # optional if you store it
                )
                best = hits[0]["_rank_score"] if hits else 0.0
                agent_scores.append((agent, best))
                results_by_agent[agent['id']] = hits
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            if not agent_scores:
                st.warning("No agents selected.")
                st.stop()

            assigned_roles, role_results, cited_docs = {}, {}, {}

            # --- telemetry buckets
            total_tokens = 0
            total_cost = 0.0
            step_stats = []  # list of dicts: {"step": str, "tokens": int, "cost": float}

            # --- DEFAULT SUMMARY
            default_agent = agent_scores[0][0]
            default_hits = results_by_agent.get(default_agent['id'], [])
            default_titles = set(h.get("source") or "" for h in default_hits)
            default_sources = ", ".join(sorted(t for t in default_titles if t))
            default_prompt = create_role_based_prompt("default", question, default_agent["description"], default_sources)

            default_summary, d_tokens, d_cost = query_agent_context(
               question, default_prompt, personality=pers)
            
            total_tokens += d_tokens or 0
            total_cost += d_cost or 0.0
            step_stats.append({"step": f"Default — {default_agent['agent_name']}", "tokens": d_tokens or 0, "cost": d_cost or 0.0})

            role_results[default_agent["agent_name"]] = default_summary
            assigned_roles[default_agent["agent_name"]] = "default"
            cited_docs[default_agent["agent_name"]] = default_sources

            # --- PEER ROLES
            for i, (agent, _) in enumerate(agent_scores[1:], start=1):
                role = role_order[i] if i < len(role_order) else "supportive"
                hits = results_by_agent.get(agent['id'], [])
                sources = ", ".join(sorted({h.get("source") for h in hits if h.get("source")}))
                prompt = create_role_based_prompt(
                    role,
                    f"Question: {question}\nSummary: {default_summary}",
                    agent["description"],
                    sources
                )
                agent_summary, a_tokens, a_cost = query_agent_context(
                f"Agent [{agent['agent_name']}] role: {role}",
                prompt,
                personality=pers
                )

                total_tokens += a_tokens or 0
                total_cost += a_cost or 0.0
                step_stats.append({"step": f"{role.title()} — {agent['agent_name']}", "tokens": a_tokens or 0, "cost": a_cost or 0.0})

                role_results[agent["agent_name"]] = agent_summary
                assigned_roles[agent["agent_name"]] = role
                cited_docs[agent["agent_name"]] = sources

            # --- CONSENSUS
            combined_context = "\n\n".join([f"[{name}] {summary}" for name, summary in role_results.items()])
            final_sources = ", ".join(sorted(set(sum([v.split(", ") for v in cited_docs.values()], [])))) if cited_docs else ""
            final_prompt = f"Final consensus answer based on all perspectives. Cite from: {final_sources}"

            final_answer, c_tokens, c_cost = query_openai_context(
                final_prompt, combined_context, purpose="consensus", personality=pers
            )
            total_tokens += c_tokens or 0
            total_cost += c_cost or 0.0
            step_stats.append({"step": "Consensus", "tokens": c_tokens or 0, "cost": c_cost or 0.0})

            # --- PRESENTATION
            st.success("🧠 Final AI response")
            st.write(final_answer)

            if final_sources:
                st.markdown("**Sources:** " + " • ".join(s for s in final_sources.split(", ") if s))

            # Telemetry summary
            st.markdown("**Run details**")
            for s in step_stats:
                st.caption(f"{s['step']}: {s['tokens']} tokens · ${s['cost']:.4f}")
            st.caption(f"**Total:** {total_tokens} tokens · **${total_cost:.4f}**")

            # Save to history (inside the same button/run scope)
            try:
                supabase.table("qa_history").insert({
                    "user_id": st.session_state['user']['id'],
                    "question": question,
                    "answer": final_answer,
                    "context": combined_context,
                    "agent_list": ",".join(role_results.keys()),
                    "agent_roles": str(assigned_roles),
                    "timestamp": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                st.warning(f"Could not save to history: {e}")

# --------- 8. History ---------
st.markdown("---")
st.subheader("🔁 Load a Previous Question")
history = supabase.table("qa_history").select("id, question, answer, timestamp, agent_list, agent_roles").order("timestamp", desc=True).limit(20).execute()
if history.data:
    for entry in history.data:
        label = f"{entry['question']} ({entry['agent_list']})"
        if st.checkbox(label, key=entry['id']):
            st.session_state['loaded_question'] = entry['question']
            st.session_state['loaded_answer'] = entry['answer']
            st.session_state['loaded_roles'] = entry['agent_roles']
            break

if 'loaded_question' in st.session_state:
    st.markdown("---")
    st.subheader("🎷 Reloaded Question")
    st.write(st.session_state['loaded_question'])
    st.subheader("👥 Agent Roles")
    st.code(st.session_state.get('loaded_roles', '{}'))
    st.subheader("🔮 Previous Answer")
    st.write(st.session_state['loaded_answer'])








