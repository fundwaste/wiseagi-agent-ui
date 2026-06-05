# app.py - WiseAGI Complete Rewrite with Cross-Agent Reasoning ✅
import streamlit as st
from supabase_config import supabase, secure_supabase
from supabase import create_client
from pymilvus import connections, Collection
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
from openai import APIStatusError
from functools import lru_cache
import os, re, json, numpy as np, requests
import tiktoken
import uuid
import jwt
from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.express as px

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Human Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- WiseAGI styling ----------------
def apply_wiseagi_theme():
    st.markdown("""
    <style>

    :root {
        --navy:#102A66;
        --navy-dark:#081A3D;
        --blue:#2256D6;
        --teal:#126E72;
        --text:#232733;
        --border:#D7DCE4;
        --bg:#F6F8FB;
        --surface:#FFFFFF;
    }

    .stApp {
        background-color: var(--bg);
    }

    h1,h2,h3 {
        color: var(--navy-dark);
    }

    section[data-testid="stSidebar"] {
        background-color:#EEF3FA;
        border-right:1px solid #D7DCE4;
    }

    .stButton > button {
        border-radius:20px;
        background-color:#2256D6;
        color:white;
        font-weight:bold;
        border:none;
    }

    .stButton > button:hover {
        background-color:#102A66;
        color:white;
    }

    div[data-testid="stChatMessage"]{
        background-color:white;
        border-radius:15px;
        padding:10px;
        border:1px solid #e0e0e0;
    }

    </style>
    """, unsafe_allow_html=True)

apply_wiseagi_theme()

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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
secure_supabase = create_client(SUPABASE_URL, SERVICE_ROLE_KEY)

EMBED_SHARED_SECRET = os.getenv("EMBED_SHARED_SECRET")

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

# --------- -----------Limited turmns pers tier  ---------
FREE_MAX_TURNS_PER_SESSION = 20  # simple MVP cap

# ----------Guardrail helpers (DeRad gatekeeper ----------
DERAD_COLLECTION = os.getenv("DERAD_COLLECTION", "derad")
GUARDRAIL_TRIGGERS_TABLE = os.getenv("GUARDRAIL_TRIGGERS_TABLE", "guardrail_triggers")

def post_system_message(conversation_id: str, content: str, meta: dict | None = None):
    """
    Optional: lets DeRad appear as a system message (⚙️) in the conversation UI.
    Your UI already supports author_type="system".
    """
    secure_supabase.table("mvp_messages").insert({
        "conversation_id": conversation_id,
        "author_id": None,
        "author_type": "system",
        "content": content,
        "meta": meta or {},
        "created_at": datetime.utcnow().isoformat(),
    }).execute()


@lru_cache(maxsize=1)
def load_guardrail_triggers() -> list[dict]:
    """
    Expects rows like:
      { phrase: "how to join", is_regex: false, severity: "high", enabled: true }
    Keep this cached for performance; clear cache if you add an admin 'refresh' button.
    """
    try:
        rows = (
            secure_supabase.table(GUARDRAIL_TRIGGERS_TABLE)
            .select("id,pattern,match_type,severity,action,is_active,category,language")
            .eq("is_active", True)
            .execute()
            .data
            or []
        )
        return rows
    except Exception:
        return []

def _match_trigger(text: str, trig: dict) -> bool:
    text = text or ""
    s = text.lower()

    pattern = (trig.get("pattern") or "").strip()
    if not pattern:
        return False

    match_type = (trig.get("match_type") or "phrase").lower()

    if match_type == "regex":
        try:
            return bool(re.search(pattern, text, flags=re.IGNORECASE))
        except Exception:
            return False

    # default phrase match
    return pattern.lower() in s

def guardrail_detect(question: str):
    triggers = load_guardrail_triggers()
    matched_patterns = []
    matched_ids = []
    max_sev = 0  # smallint in DB

    for t in triggers:
        if _match_trigger(question, t):
            matched_ids.append(t.get("id"))
            matched_patterns.append(t.get("pattern", ""))
            sev = int(t.get("severity") or 0)
            if sev > max_sev:
                max_sev = sev

    return (len(matched_patterns) > 0, matched_ids[:20], matched_patterns[:20], max_sev)

def _search_derad_collection(query_text: str, top_k: int = 8) -> list[dict]:
    """
    Attempts DeRad search robustly:
    - tries 384 embedding first, then 768 (if dimension mismatch)
    - tries typical field sets for output_fields
    Returns list of dicts with keys: text, source, distance
    """
    if not utility.has_collection(DERAD_COLLECTION):
        return []

    col = Collection(DERAD_COLLECTION)

    # candidates for output fields; we try in order
    field_sets = [
        ["Text", "Source", "agent_id"],
        ["Text", "Source"],
        ["text", "source"],
    ]

    # candidates for embeddings; we try in order
    vecs = []
    try:
        vecs.append(np.asarray(generate_embedding(query_text), dtype=np.float32))
    except Exception:
        pass

    try:
        vecs.append(embed_arabic_768(query_text))
    except Exception:
        pass

    last_err = None
    for vec in vecs:
        for fields in field_sets:
            try:
                results = col.search(
                    data=[vec],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=top_k,
                    output_fields=fields,
                )
                if not results or not results[0]:
                    return []

                out = []
                for hit in results[0]:
                    ent = hit.entity
                    txt = _safe_ent_get(ent, "Text", "") or _safe_ent_get(ent, "text", "")
                    src = _safe_ent_get(ent, "Source", "") or _safe_ent_get(ent, "source", "")
                    out.append({
                        "text": txt,
                        "source": src,
                        "distance": getattr(hit, "distance", 0.0),
                    })
                return out
            except Exception as e:
                last_err = e
                continue

    # optional: log last_err somewhere if you want
    return []


def run_derad_guardrail(question: str, personality: dict | None = None) -> tuple[bool, str, dict]:
    """
    Returns:
      (triggered, derad_response, meta)
    If triggered=True, you should SKIP the normal agent run and return this response instead.
    """
    triggered, matched_ids, matched_patterns, max_severity = guardrail_detect(question)
    if not triggered:
        return (False, "", {})

    hits = _norm(_search_derad_collection(question, top_k=8))
    sources = ", ".join(sorted({h.get("source") for h in hits if h.get("source")}))

    # Build a small RAG context from DeRad collection
    ctx_blocks = []
    for h in hits[:6]:
        if h.get("text"):
            ctx_blocks.append(f"- {h['text']}")
    derad_context = "\n".join(ctx_blocks).strip()

    system_prompt = (
        "You are DeRad, a safety and guidance assistant. "
        "Provide a non-radical, mainstream response. "
        "If the question is about harming others, illegal activity, or violent extremism, refuse to assist with harm. "
        "Instead, redirect to peaceful, lawful, ethical perspectives and encourage seeking help from trusted local community leaders or professionals. "
        "Keep the tone respectful and calm. "
        f"{'Use the provided sources where relevant.' if sources else ''}"
    )

    user_context = (
        f"Question:\n{question}\n\n"
        f"DeRad reference context:\n{derad_context}\n\n"
        f"Available sources: {sources}\n"
    )

    # Use cheap model route; you can change purpose if you want
    reply, tokens, cost = query_openai_context(
        prompt=system_prompt,
        context=user_context,
        purpose="peer_review",
        personality=personality,
    )

    meta = {
        "guardrail": True,
        "matched_trigger_ids": matched_ids,
        "matched_patterns": matched_patterns,
        "max_severity": max_severity,
        "sources": sources.split(", ") if sources else [],
        "tokens": tokens,
        "cost": cost,
        "derad_collection": DERAD_COLLECTION,
    }
    return (True, reply or "", meta)

# ---------log Guardrail events----------------------
def _sev_to_smallint(value) -> int:
    """
    Supports both:
    - old style: 'low'/'medium'/'high'
    - new style: numeric severity (0/1/2/3...)
    """
    if value is None:
        return 0

    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return 0

    s = str(value).strip().lower()
    return {
        "low": 1,
        "medium": 2,
        "high": 3,
    }.get(s, 0)


def log_guardrail_event(
    user_id: str | None,
    question: str,
    meta: dict | None = None,
    conversation_id: str | None = None,
    message_id: str | None = None,
    decision: str = "DERAD_ONLY",
):
    """
    Writes a row to public.guardrail_events.
    Compatible with both your current meta keys and the future schema-aligned ones.
    """
    meta = meta or {}

    # Backwards compatibility with current run_derad_guardrail meta
    matched_patterns = (
        meta.get("matched_patterns")
        or meta.get("matched_triggers")
        or []
    )

    matched_trigger_ids = meta.get("matched_trigger_ids") or []

    max_severity = _sev_to_smallint(
        meta.get("max_severity", meta.get("severity"))
    )

    payload = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "input_text": (question or "")[:4000],   # avoid overly long payloads
        "triggered": True,
        "decision": decision,
        "matched_trigger_ids": matched_trigger_ids,
        "matched_patterns": matched_patterns,
        "max_severity": max_severity,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Remove keys with None if your table/RLS is strict
    payload = {k: v for k, v in payload.items() if v is not None}

    secure_supabase.table("guardrail_events").insert(payload).execute()

# --------- Conversation Risk Reviewer Agent ----------------------

REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "gpt-4o-mini")


def fetch_recent_user_messages_for_review(conversation_id: str, limit: int = 12):
    """
    Fetches recent user messages from one conversation.
    Used by the reviewer agent to detect gradual escalation.
    """
    if not conversation_id:
        return []

    try:
        rows = (
            secure_supabase
            .table("mvp_messages")
            .select("id, author_type, content, created_at")
            .eq("conversation_id", conversation_id)
            .eq("author_type", "user")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )

        rows.reverse()
        return rows

    except Exception as e:
        print(f"Risk reviewer message fetch failed: {e}")
        return []


def safe_json_loads(raw_text: str) -> dict:
    """
    Safely parses JSON returned by the reviewer model.
    Prevents the app crashing if the model returns imperfect JSON.
    """
    try:
        return json.loads(raw_text)
    except Exception:
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw_text[start:end])
        except Exception:
            pass

    return {
        "single_message_risk": 0,
        "conversation_chain_risk": 0,
        "trend": "unclear",
        "risk_type": "unknown",
        "decision": "LOG_ONLY",
        "reason": "Reviewer JSON could not be parsed.",
        "candidate_phrases": [],
        "raw_output": raw_text,
    }

def classify_human_signal(latest_message: str, conversation_history: str = "") -> dict:
    """
    Hidden classifier. It does not answer the user.
    It decides the human signal, route, confidence and review queue.
    """

    system_prompt = """
You are the WiseAGI Human Signals Classifier.

Your job is NOT to answer the user.
Your job is to classify the human signals present in the latest message and recent conversation.

Return JSON only.

Allowed routes:
NORMAL
SOFT_SUPPORT
ELEVATED_REVIEW
CRITICAL_SAFETY

Allowed domains:
none
wellbeing
bullying_respect
child_protection
counter_extremism
digital_safety
workplace_behaviour

Allowed signals:
none
isolation
low_belonging
stress
anxiety
emotional_distress
burnout
disengagement
bullying
harassment
exclusion
discrimination
abuse
grooming
exploitation
safeguarding_concern
extremism
recruitment
radicalisation
violence
prompt_injection
privacy_risk
cyber_risk
unsafe_ai_use

Return this exact JSON shape:
{
  "route": "NORMAL",
  "domain": "none",
  "primary_signal": "none",
  "secondary_signal": "none",
  "confidence": 0.0,
  "severity": 0,
  "trend": "stable",
  "review_required": false,
  "review_queue": "none",
  "reason": "Short explanation."
}
"""

    user_prompt = f"""
Latest message:
{latest_message}

Recent conversation:
{conversation_history}
"""

    try:
        response = openai_client.chat.completions.create(
            model=REVIEWER_MODEL,
            temperature=0,
            max_tokens=350,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()
        classification = safe_json_loads(raw)

    except Exception as e:
        classification = {
            "route": "NORMAL",
            "domain": "none",
            "primary_signal": "none",
            "secondary_signal": "none",
            "confidence": 0.0,
            "severity": 0,
            "trend": "stable",
            "review_required": False,
            "review_queue": "none",
            "reason": f"Classifier failed safely: {e}",
        }

    return classification

def run_support_guidance_agent(question, classification):
    route = classification.get("route", "SOFT_SUPPORT")
    signal = classification.get("primary_signal", "")
    reason = classification.get("reason", "")
    memory_note = classification.get("memory_note", "")

    system_prompt = f"""
    You are the WiseAGI Support & Guidance Agent.

    Route: {route}
    Primary Signal: {signal}
    Reason: {reason}

    Previous memory:
    {memory_note}

    You respond like a calm, caring support worker.

    Always follow this order:
    1. Acknowledge the emotion.
    2. Validate the feeling.
    3. Reassure the user they are not alone.
    4. Offer one practical safety step.
    5. Ask one or two gentle follow-up questions.

    Important style rules:
    - Do not repeat the same opening sentence in every reply.
    - Respond to the newest detail the user has shared.
    - Sound like a calm, caring person, not a scripted chatbot.
    - Use natural, varied language.
    - Keep the response short.
    - Do not give a long list.
    - Do not overuse phrases like "you are not alone".
    - Do not keep repeating "talk to a trusted adult" in every message.
    - If you already gave safety advice, build on it rather than repeating it.
    - Ask one natural follow-up question.
    - Show that you remember the conversation so far.

    If the user shares a new serious detail, respond directly to that detail.

    Example:
    User: "They want me to hurt people and I do not want to."
    Better response:
    "Thank you for telling me. The most important thing is that you do not want to hurt anyone, and that matters. You are being pressured, and that is not your fault. Let us focus on keeping you and your sister safe right now. Are you somewhere safe while we talk?"
    
    If the user may be in immediate danger:
    - Tell them their safety is the most important thing right now.
    - Suggest moving towards a busy public place, trusted adult, shop, neighbour, reception desk, teacher, manager, or local support point.
    - Encourage emergency help if they believe they are in immediate danger.

    For bullying, fear, pressure, abuse, coercion, grooming, or safeguarding:
    - Tell the user they do not deserve this.
    - Encourage trusted human support.
    - Ask: "Are you safe right now?"
    - Ask: "Is this happening today or has it happened before?"
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=260,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Support & Guidance Agent error: {e}"
    
def review_conversation_risk(
    conversation_id: str,
    user_id: str,
    latest_message: str,
    latest_message_id: str | None = None,
    company_id: str | None = None,
):
    """
    Always-on reviewer agent.
    It does not answer the user.
    It scores the latest message and recent message chain.
    """

    recent_messages = fetch_recent_user_messages_for_review(conversation_id, limit=12)

    thread_text = "\n".join([
        f"{m.get('created_at', '')}: {m.get('content', '')}"
        for m in recent_messages
    ])

    system_prompt = """
You are an always-on safety reviewer for an educational and institutional AI platform.

Your task is NOT to answer the user.
Your task is to assess risk.

Review:
1. The latest message on its own.
2. The pattern across the recent conversation.

Look for:
- gradual escalation
- repeated probing
- rising harmful intent
- safeguarding concerns
- extremist grooming
- self-harm indicators
- violence
- abuse
- exploitation
- illegal operational requests
- coercion
- hate or targeted harm

Return JSON only.

Use this exact JSON shape:
{
  "single_message_risk": 0,
  "conversation_chain_risk": 0,
  "trend": "stable",
  "risk_type": "unknown",
  "decision": "ALLOW_NORMAL",
  "reason": "Short explanation.",
  "candidate_phrases": []
}

Risk scale:
0 = no concern
1 = mild concern
2 = emerging concern
3 = clear concern
4 = serious or urgent concern

Allowed decisions:
ALLOW_NORMAL
LOG_ONLY
ADD_SAFETY_NOTE
DERAD_ONLY
ESCALATE_FOR_REVIEW
"""

    user_prompt = f"""
Latest message:
{latest_message}

Recent conversation:
{thread_text}
"""

    try:
        response = openai_client.chat.completions.create(
            model=REVIEWER_MODEL,
            temperature=0,
            max_tokens=450,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content.strip()
        review = safe_json_loads(raw)

    except Exception as e:
        review = {
            "single_message_risk": 0,
            "conversation_chain_risk": 0,
            "trend": "unclear",
            "risk_type": "unknown",
            "decision": "LOG_ONLY",
            "reason": f"Reviewer failed safely: {e}",
            "candidate_phrases": [],
        }

    # Normalise values
    try:
        single_risk = int(review.get("single_message_risk", 0))
    except Exception:
        single_risk = 0

    try:
        chain_risk = int(review.get("conversation_chain_risk", 0))
    except Exception:
        chain_risk = 0

    decision = (review.get("decision") or "LOG_ONLY").upper()

    if decision not in [
        "ALLOW_NORMAL",
        "LOG_ONLY",
        "ADD_SAFETY_NOTE",
        "DERAD_ONLY",
        "ESCALATE_FOR_REVIEW",
    ]:
        decision = "LOG_ONLY"

    # Save review to Supabase
    try:
        secure_supabase.table("conversation_risk_reviews").insert({
            "company_id": company_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id": latest_message_id,
            "latest_message": (latest_message or "")[:4000],
            "reviewed_context": (thread_text or "")[:8000],
            "single_message_risk": single_risk,
            "conversation_chain_risk": chain_risk,
            "trend": review.get("trend", "unclear"),
            "risk_type": review.get("risk_type", "unknown"),
            "decision": decision,
            "reason": review.get("reason", ""),
            "reviewer_model": REVIEWER_MODEL,
            "reviewer_raw": review,
            "reviewed_by_human": False,
        }).execute()
    except Exception as e:
        print(f"Could not save conversation risk review: {e}")

    # Save candidate phrases if provided
    try:
        save_candidate_patterns(
            candidates=review.get("candidate_phrases", []),
            conversation_id=conversation_id,
            message_id=latest_message_id,
        )
    except Exception as e:
        print(f"Could not save candidate phrases: {e}")

    # Make sure the returned review uses the cleaned values
    review["decision"] = decision
    review["single_message_risk"] = single_risk
    review["conversation_chain_risk"] = chain_risk

    risk_type = (review.get("risk_type") or "").lower()
    reason = (review.get("reason") or "").lower()
    latest_text = (latest_message or "").lower()

    combined_text = f"{risk_type} {reason} {latest_text}"

    signal_name = None

    if any(word in combined_text for word in ["isolated", "isolation", "lonely", "alone", "no friends"]):
        signal_name = "isolation"

    elif any(word in combined_text for word in ["stress", "stressed", "overwhelmed", "pressure", "anxiety", "anxious"]):
        signal_name = "stress"

    elif any(word in combined_text for word in ["bully", "bullied", "mocking", "harassment", "excluded"]):
        signal_name = "bullying"

    elif any(word in combined_text for word in ["abuse", "exploitation", "grooming", "unsafe at home"]):
        signal_name = "abuse"

    elif any(word in combined_text for word in ["extremism", "radical", "terrorism", "violence", "recruitment"]):
        signal_name = "extremism"

    elif any(word in combined_text for word in ["jailbreak", "prompt injection", "hack", "privacy", "cyber"]):
        signal_name = "digital_safety"

    if signal_name:
        confidence = max(
            float(review.get("single_message_risk", 0)),
            float(review.get("conversation_chain_risk", 0))
        ) / 4

        log_human_signal_event(
            company_id=company_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=latest_message_id,
            signal_name=signal_name,
            confidence=confidence,
            trend_score=float(review.get("conversation_chain_risk", 0)) / 4,
            raw_review=review,
        )

    print("HUMAN SIGNAL DEBUG:", {
        "risk_type": review.get("risk_type"),
        "reason": review.get("reason"),
        "signal_name": signal_name,
    })

    classification = classify_human_signal(
        latest_message=latest_message,
        conversation_history=thread_text
    )

    print("HUMAN SIGNAL CLASSIFIER:", classification)

    primary_signal = classification.get("primary_signal")

    if primary_signal and primary_signal != "none":
        log_human_signal_event(
            company_id=company_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=latest_message_id,
            signal_name=primary_signal,
            confidence=float(classification.get("confidence", 0)),
            trend_score=float(classification.get("severity", 0)) / 4,
            source="human_signals_classifier",
            raw_review=classification,
        )

    return review

def candidate_phrase_exists(phrase: str) -> bool:
    """
    Checks whether a suggested phrase already exists as a trigger or candidate.
    """
    phrase = (phrase or "").strip().lower()

    if not phrase:
        return True

    try:
        existing_trigger = (
            secure_supabase
            .table("guardrail_triggers")
            .select("id")
            .ilike("pattern", phrase)
            .limit(1)
            .execute()
            .data or []
        )

        if existing_trigger:
            return True

        existing_candidate = (
            secure_supabase
            .table("guardrail_candidate_patterns")
            .select("id")
            .ilike("phrase", phrase)
            .neq("status", "rejected")
            .limit(1)
            .execute()
            .data or []
        )

        return bool(existing_candidate)

    except Exception as e:
        print(f"Candidate duplicate check failed: {e}")
        return False


def save_candidate_patterns(candidates, conversation_id=None, message_id=None):
    """
    Saves possible new guardrail phrases suggested by the reviewer agent.
    Supports both:
    - list of strings
    - list of dictionaries
    """
    if not candidates:
        return

    for c in candidates:

        if isinstance(c, str):
            phrase = c.strip()
            category = "unknown"
            language = "unknown"
            suggested_severity = 1
            reviewer_reason = ""

        elif isinstance(c, dict):
            phrase = (c.get("phrase") or "").strip()
            category = c.get("category", "unknown")
            language = c.get("language", "unknown")
            suggested_severity = int(c.get("suggested_severity", 1) or 1)
            reviewer_reason = c.get("reason", "")

        else:
            continue

        if not phrase:
            continue

        if candidate_phrase_exists(phrase):
            continue

        secure_supabase.table("guardrail_candidate_patterns").insert({
            "phrase": phrase,
            "category": category,
            "language": language,
            "source_conversation_id": conversation_id,
            "source_message_id": message_id,
            "suggested_severity": suggested_severity,
            "reviewer_reason": reviewer_reason,
            "status": "pending",
        }).execute()

# ---------Memory helpers----------------------

def update_user_memory_profile(
    user_id,
    company_id=None,
    conversation_id=None,
    message_id=None,
    classification=None,
    latest_message=None,
):
    if not user_id:
        return

    classification = classification or {}

    signal = classification.get("primary_signal", "none")
    route = classification.get("route", "NORMAL")
    severity = int(classification.get("severity", 0) or 0)
    confidence = float(classification.get("confidence", 0) or 0)

    if signal == "none" or confidence < 0.5:
        return

    memory_summary = f"User recently discussed a concern linked to {signal}."
    wellbeing_summary = classification.get("reason", "")
    signal_summary = f"Signal: {signal}. Route: {route}. Severity: {severity}."

    checkin_prompt = (
        f"Last time we spoke, you mentioned something linked to {signal}. "
        "How are things feeling today?"
    )

    payload = {
        "user_id": user_id,
        "company_id": company_id,
        "memory_summary": memory_summary,
        "wellbeing_summary": wellbeing_summary,
        "signal_summary": signal_summary,
        "last_signal": signal,
        "last_route": route,
        "last_severity": severity,
        "last_conversation_id": conversation_id,
        "last_message_id": message_id,
        "checkin_prompt": checkin_prompt,
        "confidence": confidence,
        "updated_at": datetime.utcnow().isoformat(),
    }

    secure_supabase.table("user_memory_profiles").upsert(
        payload,
        on_conflict="user_id,company_id"
    ).execute()


def fetch_user_memory_profile(user_id, company_id=None):
    if not user_id:
        return None

    q = (
        secure_supabase
        .table("user_memory_profiles")
        .select("*")
        .eq("user_id", user_id)
    )

    if company_id:
        q = q.eq("company_id", company_id)

    rows = q.limit(1).execute().data or []
    return rows[0] if rows else None

def get_runtime_context() -> Dict[str, Any]:
    """
    Universal context for memory and prompt shaping.
    Works for education and non-education verticals.
    """

    user = st.session_state.get("user") or {}
    learning_context = st.session_state.get("learning_context") or {}

    company_id = user.get("company_id")
    user_id = user.get("id")
    project_id = st.session_state.get("active_project_id")
    agent_id = st.session_state.get("active_agent_id")
    conversation_id = st.session_state.get("conversation_id")

    # Plan: use session first if you already store it there, otherwise fall back
    plan_code = (
        st.session_state.get("user_plan")
        or st.session_state.get("plan")
        or "free"
    )

    # Decide vertical
    # For now: if learning_context exists, treat as education, otherwise general/corporate
    if learning_context:
        vertical = "education"
        context_json = {
            "subject": learning_context.get("subject"),
            "year_group": learning_context.get("year_group"),
            "topic": learning_context.get("topic"),
            "external_user_id": learning_context.get("external_user_id"),
            "support_profile": learning_context.get("support_profile"),
            "source": learning_context.get("source"),
        }
    else:
        vertical = st.session_state.get("vertical") or "general"
        context_json = {
            "department": st.session_state.get("department"),
            "task_type": st.session_state.get("task_type"),
            "workspace": st.session_state.get("workspace"),
        }

    return {
        "company_id": company_id,
        "user_id": user_id,
        "project_id": project_id,
        "agent_id": agent_id,
        "conversation_id": conversation_id,
        "plan_code": plan_code,
        "vertical": vertical,
        "external_user_id": learning_context.get("external_user_id"),
        "context_json": context_json,
    }

def update_user_memory_profile(
    user_id,
    company_id=None,
    conversation_id=None,
    message_id=None,
    classification=None,
    latest_message=None,
):
    classification = classification or {}

    signal = classification.get("primary_signal", "none")
    route = classification.get("route", "NORMAL")
    severity = int(classification.get("severity", 0) or 0)
    confidence = float(classification.get("confidence", 0) or 0)

    if signal == "none" or confidence < 0.5:
        return

    memory_summary = f"User recently discussed a concern linked to {signal}."
    wellbeing_summary = classification.get("reason", "")
    signal_summary = f"Signal: {signal}. Route: {route}. Severity: {severity}."

    checkin_prompt = (
        f"Last time we spoke, you mentioned something linked to {signal}. "
        "How are things feeling today?"
    )

    payload = {
        "user_id": user_id,
        "company_id": company_id,
        "memory_summary": memory_summary,
        "wellbeing_summary": wellbeing_summary,
        "signal_summary": signal_summary,
        "last_signal": signal,
        "last_route": route,
        "last_severity": severity,
        "last_conversation_id": conversation_id,
        "last_message_id": message_id,
        "checkin_prompt": checkin_prompt,
        "confidence": confidence,
        "updated_at": datetime.utcnow().isoformat(),
    }

    secure_supabase.table("user_memory_profiles").upsert(
        payload,
        on_conflict="user_id,company_id"
    ).execute()

def get_memory_policy(company_id: Optional[str], plan_code: str = "free") -> Dict[str, Any]:
    """
    Resolve memory settings for the current company and plan.
    Uses secure_supabase so it works even if RLS policies are not complete yet.
    """

    default_policy = {
        "memory_short_enabled": True,
        "memory_medium_enabled": False,
        "memory_long_enabled": False,
        "memory_vector_enabled": False,
        "memory_retrieval_limit": 3,
        "memory_retention_days": 30,
        "memory_capture_mode": "triggered",
        "memory_admin_controls": False,
    }

    if not company_id:
        return default_policy

    try:
        # Most specific: exact company + plan
        res = (
            secure_supabase
            .table("company_memory_policies")
            .select("*")
            .eq("company_id", company_id)
            .eq("plan_code", plan_code)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if rows:
            row = rows[0]
            return {**default_policy, **row}

        # Fallback: company default policy where plan_code is null
        res2 = (
            secure_supabase
            .table("company_memory_policies")
            .select("*")
            .eq("company_id", company_id)
            .is_("plan_code", "null")
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        rows2 = res2.data or []
        if rows2:
            row = rows2[0]
            return {**default_policy, **row}

    except Exception as e:
        print(f"Memory policy lookup failed: {e}")

    return default_policy

def fetch_recent_thread_messages(conversation_id: Optional[str], limit: int = 8) -> List[Dict[str, Any]]:
    """
    Get the most recent messages for short-term memory.
    Returns oldest-to-newest so prompt order is natural.
    """
    if not conversation_id:
        return []

    try:
        rows = (
            secure_supabase
            .table("mvp_messages")
            .select("author_type, content, meta, created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )

        rows.reverse()  # convert newest-first back to oldest-first
        return rows

    except Exception as e:
        print(f"Thread fetch failed: {e}")
        return []
    
def format_thread_memory_for_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Turn recent thread messages into prompt text.
    """
    if not messages:
        return ""

    lines = []
    for m in messages:
        speaker = "User" if m.get("author_type") == "user" else "Assistant"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{speaker}: {content}")

    if not lines:
        return ""

    return "\n".join(lines)

def build_memory_expiry(retention_days: int = 30) -> Optional[str]:
    """
    Return ISO timestamp for expires_at.
    """
    try:
        days = int(retention_days or 30)
        return (datetime.utcnow() + timedelta(days=days)).isoformat()
    except Exception:
        return None

def save_episode(
    summary: str,
    trigger_type: str = "consensus_end",
    tags: Optional[Dict[str, Any]] = None,
    turn_id: Optional[str] = None
) -> None:
    """
    Save medium-term episodic memory.
    """
    if not summary or not summary.strip():
        return

    runtime = get_runtime_context()
    policy = get_memory_policy(runtime.get("company_id"), runtime.get("plan_code", "free"))

    if not policy.get("memory_medium_enabled", False):
        return

    try:
        row = {
            "company_id": runtime.get("company_id"),
            "user_id": runtime.get("user_id"),
            "external_user_id": runtime.get("external_user_id"),
            "conversation_id": runtime.get("conversation_id"),
            "turn_id": turn_id or str(uuid.uuid4()),
            "project_id": runtime.get("project_id"),
            "agent_id": runtime.get("agent_id"),
            "plan_code": runtime.get("plan_code"),
            "vertical": runtime.get("vertical", "general"),
            "memory_type": "episodic",
            "summary": summary.strip(),
            "tags": tags or {},
            "context_json": runtime.get("context_json") or {},
            "trigger_type": trigger_type,
            "expires_at": build_memory_expiry(policy.get("memory_retention_days", 30)),
            "is_active": True,
        }

        secure_supabase.table("episodic_memories").insert(row).execute()

    except Exception as e:
        print(f"Save episode failed: {e}")

def fetch_recent_episodes(limit: Optional[int] = None) -> List[str]:
    """
    Fetch recent episodic memories for the current user and context.
    """
    runtime = get_runtime_context()
    policy = get_memory_policy(runtime.get("company_id"), runtime.get("plan_code", "free"))

    if not policy.get("memory_medium_enabled", False):
        return []

    fetch_limit = limit or int(policy.get("memory_retrieval_limit", 3))

    try:
        q = (
            secure_supabase
            .table("episodic_memories")
            .select("summary, created_at, context_json, vertical")
            .eq("company_id", runtime.get("company_id"))
            .eq("user_id", runtime.get("user_id"))
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(fetch_limit)
        )

        # Keep retrieval relevant to the same vertical
        if runtime.get("vertical"):
            q = q.eq("vertical", runtime.get("vertical"))

        rows = q.execute().data or []
        return [r["summary"] for r in rows if r.get("summary")]

    except Exception as e:
        print(f"Fetch episodes failed: {e}")
        return []

def inject_episodic_memory(sys_prompt: str, limit: Optional[int] = None) -> str:
    """
    Add episodic memory notes to the system prompt.
    """
    episodes = fetch_recent_episodes(limit=limit)
    if not episodes:
        return sys_prompt

    sys_prompt += "\n\nRelevant prior memory notes (use silently to personalise the response):\n"
    for ep in episodes:
        sys_prompt += f"- {ep}\n"

    sys_prompt += "\nDo not mention these notes explicitly. Use them only to improve relevance, clarity, tone, or structure."
    return sys_prompt

def build_episode_summary(question: str, final_answer: str) -> str:
    """
    Very simple MVP memory note.
    Keep it short and useful.
    """
    runtime = get_runtime_context()
    vertical = runtime.get("vertical", "general")
    ctx = runtime.get("context_json") or {}

    if vertical == "education":
        subject = ctx.get("subject") or "General"
        topic = ctx.get("topic") or "General topic"
        return f"{subject} / {topic}: learner needed a clear explanation with concise steps."

    task_type = ctx.get("task_type") or "general task"
    return f"{vertical}: user benefited from a concise response style for {task_type}."

# --------- Admin / Observability helpers ---------
def log_dashboard_visit(page_name="admin_dashboard"):
    try:
        user = st.session_state.get("user") or {}

        if not user.get("id"):
            return

        key = f"logged_{page_name}"

        if st.session_state.get(key):
            return

        secure_supabase.table("dashboard_login_events").insert({
            "user_id": user.get("id"),
            "company_id": user.get("company_id"),
            "email": user.get("email"),
            "page_name": page_name,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        st.session_state[key] = True

    except Exception as e:
        print(f"Dashboard visit log failed: {e}")

log_dashboard_visit("admin_dashboard")

def fetch_llm_usage_admin(days=30):
    start_iso = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = (
        secure_supabase
        .table("llm_usage")
        .select("*")
        .gte("created_at", start_iso)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
        .data or []
    )

    return pd.DataFrame(rows)

def fetch_company_dashboard_analytics(days=30):
    start_iso = (datetime.utcnow() - timedelta(days=days)).isoformat()

    login_rows = (
        secure_supabase
        .table("dashboard_login_events")
        .select("company_id, email, created_at")
        .gte("created_at", start_iso)
        .execute()
        .data or []
    )

    usage_rows = (
        secure_supabase
        .table("llm_usage")
        .select("company_id, sell_price_gbp, cost_cost_gbp, created_at")
        .gte("created_at", start_iso)
        .execute()
        .data or []
    )

    companies = (
        secure_supabase
        .table("companies")
        .select("id, name")
        .execute()
        .data or []
    )

    company_map = {c["id"]: c["name"] for c in companies}

    login_df = pd.DataFrame(login_rows)
    usage_df = pd.DataFrame(usage_rows)

    if not login_df.empty:
        login_df["company_name"] = login_df["company_id"].map(company_map)

    if not usage_df.empty:
        usage_df["company_name"] = usage_df["company_id"].map(company_map)

    return login_df, usage_df

def get_signal_id(signal_name: str):
    try:
        rows = (
            secure_supabase
            .table("signals_catalogue")
            .select("id")
            .eq("signal_name", signal_name)
            .eq("active", True)
            .limit(1)
            .execute()
            .data or []
        )

        return rows[0]["id"] if rows else None

    except Exception as e:
        print(f"Signal lookup failed: {e}")
        return None


def log_human_signal_event(
    company_id=None,
    user_id=None,
    conversation_id=None,
    message_id=None,
    signal_name=None,
    confidence=0,
    trend_score=0,
    source="conversation_risk_reviewer",
    raw_review=None,
):
    try:
        signal_id = get_signal_id(signal_name)

        if not signal_id:
            print(f"No signal found for: {signal_name}")
            return

        secure_supabase.table("human_signal_events").insert({
            "company_id": company_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "signal_id": signal_id,
            "confidence": confidence,
            "trend_score": trend_score,
            "source": source,
            "raw_review": raw_review or {},
            "created_at": datetime.utcnow().isoformat(),
        }).execute()

    except Exception as e:
        print(f"Human signal event logging failed: {e}")

def fetch_human_signals_admin(limit=500):
    rows = (
        secure_supabase
        .table("human_signal_events")
        .select("""
            id,
            company_id,
            user_id,
            conversation_id,
            message_id,
            confidence,
            trend_score,
            source,
            raw_review,
            created_at,
            signals_catalogue (
                signal_name,
                description
            )
        """)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data or []
    )

    cleaned = []

    for r in rows:
        signal = r.get("signals_catalogue") or {}
        raw = r.get("raw_review") or {}

        cleaned.append({
            "created_at": r.get("created_at"),
            "signal": signal.get("signal_name"),
            "confidence": r.get("confidence"),
            "trend_score": r.get("trend_score"),
            "decision": raw.get("decision") or raw.get("route"),
            "trend": raw.get("trend"),
            "risk_type": raw.get("risk_type") or raw.get("domain"),
            "reason": raw.get("reason"),
            "company_id": r.get("company_id"),
            "user_id": r.get("user_id"),
            "conversation_id": r.get("conversation_id"),
            "message_id": r.get("message_id"),
        })

    return pd.DataFrame(cleaned)

def fetch_guardrail_events_df(days=30):
    start_iso = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = (
        secure_supabase
        .table("guardrail_events")
        .select("*")
        .gte("created_at", start_iso)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
        .data or []
    )

    return pd.DataFrame(rows)

def fetch_training_examples_for_review(limit=20, signal_filter="all"):
    q = (
        secure_supabase
        .table("signal_training_examples")
        .select("""
            id,
            text,
            dataset_category,
            mapped_route,
            review_status,
            created_at,
            signal_id,
            signals_catalogue (
                signal_name
            )
        """)
        .eq("review_status", "pending")
        .order("created_at", desc=False)
        .limit(limit)
    )

    if signal_filter != "all":
        q = q.eq("signal_id", signal_filter)

    return q.execute().data or []


def fetch_signals_for_dropdown():
    rows = (
        secure_supabase
        .table("signals_catalogue")
        .select("id, signal_name")
        .eq("active", True)
        .order("signal_name")
        .execute()
        .data or []
    )
    return rows


def save_signal_reviewer_feedback(
    training_example_id,
    original_signal_id,
    reviewer_signal_id,
    reviewer_confidence,
    reviewer_notes,
    reviewer_id=None,
):
    secure_supabase.table("signal_reviewer_feedback").insert({
        "training_example_id": training_example_id,
        "original_signal_id": original_signal_id,
        "reviewer_signal_id": reviewer_signal_id,
        "reviewer_confidence": reviewer_confidence,
        "reviewer_notes": reviewer_notes,
        "reviewer_id": reviewer_id,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()

    secure_supabase.table("signal_training_examples").update({
        "review_status": "reviewed"
    }).eq("id", training_example_id).execute()

def fetch_risk_reviews_df(days=30):
    start_iso = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = (
        secure_supabase
        .table("conversation_risk_reviews")
        .select("*")
        .gte("created_at", start_iso)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
        .data or []
    )

    return pd.DataFrame(rows)

def _sev_label(v: int) -> str:
    try:
        return {1: "low", 2: "medium", 3: "high"}.get(int(v or 0), "unknown")
    except Exception:
        return "unknown"


def get_runtime_flag(key: str, default: str = "false") -> str:
    try:
        row = (
            secure_supabase.table("app_runtime_flags")
            .select("value")
            .eq("key", key)
            .maybe_single()
            .execute()
            .data
        ) or {}
        return str(row.get("value", default)).strip().lower()
    except Exception:
        return default.lower()


def set_runtime_flag(key: str, value: str):
    secure_supabase.table("app_runtime_flags").upsert({
        "key": key,
        "value": str(value).strip().lower(),
        "updated_at": datetime.utcnow().isoformat(),
    }).execute()


def guardrails_enabled() -> bool:
    return get_runtime_flag("guardrails_enabled", "true") in ("1", "true", "yes", "on")


def validate_trigger_pattern(pattern: str, match_type: str) -> tuple[bool, str]:
    p = (pattern or "").strip()
    mt = (match_type or "phrase").strip().lower()

    if not p:
        return False, "Pattern is required."

    if mt == "regex":
        try:
            re.compile(p, flags=re.IGNORECASE)
        except re.error as e:
            return False, f"Invalid regex: {e}"

    return True, ""


def list_guardrail_triggers_admin(
    language: str = "all",
    severity: str = "all",
    active: str = "all",
):
    q = (
        secure_supabase.table("guardrail_triggers")
        .select("id,pattern,match_type,language,category,severity,action,is_active,notes,updated_at,created_at")
        .order("severity", desc=True)
        .order("updated_at", desc=True)
    )

    if language != "all":
        q = q.eq("language", language)

    if severity != "all":
        q = q.eq("severity", int(severity))

    if active == "active":
        q = q.eq("is_active", True)
    elif active == "inactive":
        q = q.eq("is_active", False)

    return q.execute().data or []


def save_guardrail_trigger_admin(
    trigger_id: str | None,
    pattern: str,
    match_type: str,
    language: str,
    category: str,
    severity: int,
    action: str,
    is_active: bool,
    notes: str,
    created_by: str | None = None,
):
    ok, msg = validate_trigger_pattern(pattern, match_type)
    if not ok:
        raise ValueError(msg)

    payload = {
        "pattern": (pattern or "").strip(),
        "match_type": (match_type or "phrase").strip().lower(),
        "language": (language or "en").strip().lower(),
        "category": (category or "general").strip().lower(),
        "severity": int(severity),
        "action": (action or "CONFIRM").strip().upper(),
        "is_active": bool(is_active),
        "notes": (notes or "").strip(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    if trigger_id:
        secure_supabase.table("guardrail_triggers").update(payload).eq("id", trigger_id).execute()
    else:
        if created_by:
            payload["created_by"] = created_by
        secure_supabase.table("guardrail_triggers").insert(payload).execute()

    # clear trigger cache used by guardrail runtime
    try:
        load_guardrail_triggers.cache_clear()
    except Exception:
        pass


def set_guardrail_trigger_active(trigger_id: str, active: bool):
    secure_supabase.table("guardrail_triggers").update({
        "is_active": bool(active),
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", trigger_id).execute()

    try:
        load_guardrail_triggers.cache_clear()
    except Exception:
        pass


def list_guardrail_events_admin(
    limit: int = 200,
    severity: str = "all",
    language: str = "all",
    decision: str = "all",
):
    q = (
        secure_supabase.table("guardrail_events")
        .select("id,created_at,user_id,conversation_id,message_id,decision,triggered,max_severity,language,latency_ms,matched_patterns,input_text")
        .order("created_at", desc=True)
        .limit(limit)
    )

    if severity != "all":
        q = q.eq("max_severity", int(severity))
    if language != "all":
        q = q.eq("language", language)
    if decision != "all":
        q = q.eq("decision", decision)

    return q.execute().data or []


def get_guardrail_metrics_admin(days: int = 7):
    start_iso = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = (
        secure_supabase.table("guardrail_events")
        .select("triggered,decision,latency_ms,max_severity,created_at")
        .gte("created_at", start_iso)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
        .data
        or []
    )

    total = len(rows)
    triggered_count = sum(1 for r in rows if r.get("triggered") is True)

    decision_counts = {
        "ALLOW_NORMAL": 0,
        "ADD_DERAD_NOTE": 0,
        "DERAD_ONLY": 0,
        "ESCALATE_FOR_REVIEW": 0,
    }
    sev_counts = {1: 0, 2: 0, 3: 0}
    lats = []

    for r in rows:
        d = (r.get("decision") or "").upper()
        if d in decision_counts:
            decision_counts[d] += 1

        try:
            sev = int(r.get("max_severity") or 0)
            if sev in sev_counts:
                sev_counts[sev] += 1
        except Exception:
            pass

        if r.get("latency_ms") is not None:
            try:
                lats.append(int(r.get("latency_ms")))
            except Exception:
                pass

    avg_latency_ms = round(sum(lats) / len(lats), 1) if lats else None
    trigger_rate_pct = round((triggered_count / total) * 100, 1) if total else 0.0

    return {
        "total_checks": total,
        "triggered": triggered_count,
        "trigger_rate_pct": trigger_rate_pct,
        "avg_latency_ms": avg_latency_ms,
        "decision_counts": decision_counts,
        "severity_counts": sev_counts,
    }

def update_candidate_pattern_status(candidate_id: str, status: str, reviewed_by: str | None = None):
    secure_supabase.table("guardrail_candidate_patterns").update({
        "status": status,
        "reviewed_by": reviewed_by,
        "reviewed_at": datetime.utcnow().isoformat(),
    }).eq("id", candidate_id).execute()


def approve_candidate_as_guardrail(candidate: dict, reviewed_by: str | None = None):
    """
    Converts a candidate phrase into a live guardrail trigger.
    """
    phrase = candidate.get("phrase")
    if not phrase:
        raise ValueError("Candidate phrase is missing.")

    secure_supabase.table("guardrail_triggers").insert({
        "pattern": phrase,
        "match_type": "phrase",
        "language": candidate.get("language") or "unknown",
        "category": candidate.get("category") or "unknown",
        "severity": int(candidate.get("suggested_severity") or 1),
        "action": "DERAD_ONLY",
        "is_active": True,
        "notes": f"Added from reviewer candidate: {candidate.get('reviewer_reason', '')}",
        "created_by": reviewed_by,
        "updated_at": datetime.utcnow().isoformat(),
    }).execute()

    update_candidate_pattern_status(
        candidate_id=candidate["id"],
        status="added_to_guardrails",
        reviewed_by=reviewed_by,
    )

    try:
        load_guardrail_triggers.cache_clear()
    except Exception:
        pass

# --------- Risk Intelligence Leaderboard Helpers ----------------------

def calculate_leaderboard_score(
    trigger_count=0,
    unique_conversations=0,
    avg_chain_risk=0,
    max_severity=0,
    trend="stable",
    escalations=0,
):
    score = 0
    score += int(trigger_count or 0) * 2
    score += int(unique_conversations or 0) * 3
    score += float(avg_chain_risk or 0) * 10
    score += int(max_severity or 0) * 8
    score += int(escalations or 0) * 15

    if trend == "rising":
        score += 20

    return round(score, 2)


def get_top_guardrail_patterns(days: int = 30, limit: int = 20):
    start_iso = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = (
        secure_supabase
        .table("guardrail_events")
        .select("id,created_at,conversation_id,user_id,matched_patterns,max_severity,decision")
        .gte("created_at", start_iso)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
        .data or []
    )

    stats = {}

    for r in rows:
        patterns = r.get("matched_patterns") or []
        for p in patterns:
            key = (p or "").strip()
            if not key:
                continue

            if key not in stats:
                stats[key] = {
                    "pattern": key,
                    "trigger_count": 0,
                    "conversation_ids": set(),
                    "user_ids": set(),
                    "max_severity": 0,
                    "escalations": 0,
                    "last_seen_at": r.get("created_at"),
                    "example_event_ids": [],
                }

            stats[key]["trigger_count"] += 1

            if r.get("conversation_id"):
                stats[key]["conversation_ids"].add(r.get("conversation_id"))

            if r.get("user_id"):
                stats[key]["user_ids"].add(r.get("user_id"))

            sev = int(r.get("max_severity") or 0)
            stats[key]["max_severity"] = max(stats[key]["max_severity"], sev)

            if r.get("decision") in ["DERAD_ONLY", "ESCALATE_FOR_REVIEW"]:
                stats[key]["escalations"] += 1

            if len(stats[key]["example_event_ids"]) < 5:
                stats[key]["example_event_ids"].append(r.get("id"))

    output = []

    for item in stats.values():
        unique_conversations = len(item["conversation_ids"])
        score = calculate_leaderboard_score(
            trigger_count=item["trigger_count"],
            unique_conversations=unique_conversations,
            avg_chain_risk=0,
            max_severity=item["max_severity"],
            trend="stable",
            escalations=item["escalations"],
        )

        output.append({
            "pattern": item["pattern"],
            "trigger_count": item["trigger_count"],
            "unique_conversation_count": unique_conversations,
            "unique_user_count": len(item["user_ids"]),
            "max_severity": item["max_severity"],
            "escalations": item["escalations"],
            "leaderboard_score": score,
            "last_seen_at": item["last_seen_at"],
            "example_event_ids": item["example_event_ids"],
        })

    output.sort(key=lambda x: x["leaderboard_score"], reverse=True)
    return output[:limit]


def get_rising_conversation_chains(days: int = 30, limit: int = 20):
    start_iso = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = (
        secure_supabase
        .table("conversation_risk_reviews")
        .select("id,created_at,user_id,conversation_id,message_id,single_message_risk,conversation_chain_risk,trend,risk_type,decision,reason")
        .gte("created_at", start_iso)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
        .data or []
    )

    by_conv = {}

    for r in rows:
        conv_id = r.get("conversation_id")
        if not conv_id:
            continue

        if conv_id not in by_conv:
            by_conv[conv_id] = {
                "conversation_id": conv_id,
                "user_id": r.get("user_id"),
                "risk_type": r.get("risk_type"),
                "reviews": [],
                "max_chain_risk": 0,
                "latest_decision": r.get("decision"),
                "latest_reason": r.get("reason"),
                "last_seen_at": r.get("created_at"),
                "escalations": 0,
            }

        chain_risk = int(r.get("conversation_chain_risk") or 0)
        by_conv[conv_id]["reviews"].append(r)
        by_conv[conv_id]["max_chain_risk"] = max(by_conv[conv_id]["max_chain_risk"], chain_risk)

        if r.get("decision") in ["DERAD_ONLY", "ESCALATE_FOR_REVIEW"]:
            by_conv[conv_id]["escalations"] += 1

    output = []

    for item in by_conv.values():
        risks = [
            int(r.get("conversation_chain_risk") or 0)
            for r in item["reviews"]
        ]

        avg_chain_risk = round(sum(risks) / len(risks), 2) if risks else 0

        trend = "stable"

        if len(risks) >= 2:
            latest_risk = risks[0]
            oldest_risk = risks[-1]

            if latest_risk > oldest_risk:
                trend = "rising"
            elif latest_risk < oldest_risk:
                trend = "falling"

        score = calculate_leaderboard_score(
            trigger_count=len(item["reviews"]),
            unique_conversations=1,
            avg_chain_risk=avg_chain_risk,
            max_severity=item["max_chain_risk"],
            trend=trend,
            escalations=item["escalations"],
        )

        output.append({
            "conversation_id": item["conversation_id"],
            "user_id": item["user_id"],
            "risk_type": item["risk_type"],
            "review_count": len(item["reviews"]),
            "max_chain_risk": item["max_chain_risk"],
            "avg_chain_risk": avg_chain_risk,
            "trend": trend,
            "latest_decision": item["latest_decision"],
            "latest_reason": item["latest_reason"],
            "leaderboard_score": score,
            "last_seen_at": item["last_seen_at"],
        })

    output.sort(key=lambda x: x["leaderboard_score"], reverse=True)
    return output[:limit]


def get_candidate_patterns_admin(status: str = "pending", limit: int = 100):
    q = (
        secure_supabase
        .table("guardrail_candidate_patterns")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
    )

    if status != "all":
        q = q.eq("status", status)

    return q.execute().data or []

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

    ctx = get_learning_context()
    sys = apply_learning_support(sys, personality, ctx)

    ls = (personality or {}).get("learning_support") or []
    ls_text = " ".join(ls).lower() if isinstance(ls, list) else str(ls).lower()

    if "sen" in ls_text:
        sys += " Use simple language, short steps, and check understanding. Avoid jargon."
    elif "dyslexia" in ls_text:
        sys += " Use short sentences, clear structure, and avoid dense paragraphs."
    elif "adhd" in ls_text:
        sys += (
            " Respond in 4 short bullet points maximum. "
            "Each bullet must be 1 short sentence. "
            "Then ask 1 quick check question."
        )
    elif "eal" in ls_text:
        sys += " Use simple English, define key words, and give one example."

    runtime = get_runtime_context()
    policy = get_memory_policy(runtime.get("company_id"), runtime.get("plan_code", "free"))

    # Short-term memory: only for consensus, only if enabled
    if purpose == "consensus" and policy.get("memory_short_enabled", True):
        thread_messages = fetch_recent_thread_messages(runtime.get("conversation_id"), limit=6)
        thread_text = format_thread_memory_for_prompt(thread_messages)
        if thread_text:
            sys += "\n\nRecent conversation context:\n" + thread_text

    # Medium-term memory: only for consensus, only if enabled
    if purpose == "consensus" and policy.get("memory_medium_enabled", False):
        sys = inject_episodic_memory(sys)

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

    # after you compute answer + tokens
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = int(getattr(usage, "prompt_tokens", prompt_tokens))
        completion_tokens = int(getattr(usage, "completion_tokens", completion_tokens))

    # build meta from session/user (MVP-safe defaults)
    meta = {
        "company_id": user["company_id"],
        "user_id": user["id"],
        "bill_to_company_id": user.get("company_id"),
        "plan_code": get_user_plan(user["id"]),
        "agent_id": st.session_state.get("active_agent_id"),
        "request_type": purpose,
    }

    ctx = get_learning_context()

    meta["external_user_id"] = ctx.get("external_user_id") or get_qp("external_user_id")
    meta["subject"] = ctx.get("subject") or get_qp("subject")
    meta["year_group"] = ctx.get("year_group") or get_qp("year_group")
    meta["topic"] = ctx.get("topic") or get_qp("topic")

    cost_est = log_llm_usage(
    company_id=meta["company_id"],
    user_id=meta["user_id"],
    bill_to_company_id=meta["bill_to_company_id"],
    external_user_id=meta["external_user_id"],
    plan_code=meta["plan_code"],
    agent_id=meta["agent_id"],
    request_type=meta["request_type"],
    subject=meta["subject"],
    year_group=meta["year_group"],
    topic=meta["topic"],
    provider="openai",
    model=model,
    prompt_tokens=prompt_tokens,
    completion_tokens=completion_tokens,
)

    return answer, (prompt_tokens + completion_tokens), float(cost_est or 0.0)

def resolve_agent_id_from_subject(subject: str | None) -> str | None:
    settings = st.session_state.get("company_settings") or {}
    subject_map = settings.get("subject_agent_map") or {}
    fallback = settings.get("fallback_agent_id")
    s = (subject or "").strip()
    return subject_map.get(s) or fallback

def get_teacher_intelligence_fallback_agent_id() -> str | None:
    """
    Final safety fallback for embedded student mode.
    Used when subject_agent_map and fallback_agent_id are missing.
    Looks for a public teaching/tutor agent.
    """
    try:
        rows = (
            secure_supabase
            .table("agents")
            .select("id, agent_name, description, is_public")
            .eq("is_public", True)
            .execute()
            .data or []
        )

        preferred_terms = [
            "teacher intelligence",
            "general tutor",
            "teacher",
            "tutor",
            "education",
            "learning",
        ]

        for term in preferred_terms:
            for row in rows:
                name = (row.get("agent_name") or "").lower()
                desc = (row.get("description") or "").lower()

                if term in name or term in desc:
                    return row.get("id")

    except Exception as e:
        print(f"Teacher fallback lookup failed: {e}")

    return None

def get_qp(name: str, default=None):
    """Read a URL query param safely across Streamlit versions."""
    try:
        v = st.query_params.get(name)
        if isinstance(v, list):
            return v[0] if v else default
        return v if v is not None else default
    except Exception:
        qp = st.experimental_get_query_params()
        return (qp.get(name) or [default])[0]

def apply_learning_support(sys: str, personality: dict | None, ctx: dict | None = None) -> str:
    """
    Adds enforceable formatting/behaviour rules based on learning support preferences.
    Keep MVP-safe: predictable structure and short responses.
    """
    personality = personality or {}
    ctx = ctx or {}

    # Learning support selector normalisation (works for list or string)
    ls = personality.get("learning_support") or ctx.get("support_profile") or ""
    ls_text = " ".join(ls).lower() if isinstance(ls, list) else str(ls).lower()

    # Optional behaviour flags (if you store them later)
    examples_first = bool(personality.get("examples_first", False))
    check_understanding = bool(personality.get("check_understanding", False))
    voice_mode = bool(ctx.get("voice_mode", False)) or bool(personality.get("prefer_voice", False))

    # Always-safe tutor tone (good for displaced learners and SEN without needing to mention trauma)
    sys += (
        " Be calm, kind, and non-judgemental. "
        "Avoid shaming language. "
        "Use neutral examples that do not assume culture, location, or background. "
    )

    # Voice-mode (spoken-friendly)
    if voice_mode:
        sys += (
            " Voice mode: use very short sentences. "
            "One idea at a time. "
            "Avoid long bullet lists. "
        )

    # Support profiles (make them enforceable)
    if "Special Educational Needs" in ls_text:
        sys += (
            " SEN support: explain in tiny steps. "
            "Use 3 short bullet points maximum. "
            "No paragraphs"
        )

    elif "Dyslexia" in ls_text:
        sys += (
            " Dyslexia-friendly: use short lines and lots of spacing. "
            "Avoid dense paragraphs. "
            "Use headings and bullets. "
            "Do not use italics or complex punctuation. "
            "Max 140 words unless asked for more. "
        )

    elif "ADHD" in ls_text:
        sys += (
            " ADHD-friendly: IMPORTANT formatting rule. "
            "Output MUST be 4 bullet points maximum. "
            "Each bullet MUST be 1 short sentence. "
            "No paragraphs. "
        )

    elif "EAL" in ls_text:
        sys += (
            " EAL support: use simple English. "
            "Define any key word in 1 short sentence. "
            "Give 1 clear example. "
            "Avoid idioms and slang. "
            "Max 140 words unless asked for more. "
        )

    # Optional behaviour modifiers
    if examples_first:
        sys += " Start with a simple example before explaining rules."
    if check_understanding:
        sys += " End with ONE quick check question (e.g., 'Does that make sense?')."

    return sys

def get_learning_context() -> dict:
    """Single source of truth for embed/topic context used across prompts + logging."""
    return st.session_state.get("learning_context") or {}

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

    ctx = get_learning_context()
    sys = apply_learning_support(sys, personality, ctx)

    ls = (personality or {}).get("learning_support")
    if ls and "SEN" in ls:
        sys += " Use simple language, short steps, and check understanding. Avoid jargon."
    elif ls and "Dyslexia" in ls:
        sys += " Use short sentences, clear structure, and avoid dense paragraphs."
    elif ls and "ADHD" in ls:
        sys += " Keep responses brief, bullet-pointed, and step-by-step."
    elif ls and "EAL" in ls:
        sys += " Use simple English, define key words, and give one example."

    full_input = f"{prompt}\n\n{context}"

    def _call(client, model_name, provider_name):
        resp = client.chat.completions.create(
            model=model_name,
            max_tokens=350,
            messages=[{"role":"system","content":sys},{"role":"user","content":full_input}],
        )
        answer = resp.choices[0].message.content.strip()

        usage = getattr(resp, "usage", None)
        if usage:
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
        else:
            pt = count_tokens(full_input)
            ct = count_tokens(answer)

        ctx = get_learning_context()

        cost_est = log_llm_usage(
            company_id=user["company_id"],
            user_id=user["id"],
            bill_to_company_id=user.get("company_id"),
            external_user_id=ctx.get("external_user_id") or get_qp("external_user_id"),
            plan_code=get_user_plan(user["id"]),
            agent_id=st.session_state.get("active_agent_id"),
            request_type="agent_turn",
            subject=ctx.get("subject") or get_qp("subject"),
            year_group=ctx.get("year_group") or get_qp("year_group"),
            topic=ctx.get("topic") or get_qp("topic"),
            provider=provider_name,
            model=model_name,
            prompt_tokens=pt,
            completion_tokens=ct,
        )

        return answer, (pt + ct), float(cost_est or 0.0)

    try:
        if provider == "grok" and grok_client:
            return _call(grok_client, XAI_MODEL, "grok")
        if provider == "deepseek" and deepseek_client:
            return _call(deepseek_client, DEEPSEEK_MODEL, "deepseek")

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

# === Model Cost Helpers ===

from decimal import Decimal

def get_model_cost_rate(provider: str, model: str):
    rows = (secure_supabase.table("model_cost_rates")
        .select("prompt_per_1k_gbp,completion_per_1k_gbp,valid_from")
        .eq("is_active", True)
        .eq("provider", provider)
        .eq("model", model)
        .order("valid_from", desc=True)
        .limit(1)
        .execute().data) or []
    return rows[0] if rows else None

def list_pricing_rules(company_id: str):
    return (secure_supabase.table("pricing_rules")
        .select("*")
        .eq("is_active", True)
        .or_(f"company_id.is.null,company_id.eq.{company_id}")
        .execute().data) or []

def pick_pricing_rule(rules, *, company_id, plan_code, agent_id, request_type, provider, model):
    def match(r):
        def ok(field, val):
            return (r.get(field) is None) or (str(r.get(field)) == str(val))
        # company_id can be NULL (global) or exact match
        return (
            (r.get("company_id") is None or str(r.get("company_id")) == str(company_id)) and
            ok("plan_code", plan_code) and
            ok("agent_id", agent_id) and
            ok("request_type", request_type) and
            ok("provider", provider) and
            ok("model", model)
        )

    candidates = [r for r in rules if match(r)]
    if not candidates:
        return None

    def specificity(r):
        keys = ["company_id","plan_code","agent_id","request_type","provider","model"]
        return sum(1 for k in keys if r.get(k) is not None)

    # lowest priority number wins; if tie, more specific wins
    candidates.sort(key=lambda r: (int(r.get("priority", 100)), -specificity(r)))
    return candidates[0]

def compute_costs_gbp(*, provider, model, prompt_tokens, completion_tokens, pricing_rule, default_markup_pct=Decimal("10")):
    rate = get_model_cost_rate(provider, model)
    if rate:
        p = Decimal(str(rate["prompt_per_1k_gbp"]))
        c = Decimal(str(rate["completion_per_1k_gbp"]))
        provider_cost = (Decimal(prompt_tokens)/Decimal(1000))*p + (Decimal(completion_tokens)/Decimal(1000))*c
    else:
        provider_cost = Decimal("0")

    # Sell price: either explicit sell rates OR markup over provider cost
    if pricing_rule and pricing_rule.get("sell_prompt_per_1k_gbp") is not None and pricing_rule.get("sell_completion_per_1k_gbp") is not None:
        sp = Decimal(str(pricing_rule["sell_prompt_per_1k_gbp"]))
        sc = Decimal(str(pricing_rule["sell_completion_per_1k_gbp"]))
        sell_price = (Decimal(prompt_tokens)/Decimal(1000))*sp + (Decimal(completion_tokens)/Decimal(1000))*sc
        markup_pct = pricing_rule.get("markup_pct")  # optional
    else:
        markup_pct = Decimal(str((pricing_rule or {}).get("markup_pct", default_markup_pct)))
        sell_price = provider_cost * (Decimal("1") + (markup_pct/Decimal("100")))

    return float(provider_cost), float(sell_price), float(markup_pct) if markup_pct is not None else None, (pricing_rule or {}).get("id")

# === One logging function that writes every call to llm_usage ===

def log_llm_usage(
    *,
    company_id: str,
    user_id: str,
    bill_to_company_id: str | None,
    external_user_id: str | None,
    plan_code: str | None,
    agent_id: str | None,
    request_type: str | None,
    subject: str | None,
    year_group: str | None,
    topic: str | None,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
):
    bill_to = bill_to_company_id or company_id

    rules = list_pricing_rules(bill_to)
    rule = pick_pricing_rule(
        rules,
        company_id=bill_to,
        plan_code=plan_code,
        agent_id=agent_id,
        request_type=request_type,
        provider=provider,
        model=model,
    )

    provider_cost_gbp, sell_price_gbp, markup_pct, pricing_rule_id = compute_costs_gbp(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pricing_rule=rule,
    )

    row = {
        "company_id": company_id,
        "user_id": user_id,
        "bill_to_company_id": bill_to,
        "external_user_id": external_user_id,
        "plan_code": plan_code,
        "agent_id": agent_id,
        "request_type": request_type,
        "subject": subject,
        "year_group": year_group,
        "topic": topic,
        "provider": provider,
        "model": model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost_cost_gbp": provider_cost_gbp,
        "sell_price_gbp": sell_price_gbp,
        "pricing_rule_id": pricing_rule_id,
        "markup_pct": markup_pct,

        # Backwards compatibility if you still use these elsewhere:
        "base_cost_gbp": provider_cost_gbp,
        "billed_cost_gbp": sell_price_gbp,
    }

    secure_supabase.table("llm_usage").insert(row).execute()

    # MVP: return gross/sell cost so UI can display it
    try:
        return float(row.get("sell_price_gbp") or 0.0)
    except Exception:
        return 0.0

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

def get_or_create_free_conversation(user: dict):
    """
    Creates or reuses a database-backed conversation.
    For embedded students, reuse their latest conversation so history appears.
    """
    if st.session_state.get("free_conversation_id"):
        return st.session_state["free_conversation_id"]

    is_embedded = st.session_state.get("is_embedded", False)

    if is_embedded:
        existing = (
            secure_supabase
            .table("mvp_conversations")
            .select("id,title,created_at")
            .eq("company_id", user.get("company_id"))
            .eq("created_by", user.get("id"))
            .eq("title", "Embedded student conversation")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data or []
        )

        if existing:
            st.session_state["free_conversation_id"] = existing[0]["id"]
            return existing[0]["id"]

    title = "Embedded student conversation" if is_embedded else "Free tier conversation"

    conv = (
        secure_supabase
        .table("mvp_conversations")
        .insert({
            "title": title,
            "company_id": user.get("company_id"),
            "created_by": user.get("id"),
        })
        .execute()
        .data[0]
    )

    st.session_state["free_conversation_id"] = conv["id"]
    return conv["id"]

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

# ---- Projects minimal ----
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
        personality = data.get("personality") or {}
    except Exception:
        personality = {}

    return {
        "tone": personality.get("tone", "Balanced"),
        "style": personality.get("style", "Concise"),
        "citations": personality.get("citations", True),
        "max_words": personality.get("max_words", 180),

        # ✅ add these
        "learning_support": personality.get("learning_support", "Prefer not to say"),
        "sen_self_declared": bool(personality.get("sen_self_declared", False)),
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
                    key = f"logos/{uuid.uuid.uuid4()}_{new_logo.name}"
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
        except Exception as e:
            print(f"Could not restore Supabase session: {e}")

            st.session_state.pop("sb_access_token", None)
            st.session_state.pop("sb_refresh_token", None)
            st.session_state.pop("user", None)

def _get_qp(name: str):
    """Read query params across Streamlit versions."""
    try:
        v = st.query_params.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        qp = st.experimental_get_query_params()
        return (qp.get(name) or [None])[0]


def verify_embed_token(token: str) -> dict | None:
    if not EMBED_SHARED_SECRET:
        st.error("EMBED_SHARED_SECRET is not set in your environment.")
        return None
    try:
        # requires exp if you set it in Phase 4
        payload = jwt.decode(
            token,
            EMBED_SHARED_SECRET,
            algorithms=["HS256"],
            options={"require": ["exp"]},
        )
        return payload
    except jwt.ExpiredSignatureError:
        st.warning("This help link has expired. Please reopen help from the app.")
        return None
    except Exception:
        st.error("Invalid help link (embed token).")
        return None


def get_or_create_embedded_user(company_id: str, external_user_id: str) -> dict | None:
    """
    Finds or creates the internal Users row and the external_identities mapping.
    Uses secure_supabase (service role) because embedded students have no Supabase Auth session.
    """
    # 1) Resolve via external_identities
    row = (
        secure_supabase.table("external_identities")
        .select("user_id")
        .eq("company_id", company_id)
        .eq("external_user_id", external_user_id)
        .maybe_single()
        .execute()
        .data
        or {}
    )

    uid = row.get("user_id")

    if not uid:
        # 2) Optional auto-provision (works if Users has no FK to auth.users)
        uid = str(uuid.uuid4())
        pseudo_email = f"embedded+{external_user_id}@local"

        try:
            secure_supabase.table("Users").insert(
                {
                    "id": uid,
                    "email": pseudo_email,
                    "company_id": company_id,
                    "is_admin": False,
                    "is_agent": False,
                    "plan": "free",
                }
            ).execute()

            secure_supabase.table("external_identities").insert(
                {
                    "company_id": company_id,
                    "external_user_id": external_user_id,
                    "user_id": uid,
                }
            ).execute()
        except Exception as e:
            st.error(
                "No external identity mapping exists, and auto-provisioning failed. "
                "Create a row in external_identities for this student."
            )
            st.exception(e)
            return None

    # Minimal user dict (so your sidebar doesn’t KeyError on user['email']) :contentReference[oaicite:2]{index=2}
    return {
        "id": uid,
        "email": f"embedded:{external_user_id}",
        "company_id": company_id,
        "is_admin": False,
    }


def try_bootstrap_embedded_session() -> bool:
    """
    If embed_token is present and valid:
      - sets st.session_state['user']
      - marks embedded mode
      - stores learning_context (subject/year/topic etc.)
    """
    token = _get_qp("embed_token")
    if not token:
        return False

    payload = verify_embed_token(token)
    if not payload:
        st.stop()

    company_id = payload.get("company_id")
    external_user_id = payload.get("external_user_id")

    if not company_id or not external_user_id:
        st.error("Embed token is missing company_id or external_user_id.")
        st.stop()

    user = get_or_create_embedded_user(company_id, external_user_id)
    if not user:
        st.stop()

    st.session_state["user"] = user
    st.session_state["is_embedded"] = True

    st.session_state["learning_context"] = {
        "external_user_id": external_user_id,
        "company_id": company_id,
        "subject": payload.get("subject"),
        "year_group": payload.get("year_group"),
        "topic": payload.get("topic"),
        "support_profile": payload.get("support_profile") or [],
        "tier": payload.get("tier", "free"),
        "source": payload.get("source", "school_app"),
        "voice_mode": bool(payload.get("voice_mode", False)),
    }

    return True

# --- Auth bootstrap (embedded first, otherwise normal login) ---
if not try_bootstrap_embedded_session():
    restore_supabase_session()

    try:
        sess = supabase.auth.get_session()
    except Exception as e:
        print(f"Supabase session refresh failed: {e}")

        st.session_state.pop("sb_access_token", None)
        st.session_state.pop("sb_refresh_token", None)
        st.session_state.pop("user", None)

        try:
            supabase.auth.sign_out()
        except Exception:
            pass

        st.warning("Your session has expired. Please log in again.")
        login_page()
        st.stop()

    if not (sess and sess.user):
        st.session_state.pop("user", None)
        login_page()
        st.stop()

# --------- Embedded mobile styling ---------
if st.session_state.get("is_embedded"):
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }

            [data-testid="collapsedControl"] {
                display: none;
            }

            .block-container {
                padding-top: 0.75rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                max-width: 100%;
            }

            h1 {
                font-size: 1.4rem !important;
            }

            h2, h3 {
                font-size: 1.1rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

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
db = secure_supabase if st.session_state.get("is_embedded") else supabase

comp = db.table("companies") \
    .select("name,logo_url,settings") \
    .eq("id", user["company_id"]) \
    .maybe_single() \
    .execute().data or {}

st.session_state["company_settings"] = comp.get("settings") or {}

# ---------------- Set active agent from embed context (preferred) ----------------
ctx = get_learning_context()

# Prefer embed token context; fall back to URL param for local testing
subject = ctx.get("subject") or get_qp("subject")

if subject:
    agent_id = resolve_agent_id_from_subject(subject)

    # Final fallback if company settings do not contain subject mapping or fallback_agent_id
    if not agent_id and st.session_state.get("is_embedded"):
        agent_id = get_teacher_intelligence_fallback_agent_id()

    st.session_state["active_agent_id"] = agent_id

    if st.session_state.get("is_embedded"):
        # Use DeepSeek by default for lower-cost teaching intelligence.
        # Change to "openai" if you want OpenAI as the default.
        st.session_state["agent_provider"] = os.getenv("EMBED_AGENT_PROVIDER", "deepseek").lower()

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
st.sidebar.caption(f"DEBUG saved learning_support: {(_curr or {}).get('learning_support')}")

tone  = st.sidebar.selectbox("Tone", ["Balanced","Analytical","Strategic","Supportive","Challenger"],
                             index=["Balanced","Analytical","Strategic","Supportive","Challenger"].index(_curr.get("tone","Balanced")))
style = st.sidebar.selectbox("Style", ["Concise","Detailed"], index=0 if _curr.get("style","Concise")=="Concise" else 1)

support_options = [
    "Prefer not to say",
    "No additional support",
    "Special Educational Needs (SEN) — self-declared",
    "Dyslexia-friendly support",
    "ADHD-friendly support",
    "English as an Additional Language (EAL)",
]

support = st.sidebar.selectbox(
    "Learning support (optional)",
    support_options,
    index=support_options.index(_curr.get("learning_support", "Prefer not to say"))
        if _curr.get("learning_support") in support_options else 0
)

cit   = st.sidebar.checkbox("Include sources/citations", value=bool(_curr.get("citations", True)))
mxw   = st.sidebar.slider("Max words", min_value=80, max_value=500, value=int(_curr.get("max_words", 180)), step=10)

if st.sidebar.button("💾 Save personality"):
    save_user_personality(user["id"], user["company_id"], {
        "tone": tone,
        "style": style,
        "citations": cit,
        "max_words": mxw,
        "learning_support": support,
        "sen_self_declared": (support == "Special Educational Needs (SEN) — self-declared"),
    })
    st.sidebar.success("Saved.")

# --------- Admin Panel Updated: Agent Role Prompts ---------

def admin_page():
    st.title("🛠️ Admin Panel")
    st.markdown("Manage agents, their descriptions, and role-based prompt templates.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Manage Agents",
    "Role Prompts",
    "Guardrails",
    "Guardrail Events",
    "Risk Intelligence",
    "Analytics Dashboard",
    "Usage & Cost",
    "Human Signals",
    "Safety Signals Reviewer",
    ])

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

    # --- TAB 3: Guardrails ---
    with tab3:
        st.subheader("🛡️ Guardrail Management")

        # Kill switch
        current_enabled = guardrails_enabled()
        new_enabled = st.toggle(
            "Guardrails enabled",
            value=current_enabled,
            help="Emergency rollback switch",
        )
        if new_enabled != current_enabled:
            set_runtime_flag("guardrails_enabled", "true" if new_enabled else "false")
            st.success(f"Guardrails {'enabled' if new_enabled else 'disabled'}.")
            st.rerun()

        st.markdown("---")
        st.markdown("### Trigger filters")
        c1, c2, c3 = st.columns(3)
        flt_lang = c1.selectbox("Language", ["all", "en", "ar", "any"], index=0, key="gr_filter_lang")
        flt_sev = c2.selectbox(
            "Severity",
            ["all", "3", "2", "1"],
            index=0,
            key="gr_filter_sev",
            format_func=lambda x: {"all": "all", "1": "low", "2": "medium", "3": "high"}.get(x, x),
        )
        flt_status = c3.selectbox("Status", ["all", "active", "inactive"], index=0, key="gr_filter_status")

        triggers = list_guardrail_triggers_admin(language=flt_lang, severity=flt_sev, active=flt_status)

        st.markdown("### Existing triggers")
        if not triggers:
            st.info("No triggers found for the selected filters.")

        for t in triggers:
            tid = t["id"]
            sev_label = _sev_label(t.get("severity"))
            state_badge = "✅" if t.get("is_active") else "⛔"
            title = f"{state_badge} {sev_label.upper()} | {t.get('language','-')} | {t.get('category','general')} | {t.get('pattern','')}"
            with st.expander(title):
                left, right = st.columns([2, 1])

                pattern = left.text_input("Pattern", value=t.get("pattern", ""), key=f"gr_pattern_{tid}")

                mt_options = ["keyword", "phrase", "regex"]
                current_mt = (t.get("match_type") or "phrase").lower()
                mt_idx = mt_options.index(current_mt) if current_mt in mt_options else 1
                match_type = left.selectbox("Match type", mt_options, index=mt_idx, key=f"gr_mt_{tid}")

                lang_options = ["en", "ar", "any"]
                current_lang = (t.get("language") or "en").lower()
                lang_idx = lang_options.index(current_lang) if current_lang in lang_options else 0
                language = left.selectbox("Language", lang_options, index=lang_idx, key=f"gr_lang_{tid}")

                category = left.text_input("Category", value=t.get("category") or "general", key=f"gr_cat_{tid}")
                notes = left.text_area("Notes", value=t.get("notes") or "", height=70, key=f"gr_notes_{tid}")

                sev_options = [1, 2, 3]
                curr_sev = int(t.get("severity") or 1)
                sev_idx = sev_options.index(curr_sev) if curr_sev in sev_options else 0
                severity = right.selectbox(
                    "Severity",
                    sev_options,
                    index=sev_idx,
                    key=f"gr_sev_{tid}",
                    format_func=lambda x: f"{x} ({_sev_label(x)})",
                )

                action_options = ["CONFIRM", "ADD_DERAD_NOTE", "DERAD_ONLY", "ALLOW"]
                current_action = (t.get("action") or "CONFIRM").upper()
                act_idx = action_options.index(current_action) if current_action in action_options else 0
                action = right.selectbox("Action", action_options, index=act_idx, key=f"gr_act_{tid}")

                is_active = right.checkbox("Active", value=bool(t.get("is_active")), key=f"gr_active_{tid}")

                b1, b2 = st.columns(2)
                if b1.button("Save trigger", key=f"gr_save_{tid}"):
                    try:
                        save_guardrail_trigger_admin(
                            trigger_id=tid,
                            pattern=pattern,
                            match_type=match_type,
                            language=language,
                            category=category,
                            severity=int(severity),
                            action=action,
                            is_active=is_active,
                            notes=notes,
                            created_by=user["id"] if "user" in globals() and user else None,
                        )
                        st.success("Trigger saved.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

                if b2.button("Disable" if is_active else "Enable", key=f"gr_toggle_{tid}"):
                    try:
                        set_guardrail_trigger_active(tid, not is_active)
                        st.success("Trigger status updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.markdown("---")
        st.subheader("➕ Add New Trigger")

        l1, l2 = st.columns([2, 1])

        new_pattern = l1.text_input("Pattern", key="gr_new_pattern")
        new_match_type = l1.selectbox("Match type", ["keyword", "phrase", "regex"], index=1, key="gr_new_mt")
        new_language = l1.selectbox("Language", ["en", "ar", "any"], index=0, key="gr_new_lang")
        new_category = l1.text_input("Category", value="general", key="gr_new_category")
        new_notes = l1.text_area("Notes", height=70, key="gr_new_notes")

        new_severity = l2.selectbox(
            "Severity",
            [1, 2, 3],
            index=1,
            key="gr_new_sev",
            format_func=lambda x: f"{x} ({_sev_label(x)})",
        )
        new_action = l2.selectbox(
            "Action",
            ["CONFIRM", "ADD_DERAD_NOTE", "DERAD_ONLY", "ALLOW"],
            index=0,
            key="gr_new_action",
        )
        new_active = l2.checkbox("Active", value=True, key="gr_new_active")

        if st.button("Add trigger", key="gr_add"):
            try:
                save_guardrail_trigger_admin(
                    trigger_id=None,
                    pattern=new_pattern,
                    match_type=new_match_type,
                    language=new_language,
                    category=new_category,
                    severity=int(new_severity),
                    action=new_action,
                    is_active=new_active,
                    notes=new_notes,
                    created_by=user["id"] if "user" in globals() and user else None,
                )
                st.success("Trigger added.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # --- TAB 4: Guardrail Events ---
    with tab4:
        st.subheader("📊 Guardrail Events Dashboard")

        days = st.selectbox(
            "Window",
            [1, 7, 14, 30],
            index=1,
            format_func=lambda d: f"Last {d} day(s)",
            key="guardrail_events_window"
        )

        metrics = get_guardrail_metrics_admin(days=days)

        # ---------------- KPI Cards ----------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Checks", metrics["total_checks"])
        m2.metric("Triggered", metrics["triggered"])
        m3.metric("Trigger rate", f"{metrics['trigger_rate_pct']}%")
        m4.metric(
            "Avg latency",
            f"{metrics['avg_latency_ms']} ms" if metrics["avg_latency_ms"] is not None else "n/a"
        )

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Allow Normal", metrics["decision_counts"].get("ALLOW_NORMAL", 0))
        d2.metric(
            "Safety Note",
            metrics["decision_counts"].get("ADD_DERAD_NOTE", 0)
            + metrics["decision_counts"].get("ADD_SAFETY_NOTE", 0)
        )
        d3.metric("DeRad Only", metrics["decision_counts"].get("DERAD_ONLY", 0))
        d4.metric("Escalate", metrics["decision_counts"].get("ESCALATE_FOR_REVIEW", 0))

        s1, s2, s3 = st.columns(3)
        s1.metric("High Severity", metrics["severity_counts"].get(3, 0))
        s2.metric("Medium Severity", metrics["severity_counts"].get(2, 0))
        s3.metric("Low Severity", metrics["severity_counts"].get(1, 0))

        st.markdown("---")

        # ---------------- Visual Dashboard ----------------
        st.markdown("### 📈 Visual Summary")

        chart_rows = list_guardrail_events_admin(
            limit=1000,
            severity="all",
            language="all",
            decision="all"
        )

        if chart_rows:
            chart_df = pd.DataFrame(chart_rows)

            c1, c2 = st.columns(2)

            with c1:
                if "decision" in chart_df.columns:
                    decision_df = (
                        chart_df.groupby("decision")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.pie(
                        decision_df,
                        names="decision",
                        values="count",
                        title="Decision Breakdown"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab4_chart_1")

            with c2:
                if "max_severity" in chart_df.columns:
                    severity_df = (
                        chart_df.groupby("max_severity")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.bar(
                        severity_df,
                        x="max_severity",
                        y="count",
                        title="Events by Severity"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab4_chart_2")

            if "created_at" in chart_df.columns:
                chart_df["date"] = pd.to_datetime(chart_df["created_at"]).dt.date

                daily_df = (
                    chart_df.groupby("date")
                    .size()
                    .reset_index(name="events")
                )

                fig = px.line(
                    daily_df,
                    x="date",
                    y="events",
                    title="Guardrail Events Over Time"
                )
                st.plotly_chart(fig, use_container_width=True, key="tab4_chart_3")

        else:
            st.info("No chart data available yet.")

        st.markdown("---")

        # ---------------- Filters ----------------
        st.markdown("### 🔎 Event Filters")

        f1, f2, f3 = st.columns(3)

        ev_sev = f1.selectbox(
            "Severity",
            ["all", "3", "2", "1"],
            index=0,
            key="ev_filter_sev",
            format_func=lambda x: {
                "all": "all",
                "1": "low",
                "2": "medium",
                "3": "high"
            }.get(x, x),
        )

        ev_lang = f2.selectbox(
            "Language",
            ["all", "en", "ar"],
            index=0,
            key="ev_filter_lang"
        )

        ev_dec = f3.selectbox(
            "Decision",
            ["all", "ALLOW_NORMAL", "ADD_DERAD_NOTE", "ADD_SAFETY_NOTE", "DERAD_ONLY", "ESCALATE_FOR_REVIEW"],
            index=0,
            key="ev_filter_dec",
        )

        rows = list_guardrail_events_admin(
            limit=200,
            severity=ev_sev,
            language=ev_lang,
            decision=ev_dec
        )

        # ---------------- Event Log ----------------
        st.markdown("### 📄 Event Log")

        if not rows:
            st.info("No events found.")
        else:
            for r in rows:
                ts = (r.get("created_at") or "")[:19]
                sev = _sev_label(r.get("max_severity"))
                lang = r.get("language") or "-"
                decision = r.get("decision") or "UNKNOWN"

                with st.expander(f"{ts} | {decision} | {sev} | {lang}"):
                    st.write("**Input**")
                    st.write(r.get("input_text") or "")

                    st.write("**Matched patterns**")
                    st.json(r.get("matched_patterns") or [])

                    st.write("**Latency**")
                    st.write(
                        f"{r.get('latency_ms')} ms"
                        if r.get("latency_ms") is not None
                        else "n/a"
                    )

                    st.write("**IDs**")
                    st.write({
                        "user_id": r.get("user_id"),
                        "conversation_id": r.get("conversation_id"),
                        "message_id": r.get("message_id"),
                    })

    # --- TAB 5: Risk Intelligence ---
    with tab5:
        st.subheader("🛡 Risk Intelligence Leaderboard")

        days = st.selectbox(
            "Leaderboard window",
            [1, 7, 14, 30],
            index=3,
            format_func=lambda d: f"Last {d} day(s)",
            key="risk_lb_days"
        )

        lb_tab1, lb_tab2, lb_tab3 = st.tabs([
            "Top Triggered Expressions",
            "Rising Conversation Chains",
            "Candidate New Guardrails",
        ])

        with lb_tab1:
            st.markdown("### Top Triggered Expressions")

            pattern_rows = get_top_guardrail_patterns(days=days, limit=20)

            if not pattern_rows:
                st.info("No guardrail pattern data yet.")
            else:
                for idx, r in enumerate(pattern_rows, start=1):
                    with st.expander(
                        f"#{idx} | Score {r['leaderboard_score']} | {r['pattern']}"
                    ):
                        st.write({
                            "Trigger count": r["trigger_count"],
                            "Unique conversations": r["unique_conversation_count"],
                            "Unique users": r["unique_user_count"],
                            "Max severity": r["max_severity"],
                            "Escalations": r["escalations"],
                            "Last seen": r["last_seen_at"],
                        })
                        st.write("Example event IDs")
                        st.json(r["example_event_ids"])

        with lb_tab2:
            st.markdown("### Rising Conversation Chains")

            chain_rows = get_rising_conversation_chains(days=days, limit=20)

            if not chain_rows:
                st.info("No conversation chain risk data yet.")
            else:
                for idx, r in enumerate(chain_rows, start=1):
                    with st.expander(
                        f"#{idx} | Score {r['leaderboard_score']} | {r['risk_type']} | {r['trend']}"
                    ):
                        st.write({
                            "Conversation ID": r["conversation_id"],
                            "User ID": r["user_id"],
                            "Review count": r["review_count"],
                            "Max chain risk": r["max_chain_risk"],
                            "Average chain risk": r["avg_chain_risk"],
                            "Latest decision": r["latest_decision"],
                            "Last seen": r["last_seen_at"],
                        })
                        st.write("Reason")
                        st.write(r["latest_reason"])

        with lb_tab3:
            st.markdown("### Candidate New Guardrails")

            status_filter = st.selectbox(
                "Status",
                ["pending", "approved", "rejected", "added_to_guardrails", "all"],
                index=0,
                key="candidate_status_filter"
            )

            candidates = get_candidate_patterns_admin(status=status_filter, limit=100)

            if not candidates:
                st.info("No candidate phrases found.")
            else:
                for c in candidates:
                    with st.expander(
                        f"{c.get('phrase')} | {c.get('category')} | severity {c.get('suggested_severity')}"
                    ):
                        st.write("Reason")
                        st.write(c.get("reviewer_reason") or "")

                        st.write("Details")
                        st.json({
                            "language": c.get("language"),
                            "status": c.get("status"),
                            "conversation_id": c.get("source_conversation_id"),
                            "message_id": c.get("source_message_id"),
                            "created_at": c.get("created_at"),
                        })

                        col_a, col_b, col_c = st.columns(3)

                        if col_a.button("Add to live guardrails", key=f"add_candidate_{c['id']}"):
                            try:
                                approve_candidate_as_guardrail(
                                    candidate=c,
                                    reviewed_by=user["id"] if user else None,
                                )
                                st.success("Candidate added to live guardrails.")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                        if col_b.button("Reject", key=f"reject_candidate_{c['id']}"):
                            update_candidate_pattern_status(
                                candidate_id=c["id"],
                                status="rejected",
                                reviewed_by=user["id"] if user else None,
                            )
                            st.success("Candidate rejected.")
                            st.rerun()

                        if col_c.button("Needs specialist review", key=f"specialist_candidate_{c['id']}"):
                            update_candidate_pattern_status(
                                candidate_id=c["id"],
                                status="needs_specialist_review",
                                reviewed_by=user["id"] if user else None,
                            )
                            st.success("Candidate marked for specialist review.")
                            st.rerun()

  # --- TAB 6: Analytics Dashboard ---
    with tab6:
        st.subheader("📊 Analytics Dashboard")

        days = st.selectbox(
            "Time period",
            [7, 14, 30, 90],
            index=2,
            key="analytics_days"
        )

        usage_df = fetch_llm_usage_admin(days)
        guard_df = fetch_guardrail_events_df(days)
        risk_df = fetch_risk_reviews_df(days)

        login_df, company_usage_df = fetch_company_dashboard_analytics(days)

        # ---------------- Main KPI Cards ----------------
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("LLM Calls", len(usage_df))
        c2.metric("Guardrail Events", len(guard_df))
        c3.metric("Risk Reviews", len(risk_df))

        total_revenue = (
            usage_df["sell_price_gbp"].sum()
            if not usage_df.empty and "sell_price_gbp" in usage_df
            else 0
        )

        c4.metric("Estimated Revenue", f"£{total_revenue:.4f}")

        st.markdown("---")

        # ---------------- Company Dashboard Usage ----------------
        st.markdown("### 🏢 Company Dashboard Usage")

        d1, d2, d3, d4 = st.columns(4)

        total_logins = len(login_df) if not login_df.empty else 0

        active_companies = (
            login_df["company_id"].nunique()
            if not login_df.empty and "company_id" in login_df
            else 0
        )

        total_provider_cost = (
            company_usage_df["cost_cost_gbp"].sum()
            if not company_usage_df.empty and "cost_cost_gbp" in company_usage_df
            else 0
        )

        total_company_revenue = (
            company_usage_df["sell_price_gbp"].sum()
            if not company_usage_df.empty and "sell_price_gbp" in company_usage_df
            else 0
        )

        d1.metric("Dashboard Visits", total_logins)
        d2.metric("Active Companies", active_companies)
        d3.metric("LLM Cost Incurred", f"£{total_provider_cost:.4f}")
        d4.metric("Company Revenue", f"£{total_company_revenue:.4f}")

        if not login_df.empty and "company_name" in login_df:
            company_logins = (
                login_df.groupby("company_name")
                .size()
                .reset_index(name="dashboard_visits")
                .sort_values("dashboard_visits", ascending=False)
            )

            fig = px.bar(
                company_logins,
                x="company_name",
                y="dashboard_visits",
                title="Dashboard Visits by Company"
            )
            st.plotly_chart(fig, use_container_width=True, key="tab6_chart_1")

        if not company_usage_df.empty and "company_name" in company_usage_df:
            company_cost = (
                company_usage_df
                .groupby("company_name")[["cost_cost_gbp", "sell_price_gbp"]]
                .sum()
                .reset_index()
                .sort_values("cost_cost_gbp", ascending=False)
            )

            col_cost, col_revenue = st.columns(2)

            with col_cost:
                fig = px.bar(
                    company_cost,
                    x="company_name",
                    y="cost_cost_gbp",
                    title="Usage Cost Incurred by Company"
                )
                st.plotly_chart(fig, use_container_width=True, key="tab6_chart_2")

            with col_revenue:
                fig = px.bar(
                    company_cost,
                    x="company_name",
                    y="sell_price_gbp",
                    title="Recharge / Revenue by Company"
                )
                st.plotly_chart(fig, use_container_width=True, key="tab6_chart_3")

            st.markdown("#### Company Cost Summary")
            st.dataframe(company_cost, use_container_width=True)

        st.markdown("---")

        # ---------------- Platform Usage Trend ----------------
        st.markdown("### 📈 Platform Usage Trend")

        if not usage_df.empty and "created_at" in usage_df:
            usage_df["date"] = pd.to_datetime(usage_df["created_at"]).dt.date

            daily_usage = (
                usage_df.groupby("date")
                .size()
                .reset_index(name="LLM Calls")
            )

            fig = px.bar(
                daily_usage,
                x="date",
                y="LLM Calls",
                title="Daily Platform Usage"
            )
            st.plotly_chart(fig, use_container_width=True, key="tab6_chart_4")
        else:
            st.info("No platform usage data found for this period.")

        st.markdown("---")

        # ---------------- Provider / Model Split ----------------
        st.markdown("### 🤖 Provider and Model Usage")

        if not usage_df.empty:
            col_provider, col_model = st.columns(2)

            with col_provider:
                if "provider" in usage_df:
                    provider_df = (
                        usage_df.groupby("provider")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.pie(
                        provider_df,
                        names="provider",
                        values="count",
                        title="Usage by Provider"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab6_chart_5")

            with col_model:
                if "model" in usage_df:
                    model_df = (
                        usage_df.groupby("model")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.bar(
                        model_df,
                        x="model",
                        y="count",
                        title="Usage by Model"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab6_chart_6")
        else:
            st.info("No provider or model usage data found.")

        st.markdown("---")

        # ---------------- Safety Overview ----------------
        st.markdown("### 🛡️ Safety Overview")

        if not guard_df.empty:
            col_decision, col_severity = st.columns(2)

            with col_decision:
                if "decision" in guard_df:
                    decision_df = (
                        guard_df.groupby("decision")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.pie(
                        decision_df,
                        names="decision",
                        values="count",
                        title="Guardrail Decisions"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab6_chart_7")

            with col_severity:
                if "max_severity" in guard_df:
                    severity_df = (
                        guard_df.groupby("max_severity")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.bar(
                        severity_df,
                        x="max_severity",
                        y="count",
                        title="Guardrail Severity"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab6_chart_8")
        else:
            st.info("No guardrail data found for this period.")

        st.markdown("---")

        # ---------------- Risk Overview ----------------
        st.markdown("### ⚠️ Risk Review Overview")

        if not risk_df.empty:
            col_risk_type, col_risk_decision = st.columns(2)

            with col_risk_type:
                if "risk_type" in risk_df:
                    risk_type_df = (
                        risk_df.groupby("risk_type")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.bar(
                        risk_type_df,
                        x="risk_type",
                        y="count",
                        title="Risk Types Detected"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab6_chart_9")

            with col_risk_decision:
                if "decision" in risk_df:
                    risk_decision_df = (
                        risk_df.groupby("decision")
                        .size()
                        .reset_index(name="count")
                    )

                    fig = px.pie(
                        risk_decision_df,
                        names="decision",
                        values="count",
                        title="Risk Reviewer Decisions"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="tab6_chart_10")
        else:
            st.info("No risk review data found for this period.")

  # --- TAB 7: Risk Intelligence ---
    with tab7:
        st.subheader("💷 Usage & Cost")

        usage_df = fetch_llm_usage_admin(days)

        if usage_df.empty:
            st.info("No usage data found.")
        else:
            if "model" in usage_df and "sell_price_gbp" in usage_df:
                model_cost = usage_df.groupby("model")["sell_price_gbp"].sum().reset_index()

                fig = px.bar(model_cost, x="model", y="sell_price_gbp", title="Revenue by Model")
                st.plotly_chart(fig, use_container_width=True)

            if "provider" in usage_df:
                provider_counts = usage_df.groupby("provider").size().reset_index(name="count")

                fig = px.pie(provider_counts, names="provider", values="count", title="Usage by Provider")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(usage_df, use_container_width=True)

  # --- TAB 8: Human Signals ---
    with tab8:
        st.subheader("🧠 Human Signals Dashboard")

        signal_df = fetch_human_signals_admin(limit=500)

        if signal_df.empty:
            st.info("No human signals found.")
        else:
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Signals", len(signal_df))

            avg_confidence = signal_df["confidence"].mean() if "confidence" in signal_df else 0
            c2.metric("Avg Confidence", f"{avg_confidence:.2f}")

            escalating = len(signal_df[signal_df["trend"] == "escalating"]) if "trend" in signal_df else 0
            c3.metric("Escalating", escalating)

            safety_notes = len(signal_df[signal_df["decision"] == "ADD_SAFETY_NOTE"]) if "decision" in signal_df else 0
            c4.metric("Safety Notes", safety_notes)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                signal_count = (
                    signal_df.groupby("signal")
                    .size()
                    .reset_index(name="count")
                )

                fig = px.pie(
                    signal_count,
                    names="signal",
                    values="count",
                    title="Signals by Type"
                )
                st.plotly_chart(fig, use_container_width=True, key="human_signals_type_pie")

            with col2:
                trend_count = (
                    signal_df.groupby("trend")
                    .size()
                    .reset_index(name="count")
                )

                fig = px.bar(
                    trend_count,
                    x="trend",
                    y="count",
                    title="Signals by Trend"
                )
                st.plotly_chart(fig, use_container_width=True, key="human_signals_trend_bar")

            if "created_at" in signal_df:
                signal_df["date"] = pd.to_datetime(signal_df["created_at"]).dt.date

                daily_signals = (
                    signal_df.groupby("date")
                    .size()
                    .reset_index(name="signals")
                )

                fig = px.line(
                    daily_signals,
                    x="date",
                    y="signals",
                    title="Human Signals Over Time"
                )
                st.plotly_chart(fig, use_container_width=True, key="human_signals_over_time")

            st.markdown("### Signal Log")
            st.dataframe(signal_df, use_container_width=True)

    with tab9:
        st.subheader("🧠 Safety Signal Reviewer")

        st.caption("Review dataset examples and confirm or correct the mapped signal.")

        signals = fetch_signals_for_dropdown()

        signal_options = {"all": "All signals"}
        signal_options.update({
            s["id"]: s["signal_name"]
            for s in signals
        })

        selected_signal = st.selectbox(
            "Filter by signal",
            options=list(signal_options.keys()),
            format_func=lambda x: signal_options[x],
            key="review_signal_filter"
        )

        examples = fetch_training_examples_for_review(
            limit=10,
            signal_filter=selected_signal
        )

        if not examples:
            st.info("No pending training examples found.")
        else:
            for ex in examples:
                current_signal = (ex.get("signals_catalogue") or {}).get("signal_name", "Unknown")

                with st.expander(f"{current_signal} | {ex.get('mapped_route')} | {ex.get('dataset_category')}"):
                    st.write("**Training text**")
                    st.write(ex.get("text"))

                    st.write("**Current mapped signal**")
                    st.code(current_signal)

                    reviewer_signal_id = st.selectbox(
                        "Reviewer signal",
                        options=[s["id"] for s in signals],
                        format_func=lambda sid: next(
                            (s["signal_name"] for s in signals if s["id"] == sid),
                            sid
                        ),
                        index=max(
                            0,
                            [s["id"] for s in signals].index(ex["signal_id"])
                            if ex.get("signal_id") in [s["id"] for s in signals]
                            else 0
                        ),
                        key=f"reviewer_signal_{ex['id']}"
                    )

                    reviewer_confidence = st.slider(
                        "Reviewer confidence",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.05,
                        key=f"confidence_{ex['id']}"
                    )

                    reviewer_notes = st.text_area(
                        "Reviewer notes",
                        key=f"notes_{ex['id']}"
                    )

                    if st.button("Save review", key=f"save_review_{ex['id']}"):
                        reviewer_id = None
                        if st.session_state.get("user"):
                            reviewer_id = st.session_state["user"].get("id")

                        save_signal_reviewer_feedback(
                            training_example_id=ex["id"],
                            original_signal_id=ex["signal_id"],
                            reviewer_signal_id=reviewer_signal_id,
                            reviewer_confidence=reviewer_confidence,
                            reviewer_notes=reviewer_notes,
                            reviewer_id=reviewer_id,
                        )

                        st.success("Review saved.")
                        st.rerun()

if st.session_state.get("user", {}).get("is_admin", False):
    if st.sidebar.checkbox("🔐 Admin Mode"):
        admin_page()
        st.stop()

st.sidebar.markdown("---")

billing_status = st.query_params.get("billing")
billing_plan = st.query_params.get("plan")

if billing_status == "success" and billing_plan in ["pro", "team", "enterprise"]:
    secure_supabase.table("Users").update({
        "plan": billing_plan
    }).eq("id", user["id"]).execute()

    st.success(f"Payment successful. Your plan is now {billing_plan.title()}.")
    st.query_params.clear()
    st.rerun()

elif billing_status == "cancelled":
    st.warning("Payment cancelled.")
    st.query_params.clear()

def start_stripe_checkout(user, plan_code: str, price_id: str):
    billing_api_url = os.getenv("BILLING_API_URL", "http://localhost:8001")

    res = requests.post(
        f"{billing_api_url}/api/billing/create-checkout-session",
        json={
            "user_id": user["id"],
            "email": user.get("email"),
            "plan_code": plan_code,
            "price_id": price_id,
        },
        timeout=20,
    )

    if not res.ok:
        st.error(f"Billing server returned an error: {res.text}")
        st.stop()

    return res.json()["url"]

# ---- Plan helper (top-level helpers area is fine) ----
def get_user_plan(user_id):
    row = (supabase.table("Users").select("plan").eq("id", user_id).maybe_single().execute().data) or {}
    return (row.get("plan") or "free").lower()

# --- Plan & upgrade ---
plan = get_user_plan(user["id"]).lower()
is_paid = plan in ("pro", "team", "enterprise")
has_projects = plan in ("team", "enterprise")

st.sidebar.caption(f"Plan: {plan.title()}")

if plan == "free":
    st.sidebar.info("Upgrade to unlock more usage and paid features.")

    if st.sidebar.button("Upgrade to Pro"):
        checkout_url = start_stripe_checkout(user, "pro", os.getenv("STRIPE_PRO_PRICE_ID"))
        st.sidebar.success("Stripe checkout is ready.")
        st.sidebar.link_button("Continue to secure Stripe payment", checkout_url)

    if st.sidebar.button("Upgrade to Team"):
        checkout_url = start_stripe_checkout(user, "team", os.getenv("STRIPE_TEAM_PRICE_ID"))
        st.sidebar.success("Stripe checkout is ready.")
        st.sidebar.link_button("Continue to secure Stripe payment", checkout_url)

    if st.sidebar.button("Upgrade to Enterprise"):
        checkout_url = start_stripe_checkout(user, "enterprise", os.getenv("STRIPE_ENTERPRISE_PRICE_ID"))
        st.sidebar.success("Stripe checkout is ready.")
        st.sidebar.link_button("Continue to secure Stripe payment", checkout_url)

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
                        user_msg = post_user_message(
                            conv_id,
                            user["id"],
                            refined_q,
                            meta={"agents_requested": selected_agent_ids, "agents_used": list(allowed_ids)}
                        )

                        # >>> Always-on Conversation Risk Reviewer
                        try:
                            risk_review = review_conversation_risk(
                                conversation_id=conv_id,
                                user_id=user["id"],
                                latest_message=refined_q,
                                latest_message_id=user_msg.get("id"),
                                company_id=user.get("company_id"),
                            )
                        except Exception as e:
                            risk_review = {
                                "decision": "LOG_ONLY",
                                "reason": f"Risk reviewer failed safely: {e}",
                            }
                            st.warning(f"Risk reviewer failed safely: {e}")

                        review_decision = (risk_review.get("decision") or "LOG_ONLY").upper()

                        safety_note = ""

                        if review_decision == "ADD_SAFETY_NOTE":
                            safety_note = (
                                "\n\nSafety instruction: keep the response lawful, non-harmful, "
                                "constructive, calm, and focused on safe guidance."
                            )

                            refined_q = refined_q + safety_note

                        # >>> Human Signals Classifier response routing
                        try:
                            human_classification = classify_human_signal(
                                latest_message=question,
                                conversation_history=thread_text
                            )
                        except Exception as e:
                            human_classification = {
                                "route": "NORMAL",
                                "primary_signal": "none",
                                "reason": f"Classifier failed safely: {e}",
                            }

                        human_route = (human_classification.get("route") or "NORMAL").upper()

                        try:
                            update_user_memory_profile(
                                user_id=user["id"] if user else None,
                                company_id=user.get("company_id") if user else None,
                                conversation_id=free_conv_id,
                                message_id=free_user_msg.get("id"),
                                classification=human_classification,
                                latest_message=question,
                            )
                        except Exception as e:
                            print(f"User memory update failed: {e}")

                        if human_route in ["SOFT_SUPPORT", "ELEVATED_REVIEW", "CRITICAL_SAFETY"]:
                            try:
                                profile = fetch_user_memory_profile(
                                    user["id"],
                                    user.get("company_id")
                                )

                                if profile and profile.get("checkin_prompt"):
                                    human_classification["memory_note"] = profile.get("checkin_prompt")

                            except Exception as e:
                                print(f"Could not fetch user memory profile: {e}")

                            support_reply = run_support_guidance_agent(
                                question,
                                human_classification
                            )
                            st.session_state["messages"].append({
                                "role": "assistant",
                                "content": support_reply
                            })

                            try:
                                post_system_message(
                                    free_conv_id,
                                    f"[Support & Guidance] {support_reply}",
                                    meta={
                                        "human_signal_classification": human_classification,
                                        "conversation_risk_review": risk_review,
                                        "source": "support_guidance_agent"
                                    }
                                )
                            except Exception as e:
                                st.warning(f"Could not save Support & Guidance response: {e}")

                            if human_route in ["ELEVATED_REVIEW", "CRITICAL_SAFETY"]:
                                try:
                                    log_guardrail_event(
                                        user_id=user["id"] if user else None,
                                        question=question,
                                        meta={
                                            "human_signal_classification": human_classification,
                                            "conversation_risk_review": risk_review,
                                            "source": "support_guidance_agent",
                                            "max_severity": human_classification.get("severity", 0),
                                        },
                                        conversation_id=free_conv_id,
                                        message_id=free_user_msg.get("id"),
                                        decision="ESCALATE_FOR_REVIEW",
                                    )
                                except Exception as e:
                                    st.warning(f"Support escalation log failed: {e}")




                            st.rerun()

                        if review_decision == "ESCALATE_FOR_REVIEW":
                            post_system_message(
                                conv_id,
                                "This conversation has been flagged for human safety review.",
                                meta={"conversation_risk_review": risk_review}
                            )

                            try:
                                log_guardrail_event(
                                    user_id=user["id"],
                                    question=refined_q,
                                    meta={
                                        "conversation_risk_review": risk_review,
                                        "source": "project_tier",
                                        "max_severity": risk_review.get("conversation_chain_risk", 0),
                                    },
                                    conversation_id=conv_id,
                                    message_id=user_msg.get("id"),
                                    decision="ESCALATE_FOR_REVIEW",
                                )
                            except Exception as e:
                                st.warning(f"Escalation log failed: {e}")

                            st.rerun()

                        elif review_decision == "DERAD_ONLY":
                            post_system_message(
                                conv_id,
                                "This conversation has been routed to the safety response pathway.",
                                meta={"conversation_risk_review": risk_review}
                            )

                        # >>> Guardrail gatekeeper (DeRad)
                        pers = load_user_personality(user['id'], user['company_id'])
                        hit, derad_reply, derad_meta = run_derad_guardrail(refined_q, personality=pers)

                        # If the conversation reviewer decided DERAD_ONLY, force the DeRad pathway
                        if review_decision == "DERAD_ONLY":
                            hit = True
                            derad_meta = {
                                **(derad_meta or {}),
                                "guardrail": True,
                                "matched_patterns": derad_meta.get("matched_patterns", []) if derad_meta else [],
                                "matched_trigger_ids": derad_meta.get("matched_trigger_ids", []) if derad_meta else [],
                                "max_severity": max(
                                    int(derad_meta.get("max_severity", 0)) if derad_meta else 0,
                                    int(risk_review.get("conversation_chain_risk", 0) or 0),
                                ),
                                "conversation_risk_review": risk_review,
                            }

                            if not derad_reply:
                                derad_reply = (
                                    "I cannot help with that request. "
                                    "I can, however, offer safe, lawful and constructive guidance."
                                )                      
                        if hit:

                            # Log guardrail event (Sprint 3C)
                            try:
                                log_guardrail_event(
                                    user_id=user["id"],
                                    question=refined_q,
                                    meta=derad_meta,
                                    conversation_id=conv_id,
                                    message_id=user_msg.get("id"),
                                    decision="DERAD_ONLY",
                                )
                            except Exception as e:
                                # Do not break the chat if logging fails
                                st.warning(f"Guardrail log failed: {e}")

                            post_system_message(conv_id, f"[DeRad] {derad_reply}", meta=derad_meta)
                            st.rerun()  # show the new messages immediately

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

                                turn_id = str(uuid.uuid4())
                                st.session_state["last_turn_id"] = turn_id

                                episode_summary = build_episode_summary(refined_q, final_answer)

                                save_episode(
                                    summary=episode_summary,
                                    trigger_type="consensus_end",
                                    tags={
                                        "source": "final_consensus",
                                        "agents": list(role_results.keys()),
                                        "sources": final_sources.split(", ") if final_sources else [],
                                        "project_id": project_id
                                    },
                                    turn_id=turn_id
                                )

                                post_agent_message(conv_id, "Consensus", final_answer, {
                                    "role": "consensus",
                                    "sources": final_sources.split(", ") if final_sources else []
                                })

# --------- 7. Ask Question with Role-Aware Collaboration ---------
if not is_paid:

    # FREE HOME
    if st.session_state.get("is_embedded"):
        ctx = get_learning_context()
        subject = ctx.get("subject") or "Learning"
        topic = ctx.get("topic") or ""

        st.markdown(f"### Ask WiseAGI — {subject}")
        if topic:
            st.caption(f"Topic: {topic}")
    else:
        st.title("AI Platform — Start a conversation")
        st.caption("Ask a question with your public experts. No project needed.")

    # --- Start a clean free-tier conversation ---
    if not st.session_state.get("is_embedded"):
        if st.button("➕ Start new conversation", key="start_new_free_conversation"):
            st.session_state.pop("free_conversation_id", None)
            st.session_state["messages"] = []
            st.session_state["free_turns"] = 0
            st.rerun()

    # --- Multi-turn chat state (MVP) ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []   # list of {"role": "user"/"assistant", "content": "..."}
    if "free_turns" not in st.session_state:
        st.session_state["free_turns"] = 0

    # --- Embedded mode: create/reuse conversation and load previous history ---
    free_conv_id = None

    if st.session_state.get("is_embedded"):
        free_conv_id = get_or_create_free_conversation(user)

        if not st.session_state.get("embedded_history_loaded"):
            previous_messages = list_messages(free_conv_id, limit=50)

            st.session_state["messages"] = []

            for msg in previous_messages:
                content = (msg.get("content") or "").strip()
                if not content:
                    continue

                if msg.get("author_type") == "user":
                    st.session_state["messages"].append({
                        "role": "user",
                        "content": content,
                    })

                elif msg.get("author_type") in ["agent", "system"]:
                    clean_content = content

                    if clean_content.startswith("[Consensus]"):
                        clean_content = clean_content.replace("[Consensus]", "").strip()
                    elif clean_content.startswith("[DeRad]"):
                        clean_content = clean_content.replace("[DeRad]", "").strip()

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": clean_content,
                    })

            st.session_state["embedded_history_loaded"] = True

    # --- Render chat bubbles ---
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # --- Step 6: input box (multi-turn) ---
    prompt = st.chat_input("Ask your question…")

    if prompt:
        # Require agent selection
        if st.session_state.get("is_embedded"):
            embedded_agent_id = st.session_state.get("active_agent_id")
            selected_agent_ids = [embedded_agent_id] if embedded_agent_id else []

        if not selected_agent_ids:
            st.warning("No tutor is currently configured for this subject.")
            st.stop()

        # Free-tier cap (use your FREE_MAX_TURNS_PER_SESSION value)
        if st.session_state["free_turns"] >= FREE_MAX_TURNS_PER_SESSION:
            st.warning("Free tier: you have reached the maximum turns for this session.")
            st.stop()

        # Add user message bubble
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["free_turns"] += 1

        # Create / reuse a database conversation for free chat
        if not free_conv_id:
            free_conv_id = get_or_create_free_conversation(user)

        # Save the free user message to mvp_messages
        free_user_msg = post_user_message(
            conversation_id=free_conv_id,
            user_id=user["id"],
            text=prompt,
            meta={
                "source": "embedded" if st.session_state.get("is_embedded") else "free_tier",
                "free_turn": st.session_state["free_turns"],
                "agents_requested": selected_agent_ids,
                "learning_context": get_learning_context(),
            }
        )

        # Limit memory to last 8 messages
        last8 = st.session_state["messages"][-8:]
        thread_text = "\n".join(
            [("User" if x["role"] == "user" else "Assistant") + ": " + x["content"] for x in last8]
        )

        # Use this as the question for your existing pipeline
        question = prompt
        question_with_context = f"{question}\n\nRecent conversation:\n{thread_text}"

        # >>> Always-on Conversation Risk Reviewer for FREE tier
        try:
            risk_review = review_conversation_risk(
                conversation_id=free_conv_id,
                user_id=user["id"],
                latest_message=question,
                latest_message_id=free_user_msg.get("id"),
                company_id=user.get("company_id"),
            )
        except Exception as e:
            risk_review = {
                "decision": "LOG_ONLY",
                "reason": f"Free risk reviewer failed safely: {e}",
            }
            st.warning(f"Free risk reviewer failed safely: {e}")

        review_decision = (risk_review.get("decision") or "LOG_ONLY").upper()

        safety_note = ""

        if review_decision == "ADD_SAFETY_NOTE":
            safety_note = (
                "\n\nSafety instruction: keep the response lawful, non-harmful, "
                "constructive, calm, and focused on safe guidance."
            )

            question_with_context = question_with_context + safety_note

        # >>> Human Signals Classifier response routing
        try:
            human_classification = classify_human_signal(
                latest_message=question,
                conversation_history=thread_text
            )
        except Exception as e:
            human_classification = {
                "route": "NORMAL",
                "primary_signal": "none",
                "reason": f"Classifier failed safely: {e}",
            }

        human_route = (human_classification.get("route") or "NORMAL").upper()

        if human_route in ["SOFT_SUPPORT", "ELEVATED_REVIEW", "CRITICAL_SAFETY"]:
            support_reply = run_support_guidance_agent(
                question,
                human_classification
            )

            st.session_state["messages"].append({
                "role": "assistant",
                "content": support_reply
            })

            try:
                secure_supabase.table("mvp_messages").insert({
                    "conversation_id": free_conv_id,
                    "author_id": None,
                    "author_type": "agent",
                    "content": support_reply,
                    "meta": {
                        "human_signal_classification": human_classification,
                        "conversation_risk_review": risk_review,
                        "source": "support_guidance_agent"
                    },
                    "created_at": datetime.utcnow().isoformat(),
                }).execute()
            except Exception as e:
                st.warning(f"Could not save Support & Guidance response: {e}")

            try:
                supabase.table("qa_history").insert({
                    "user_id": st.session_state["user"]["id"],
                    "question": question,
                    "answer": support_reply,
                    "context": thread_text,
                    "agent_list": "Support & Guidance Agent",
                    "agent_roles": str({
                        "Support & Guidance Agent": human_classification.get("primary_signal", "")
                    }),
                    "timestamp": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                st.warning(f"Could not save Support & Guidance to previous questions: {e}")

            if human_route in ["ELEVATED_REVIEW", "CRITICAL_SAFETY"]:
                try:
                    log_guardrail_event(
                        user_id=user["id"] if user else None,
                        question=question,
                        meta={
                            "human_signal_classification": human_classification,
                            "conversation_risk_review": risk_review,
                            "source": "support_guidance_agent",
                            "max_severity": human_classification.get("severity", 0),
                        },
                        conversation_id=free_conv_id,
                        message_id=free_user_msg.get("id"),
                        decision="DERAD_ONLY",
                    )
                except Exception as e:
                    st.warning(f"Support escalation log failed: {e}")

            st.rerun()

        if review_decision == "ESCALATE_FOR_REVIEW":
            human_classification = classify_human_signal(
                latest_message=question,
                conversation_history=thread_text
            )

            support_reply = run_support_guidance_agent(
                question,
                human_classification
            )

            st.session_state["messages"].append({
                "role": "assistant",
                "content": support_reply
            })

            secure_supabase.table("mvp_messages").insert({
                "conversation_id": free_conv_id,
                "author_id": None,
                "author_type": "assistant",
                "content": support_reply,
                "meta": {
                    "human_signal_classification": human_classification,
                    "conversation_risk_review": risk_review,
                    "source": "support_guidance_agent"
                },
                "created_at": datetime.utcnow().isoformat(),
            }).execute()

        # ✅ From here onward, keep your existing pipeline, but use question_with_context instead of question 
        # Only public agents are allowed on the free home
        allowed_ids = assert_agents_allowed(supabase, None, selected_agent_ids)
        if not allowed_ids:
            st.warning("Please select at least one public agent in the sidebar.")
        else:
            pers = load_user_personality(user['id'], user['company_id'])

                    # >>> Guardrail gatekeeper (DeRad) — ADD HERE
            hit, derad_reply, derad_meta = run_derad_guardrail(question_with_context, personality=pers)

            # If the risk reviewer decides DERAD_ONLY, force the DeRad route
            if review_decision == "DERAD_ONLY":
                hit = True
                derad_meta = {
                    **(derad_meta or {}),
                    "guardrail": True,
                    "matched_patterns": derad_meta.get("matched_patterns", []) if derad_meta else [],
                    "matched_trigger_ids": derad_meta.get("matched_trigger_ids", []) if derad_meta else [],
                    "max_severity": max(
                        int(derad_meta.get("max_severity", 0)) if derad_meta else 0,
                        int(risk_review.get("conversation_chain_risk", 0) or 0),
                    ),
                    "conversation_risk_review": risk_review,
                    "source": "free_tier",
                }

                if not derad_reply:
                    derad_reply = (
                        "I cannot help with that request. "
                        "I can, however, offer safe, lawful and constructive guidance."
                    )
            if hit:
                try:
                    log_guardrail_event(
                        user_id=user["id"] if user else None,
                        question=question,
                        meta=derad_meta,
                        conversation_id=free_conv_id,
                        message_id=free_user_msg.get("id"),
                        decision="DERAD_ONLY",
                    )
                except Exception as e:
                    st.warning(f"Guardrail log failed: {e}")

                st.info("Safety guidance applied.")
                st.session_state["messages"].append({"role": "assistant", "content": derad_reply})
                
                try:
                    post_system_message(
                        free_conv_id,
                        f"[DeRad] {derad_reply}",
                        meta=derad_meta
                    )
                except Exception as e:
                    st.warning(f"Could not save free DeRad response: {e}")
                
                st.rerun()

            active_agents = [a for a in get_all_agents() if a['id'] in allowed_ids]

            role_order = ["default", "devil_advocate", "pragmatic", "supportive"]
            agent_scores, results_by_agent = [], {}

            # --- gather hits and score agents
            agent_scores = []
            results_by_agent = {}
            for agent in active_agents:
                hits = fetch_hits_multi(
                    agent_id=agent['id'],
                    query_text=question_with_context,
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
            default_prompt = create_role_based_prompt("default", question_with_context, default_agent["description"], default_sources)

            default_summary, d_tokens, d_cost = query_agent_context(
                question_with_context, default_prompt, personality=pers)
            
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
                    f"Question: {question_with_context}\nSummary: {default_summary}",
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

            st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        
            total_tokens += c_tokens or 0
            total_cost += c_cost or 0.0
            step_stats.append({"step": "Consensus", "tokens": c_tokens or 0, "cost": c_cost or 0.0})

            turn_id = str(uuid.uuid4())
            st.session_state["last_turn_id"] = turn_id

            episode_summary = build_episode_summary(question, final_answer)

            save_episode(
                summary=episode_summary,
                trigger_type="consensus_end",
                tags={
                    "source": "final_consensus",
                    "agents": list(role_results.keys()),
                    "sources": final_sources.split(", ") if final_sources else []
                },
                turn_id=turn_id
            )

            # --- PRESENTATION
            # Optional: keep sources + run details in an expander (still visible, but not in the bubble)
            with st.expander("Run details / sources"):
                if final_sources:
                    st.markdown("**Sources:** " + " • ".join(s for s in final_sources.split(", ") if s))
                    
            #-----------Telemetry summary ----------------------------
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

            # Refresh UI so the new bubble appears immediately
            st.rerun()

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






















