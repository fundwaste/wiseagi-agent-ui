# app.py - WiseAGI Complete Rewrite with Cross-Agent Reasoning ✅

import streamlit as st
from supabase_config import supabase
from pymilvus import connections, Collection
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
import numpy as np
import tiktoken

# --------- 1. Setup Connections ---------
connections.connect(
    alias="default",
    uri="https://in03-357b70cf3851670.serverless.gcp-us-west1.cloud.zilliz.com",
    token="ce5060c7939d564fa7ae65d5c85cad6462b6b6fe5b0a8afc6216c7e3bd80da0aeb3ed4688157c2af9a36ddd30bc5838f9f53d880"
)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
openai_client = OpenAI()

# --------- 2. Helper Functions ---------
def generate_embedding(text):
    return model.encode([text])[0].tolist()

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

def query_openai_context(prompt, context, purpose="default"):
    model = model_router(purpose)
    full_input = f"{prompt}\n\n{context}"

    prompt_tokens = count_tokens(full_input, model)

    response = openai_client.chat.completions.create(
        model=model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are a helpful and knowledgeable assistant."},
            {"role": "user", "content": full_input}
        ]
    )

    answer = response.choices[0].message.content.strip()
    completion_tokens = count_tokens(answer, model)
    total_tokens = prompt_tokens + completion_tokens
    cost = estimate_cost(prompt_tokens, completion_tokens, model)

    return answer, total_tokens, cost

def deduplicate_agents(agent_list):
    seen = set()
    unique = []
    for agent in agent_list:
        if agent['id'] not in seen:
            unique.append(agent)
            seen.add(agent['id'])
    return unique

def get_all_agents():
    return supabase.table("agents").select("id, agent_name, description, collection_name").execute().data or []

def get_user_agents(user_id):
    join = supabase.table("user_agents").select("agent_id, agents(id, agent_name, description, collection_name)").eq("user_id", user_id).execute()
    return deduplicate_agents([r["agents"] for r in join.data if r.get("agents")])

def save_user_agent(user_id, agent_id):
    supabase.table("user_agents").insert({"user_id": user_id, "agent_id": agent_id}).execute()

def upload_document_to_agent(file, agent_id):
    fname = f"{agent_id}/{file.name}"
    supabase.storage.from_("agent_documents").upload(fname, file)
    supabase.table("agent_documents").insert({"agent_id": agent_id, "file_name": file.name, "uploaded_at": datetime.utcnow().isoformat()}).execute()

# --------- 3. Authentication ---------
def login_page():
    st.title("🔐 Login to WiseAGI")
    auth = st.radio("Login or Sign Up", ["Login", "Sign Up"])
    email = st.text_input("Email")
    pwd = st.text_input("Password", type="password")

    if auth == "Sign Up":
        if st.button("Create Account"):
            res = supabase.auth.sign_up({"email": email, "password": pwd})
            if res.user:
                st.success("✅ Account created! Please verify your email.")
            else:
                st.error("❌ Signup failed.")
    else:
        if st.button("Login"):
            res = supabase.auth.sign_in_with_password({"email": email, "password": pwd})
            if res.session:
                user_id = res.user.id
                user_email = res.user.email.strip().lower()
                is_admin = user_email == "rashid@wahab.eu"
                st.session_state["user"] = {
                    "id": user_id,
                    "email": user_email,
                    "is_admin": is_admin
                }
                st.rerun()
            else:
                st.error("❌ Login failed.")

if "user" not in st.session_state:
    login_page()
    st.stop()

# --------- 4. Sidebar UI ---------
st.sidebar.image("RABIIT.jpg", use_container_width=True)
st.sidebar.title("👤 Welcome")
st.sidebar.markdown(f"**Email:** {st.session_state['user']['email']}")
st.sidebar.markdown(f"**Admin:** {st.session_state['user'].get('is_admin')}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

def admin_page():
    st.title("🛠️ Admin Panel")
    st.markdown("Welcome, Admin. You can manage collections and agents from this section.")

if st.session_state.get("user", {}).get("is_admin", False):
    if st.sidebar.checkbox("🔐 Admin Mode"):
        admin_page()
        st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Your Current Agents")
user_agents = get_user_agents(st.session_state["user"]["id"])
selected_agent_ids = [a["id"] for a in user_agents if st.sidebar.checkbox(f"{a['agent_name']} - {a['description']}", key=a["id"])]

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


# --------- 7. Ask Question with Role-Aware Collaboration ---------
st.title("WiseAGI - Your Personal Collaborative Experts")
st.markdown("---")
question = st.text_input("Ask your question:")

def create_role_based_prompt(agent_role, default_summary, agent_speciality, sources):
    citation = f"\nSources: {sources}" if sources else ""
    if agent_role == "default":
        return f"{agent_speciality}: Summarise this question with supporting details:\n{default_summary}{citation}"
    elif agent_role == "devil_advocate":
        return (
            f"{agent_speciality} (Devil's Advocate): Critically review the summary, identify gaps, question assumptions:\n{default_summary}{citation}"
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

if selected_agent_ids and st.button("Submit Question"):
    question_vector = generate_embedding(question)

    role_order = ["default", "devil_advocate", "pragmatic", "supportive"]
    agent_scores = []
    results_by_agent = {}
    col = Collection("wise_studies")
    
    total_tokens = 0
    total_cost = 0.0
    
    for agent in user_agents:
        if agent['id'] in selected_agent_ids:
            results = col.search(
                data=[question_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 5}},
                limit=3,
                output_fields=["Text", "agent_id", "document_title", "Source"]
            )
            filtered = [r for r in results[0] if r.entity.get("agent_id") == agent['id']]
            score = filtered[0].distance if filtered else 0.0
            agent_scores.append((agent, score))
            results_by_agent[agent['id']] = filtered

    agent_scores.sort(key=lambda x: x[1], reverse=True)

    assigned_roles = {}
    role_results = {}
    cited_docs = {}

    default_agent = agent_scores[0][0]
    default_hits = results_by_agent.get(default_agent['id'], [])
    default_titles = set(hit.get("document_title") or "" for hit in default_hits)
    default_sources = ", ".join(sorted(default_titles))
    default_prompt = create_role_based_prompt("default", question, default_agent["description"], default_sources)
    default_summary, tokens_used, cost = query_openai_context(question, default_prompt)
    total_tokens += tokens_used
    total_cost += cost

    role_results[default_agent["agent_name"]] = default_summary
    assigned_roles[default_agent["agent_name"]] = "default"
    cited_docs[default_agent["agent_name"]] = default_sources

    st.markdown(f"**[{default_agent['agent_name']}] Role: Default**")
    st.markdown(f"*Sources: {default_sources}*")
    st.write(default_summary)
    st.markdown(f"🧮 Tokens used: {tokens_used}")
    st.markdown(f"💰 Estimated cost: ${cost:.4f}")

    for i, (agent, _) in enumerate(agent_scores[1:], start=1):
        role = role_order[i] if i < len(role_order) else "supportive"
        hits = results_by_agent.get(agent['id'], [])
        sources = ", ".join(sorted(set(hit.entity.get("document_title", "") for hit in hits)))
        prompt = create_role_based_prompt(role, default_summary, agent["description"], sources)
        agent_summary, tokens_used, cost = query_openai_context(f"Agent [{agent['agent_name']}] role: {role}", prompt, purpose="peer_review")
        total_tokens += tokens_used
        total_cost += cost

        role_results[agent["agent_name"]] = agent_summary
        assigned_roles[agent["agent_name"]] = role
        cited_docs[agent["agent_name"]] = sources

        st.markdown(f"**[{agent['agent_name']}] Role: {role.replace('_', ' ').title()}**")
        st.markdown(f"*Sources: {sources}*")
        st.write(agent_summary)
        st.markdown(f"🧮 Tokens used: {tokens_used}")
        st.markdown(f"💰 Estimated cost: ${cost:.4f}")

    combined_context = "\n\n".join([f"[{name}] {summary}" for name, summary in role_results.items()])
    final_sources = ", ".join(sorted(set(sum([v.split(", ") for v in cited_docs.values()], []))))
    final_prompt = f"Final consensus answer based on all perspectives. Cite from: {final_sources}"
    final_answer, tokens_used, cost = query_openai_context(final_prompt, combined_context, purpose="consensus")
    total_tokens += tokens_used
    total_cost += cost

    st.success("🧠 Final AI Response:")
    st.write(final_answer)
    st.markdown(f"🧮 Tokens used for final AI response: {tokens_used}")
    st.markdown(f"💰 Estimated cost: ${cost:.4f}")

    st.markdown("### 🧾 Summary")
    st.markdown(f"📝 To answer this question with all the agents you used {total_tokens} tokens and the total cost was ${total_cost:.4f}.")

    supabase.table("qa_history").insert({
        "user_id": st.session_state['user']['id'],
        "question": question,
        "answer": final_answer,
        "context": combined_context,
        "agent_list": ",".join(role_results.keys()),
        "agent_roles": str(assigned_roles),
        "timestamp": datetime.utcnow().isoformat()
    }).execute()

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





