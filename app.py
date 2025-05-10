# app.py - WiseAGI Complete Rewrite with Cross-Agent Reasoning ✅

import streamlit as st
from supabase_config import supabase
from pymilvus import connections, Collection
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

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

def query_openai_context(prompt, context, model="gpt-4o"):
    response = openai_client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": "Summarise based only on the provided context. Include references and contributing agents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()

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
                user_email = res.user.email.strip().lower()  # 🔍 Normalise case and spaces
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

# --------- 4. Sidebar: Profile and Agents ---------
st.sidebar.image("RABIIT.jpg", use_container_width=True)
st.sidebar.title("👤 Welcome")
st.sidebar.markdown(f"**Email:** {st.session_state['user']['email']}")

# DEBUG: Show admin flag in sidebar
st.sidebar.markdown(f"**Admin:** {st.session_state['user'].get('is_admin')}")  # DEBUG

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# 🛡️ Safely check if user is admin
if st.session_state.get("user", {}).get("is_admin", False):
    if st.sidebar.checkbox("🔐 Admin Mode"):
        admin_page()
        st.stop()  # Prevents user-facing logic from executing

# Agent selection
st.sidebar.markdown("---")
st.sidebar.header("Your Current Agents")
user_agents = get_user_agents(st.session_state["user"]["id"])
selected_agent_ids = [
    a["id"]
    for a in user_agents
    if st.sidebar.checkbox(f"{a['agent_name']} - {a['description']}", key=a["id"])
]

# --------- 5. Explore and Add Agents ---------
st.sidebar.markdown("---")
st.sidebar.header("Step 1: Choose or Add Agents")
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

# --------- 6. Create New Agent ---------
st.sidebar.markdown("---")
st.sidebar.subheader("Create a New Agent")
if st.sidebar.button("➕ Create New Agent"):
    with st.sidebar.form("new_agent_form", clear_on_submit=True):
        new_name = st.text_input("Agent Name")
        new_desc = st.text_area("Agent Description")
        uploaded_files = st.file_uploader("Upload supporting documents", accept_multiple_files=True)
        submitted = st.form_submit_button("Save Agent")

        if submitted and new_name and new_desc and uploaded_files:
            res = supabase.table("agents").insert({
                "agent_name": new_name,
                "description": new_desc,
                "user_id": st.session_state['user']['id'],
                "collection_name": new_name.replace(" ", "_").lower(),
                "is_public": True
            }).execute()

            if res.data:
                agent_id = res.data[0]['id']
                for f in uploaded_files:
                    upload_document_to_agent(f, agent_id)
                save_user_agent(st.session_state['user']['id'], agent_id)
                st.success("🎉 New agent created and added to your portfolio!")
                st.rerun()
            else:
                st.error("⚠️ Could not create agent.")

# --------- 7. Ask Question with Cross-Agent Reasoning ---------
st.title("WiseAGI - Your Personal Collaborative Experts")
st.markdown("---")
question = st.text_input("Ask your question:")

if selected_agent_ids and st.button("Submit Question"):
    first_pass = {}

    for aid in selected_agent_ids:
        agent = next(a for a in user_agents if a['id'] == aid)
        try:
            col = Collection(name=agent['collection_name'])
            results = col.search(data=[generate_embedding(question)], anns_field="vector", param={"metric_type": "COSINE", "params": {"nprobe": 5}}, limit=3, output_fields=["Text", "Source"])
            chunks = [f"{r.entity.get('Text')} ({r.entity.get('Source')})" for r in results[0]]
            if chunks:
                initial = query_openai_context(f"Initial summary for: {question}", "\n".join(chunks), model="gpt-3.5-turbo")
                first_pass[agent['agent_name']] = initial
        except Exception as e:
            st.warning(f"{agent['agent_name']} query error: {e}")

    refined_summaries = []
    for name, own_summary in first_pass.items():
        peer_context = "\n".join([f"[{peer}] {summary}" for peer, summary in first_pass.items() if peer != name])
        context = f"Your Summary:\n{name}: {own_summary}\n\nPeer Summaries:\n{peer_context}"
        refined = query_openai_context(f"Refine your answer to: {question}", context, model="gpt-3.5-turbo")
        st.markdown(f"**Refined Summary from [{name}]:**")
        st.write(refined)
        refined_summaries.append(f"[{name}] {refined}")

    final_context = "\n\n".join(refined_summaries)
    final_answer = query_openai_context(f"Based on refined summaries, answer: {question}", final_context, model="gpt-4o")
    st.success("🧠 Final AI Response:")
    st.write(final_answer)

    supabase.table("qa_history").insert({
        "user_id": st.session_state['user']['id'],
        "question": question,
        "answer": final_answer,
        "context": final_context,
        "agent_list": ",".join(first_pass.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }).execute()

# --------- 8. History ---------
st.markdown("---")
st.subheader("🔁 Load a Previous Question")
history = supabase.table("qa_history").select("id, question, answer, timestamp, agent_list").order("timestamp", desc=True).limit(20).execute()
if history.data:
    for entry in history.data:
        label = f"{entry['question']} ({entry['agent_list']})"
        if st.checkbox(label, key=entry['id']):
            st.session_state['loaded_question'] = entry['question']
            st.session_state['loaded_answer'] = entry['answer']
            break

if 'loaded_question' in st.session_state:
    st.markdown("---")
    st.subheader("🎧 Reloaded Question")
    st.write(st.session_state['loaded_question'])
    st.subheader("🔮 Previous Answer")
    st.write(st.session_state['loaded_answer'])





