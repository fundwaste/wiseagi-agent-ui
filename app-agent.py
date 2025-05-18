# app.py - WiseAGI Full Rewrite ✅

import streamlit as st
from supabase_config import supabase
from pymilvus import connections, Collection
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
import requests

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
            {"role": "system", "content": "Summarise based only on the provided context. Include references to any sources mentioned."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()

def get_all_agents():
    try:
        response = supabase.table("agents").select("id, agent_name, description, collection_name").execute()
        agents = response.data or []
        return agents
    except Exception as e:
        st.error(f"⚠️ Error fetching agents: {e}")
        return []

def upload_document_to_agent(file, agent_id):
    file_name = f"{agent_id}/{file.name}"
    supabase.storage.from_("agent_documents").upload(file_name, file)
    supabase.table("agent_documents").insert({
        "agent_id": agent_id,
        "file_name": file.name,
        "uploaded_at": datetime.utcnow().isoformat()
    }).execute()

def save_user_agent(user_id, agent_id):
    supabase.table("user_agents").insert({
        "user_id": user_id,
        "agent_id": agent_id
    }).execute()

# --------- 3. Authentication ---------
def login_page():
    st.title("🔐 Login to WiseAGI")
    choice = st.radio("Login or Sign Up", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.button("Create Account"):
            response = supabase.auth.sign_up({"email": email, "password": password})
            if response.user:
                st.success("✅ Account created! Please verify email.")
            else:
                st.error("❌ Signup failed.")
    else:
        if st.button("Login"):
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.session:
                st.session_state["user"] = {"id": response.user.id, "email": response.user.email}
                st.rerun()
            else:
                st.error("❌ Login failed.")

if "user" not in st.session_state:
    login_page()
    st.stop()

# --------- 4. Sidebar: Agent Selection ---------
st.sidebar.image("RABIIT.jpg", use_container_width=True)
st.sidebar.title("👤 Welcome")
st.sidebar.markdown(f"**Email:** {st.session_state['user']['email']}")

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

st.sidebar.header("Step 1: Choose or Add Agents")
existing_agents = get_all_agents()

# Search Bar
search_term = st.sidebar.text_input("Type a description of skill and press Enter:")
if search_term:
    filtered_agents = [a for a in existing_agents if search_term.lower() in a['description'].lower()]
else:
    filtered_agents = existing_agents

# Multi-select
agent_options = {
    f"{a['agent_name']} - {a['description']}": a['id']
    for a in filtered_agents
}

selected_agent_labels = st.sidebar.multiselect(
    "Select up to 5 agents",
    options=list(agent_options.keys()),
    default=[],
    max_selections=5
)

selected_agent_ids = [agent_options[label] for label in selected_agent_labels]

# Add New Agent
st.sidebar.markdown("---")
st.sidebar.subheader("Create a New Agent")

if st.sidebar.button("➕ Create New Agent"):
    with st.sidebar.form("new_agent_form", clear_on_submit=True):
        new_name = st.text_input("Agent Name")
        new_desc = st.text_area("Agent Description")
        uploaded_files = st.file_uploader("Upload supporting documents", accept_multiple_files=True)
        submitted = st.form_submit_button("Save Agent")
        if submitted and new_name and new_desc:
            create_resp = supabase.table("agents").insert({
                "agent_name": new_name,
                "description": new_desc,
                "user_id": st.session_state['user']['id'],
                "collection_name": new_name.replace(" ", "_").lower()
            }).execute()
            if create_resp.data:
                new_agent_id = create_resp.data[0]['id']
                for f in uploaded_files:
                    upload_document_to_agent(f, new_agent_id)
                save_user_agent(st.session_state['user']['id'], new_agent_id)
                st.success("🎉 New agent created!")
                st.rerun()
            else:
                st.error("⚠️ Could not create agent.")

# --------- 5. Main Area: Ask Questions ---------
st.title("WiseAGI - Your Personal Collaborative Experts")
st.markdown("---")

question = st.text_input("Ask your question:")
agent_summaries = []

if selected_agent_ids and st.button("Submit Question"):
    for agent_id in selected_agent_ids:
        agent = next((a for a in existing_agents if a['id'] == agent_id), None)
        collection_name = agent.get("collection_name")

        if not collection_name:
            st.warning(f"⚠️ {agent['agent_name']} has no collection_name set.")
            continue

        try:
            collection = Collection(name=collection_name)
            results = collection.search(
                data=[generate_embedding(question)],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 5}},
                limit=3,
                output_fields=["Text", "Source"]
            )
            context_chunks = [f"{r.entity.get('Text')} (Source: {r.entity.get('Source')})" for r in results[0]]

            if context_chunks:
                agent_summary = query_openai_context(
                    f"Summarise what this agent found in response to the question: {question}",
                    "\n".join(context_chunks),
                    model="gpt-3.5-turbo"
                )
                agent_summaries.append(f"{agent['agent_name']}:\n{agent_summary}")

        except Exception as e:
            st.warning(f"Could not query {agent['agent_name']}: {str(e)}")

    if agent_summaries:
        final_context = "\n\n".join(agent_summaries)
        final_response = query_openai_context(
            f"Based on these summaries, provide a clear and complete answer to: {question}. Include references.",
            final_context,
            model="gpt-4o"
        )
        st.success("🧠 Final AI Response:")
        st.write(final_response)
    else:
        st.info("No useful context was found by the agents.")
else:
    st.info("👈 Please select or create at least one agent to begin.")








