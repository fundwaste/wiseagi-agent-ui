import streamlit as st
from supabase_config import supabase
from pymilvus import connections, Collection
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os

# Connect to Zilliz
connections.connect(
    alias="default",
    uri="https://in03-357b70cf3851670.serverless.gcp-us-west1.cloud.zilliz.com",
    token="ce5060c7939d564fa7ae65d5c85cad6462b6b6fe5b0a8afc6216c7e3bd80da0aeb3ed4688157c2af9a36ddd30bc5838f9f53d880"
)

# Load embedding model once
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_embedding(text):
    return model.encode([text])[0].tolist()

def query_openai_context(prompt, context):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=500,
        messages=[
            {"role": "system", "content": (
                "You are a collaborative panel of expert professors. "
                "Reference the source document in brackets like (source.pdf) "
                "and expert in square brackets like [Islam Specialist]. "
                "Only use the context provided."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()

def query_collection_openai(user_query, agent_collection_map, top_k=5):
    query_vector = generate_embedding(user_query)
    combined_context = []
    source_info = []
    contributing_agents = []

    for agent, collection_name in agent_collection_map.items():
        collection = Collection(name=collection_name)
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 5}},
            limit=top_k,
            output_fields=["Text", "Source"]
        )

        for r in results[0]:
            text = r.entity.get("Text")
            source = r.entity.get("Source")
            if text and source:
                combined_context.append(text)
                source_info.append(f"[{agent}] ({source})")
                if agent not in contributing_agents:
                    contributing_agents.append(agent)

    context = "\n".join([f"{c} {s}" for c, s in zip(combined_context, source_info)])
    if not context:
        return "No useful context found in the database.", "", []

    answer = query_openai_context(user_query, context)
    return answer, context, contributing_agents

def alternative_llm_response(prompt):
    # Stubbed out alternative LLM response for now
    return "This is a simulated alternative LLM response."

def display_star_rating(key):
    rating = st.slider("Rate this answer", 1, 5, 3, key=key)
    st.write(f"⭐ You rated: {rating} / 5")
    return rating

def login_page():
    st.set_page_config(page_title="Login to WiseAGI", layout="wide")
    st.title("🔐 Login to WiseAGI")

    choice = st.radio("Login or Sign Up", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.button("Create Account"):
            try:
                response = supabase.auth.sign_up({"email": email, "password": password})
                if response.user:
                    supabase.table("users").insert({"id": response.user.id, "email": email, "plan": "free"}).execute()
                    st.success("✅ Account created! Check your email to confirm.")
                else:
                    st.error("⚠️ Signup failed — no user returned from Supabase.")
            except Exception as e:
                st.error(f"❌ Signup error: {str(e)}")
    else:
        if st.button("Login"):
            try:
                response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if response.session:
                    st.session_state["user"] = {"id": response.user.id, "email": response.user.email}
                    st.success("✅ Logged in successfully.")
                    st.rerun()
                else:
                    st.error("❌ Login failed. Check your credentials.")
            except Exception as e:
                st.error(f"❌ Login error: {str(e)}")

if "user" not in st.session_state:
    login_page()
    st.stop()

st.set_page_config(page_title="WiseAGI", layout="wide")

with st.sidebar:
    st.image("RABIIT.jpg", use_container_width=True)
    st.title("👤 Account")
    user_email = st.session_state["user"]["email"]
    user_id = st.session_state["user"]["id"]

    try:
        user_data = supabase.table("users").select("plan").eq("id", user_id).single().execute()
        plan = user_data.data["plan"]
    except:
        plan = "unknown"

    st.markdown(f"**Email:** {user_email}")
    st.markdown(f"**Plan:** {plan.title() if plan else 'Unknown'}")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Select Contributors")
    contributors = {
        "Islam Specialist": "wise_studies",
        "Rwanda Specialist": "rwanda_studies",
        "Marketing Specialist": "marketing_studies",
        "Operations Specialist": "operations_studies"
    }

    selected_agents = {
        name: collection for name, collection in contributors.items() if st.checkbox(name, key=name)
    }
    st.session_state["selected_agents"] = selected_agents

st.title("Welcome to WiseAGI Agents")
st.markdown("Ask your question below and compare the WiseAGI panel response with an alternative LLM.")

st.markdown("---")
st.subheader("Ask a Question")
question = st.text_input("What would you like to ask?", value="")

if st.button("Submit Question"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state["selected_agents"]:
        st.warning("Please select at least one contributor from the sidebar.")
    else:
        answer, context, agents_used = query_collection_openai(question, st.session_state["selected_agents"])
        alt_answer = alternative_llm_response(question)

        st.markdown("## 🔍 Compare Responses")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧠 WiseAGI Response")
            st.write(answer)
            display_star_rating("wiseagi_rating")

        with col2:
            st.markdown("### 🤖 Alternative LLM")
            st.write(alt_answer)
            display_star_rating("alt_rating")

        supabase.table("qa_history").insert({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "context": context,
            "agent_list": ",".join(agents_used),
            "timestamp": datetime.utcnow().isoformat()
        }).execute()









