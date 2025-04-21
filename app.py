import streamlit as st
from supabase_config import supabase
from pymilvus import connections, Collection
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
import textwrap

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
            {
                "role": "system",
                "content": (
                    "You are a collaborative panel of expert professors. "
                    "For each answer you provide, clearly reference the source document name in brackets like (source.pdf) "
                    "and the contributing expert in square brackets like [Islam Specialist]. "
                    "Only use the context provided. Do not make up or guess information beyond it."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {prompt}"
            }
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

        st.write(f"\n🔍 Agent: {agent} | Collection: {collection_name}")
        for r in results[0]:
            text = r.entity.get("Text")
            source = r.entity.get("Source")
            if text and source:
                combined_context.append(text)
                source_info.append(f"[{agent}] ({source})")
                if agent not in contributing_agents:
                    contributing_agents.append(agent)
                st.write(f"- Source: {source}, Text snippet: {text[:80]}...")
            else:
                st.warning(f"⚠️ Missing text or source for {agent}: {r}")

    context = "\n".join([f"{c} {s}" for c, s in zip(combined_context, source_info)])
    if not context:
        return "No useful context was found in the database.", "", []

    answer = query_openai_context(user_query, context)
    return answer, context, contributing_agents

def classify_specialist(question):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a classification model. Your task is to determine which specialist "
                    "is best suited to answer the following question. The options are: "
                    "'Islam Specialist', 'Rwanda Specialist', 'Marketing Specialist', and 'Operations Specialist'. "
                    "Return only one of these as your answer."
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

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
                    st.success("✅ Account created! Please check your email to confirm.")
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
                    st.error("❌ Login failed. Please check your email or password.")
            except Exception as e:
                st.error(f"❌ Login error: {str(e)}")

if "user" not in st.session_state:
    login_page()
    st.stop()

st.set_page_config(page_title="WiseAGI", layout="wide")

with st.sidebar:
    st.image("RABIIT.jpg", use_container_width=True)  # ✅ Logo at top of sidebar

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

    st.markdown("---")
    st.markdown("### 🕓 Previous Questions")

    # Fetch all history for this user
    history = supabase.table("qa_history").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute()

    if history.data:
        specialists = ["Islam Specialist", "Rwanda Specialist", "Marketing Specialist", "Operations Specialist"]
        
        for specialist in specialists:
            st.markdown(f"#### 📂 {specialist}")
            filtered = [q for q in history.data if q.get("assigned_specialist") == specialist]
            
            if not filtered:
                st.caption("No questions yet.")
                continue
            
            for q in filtered:
                label = f"{q['question']} ({q['timestamp'][:10]})"
                if st.checkbox(label, key=f"hist_{q['id']}"):
                    st.session_state["loaded_question"] = q["question"]
                    st.session_state["loaded_answer"] = q["answer"]
                    break
    else:
        st.caption("No past questions found.")


st.title("Welcome to WiseAGI Agents")
st.markdown("""
Explore specialist contributors with deep knowledge on topics such as Islam, Rwanda, Marketing, and more.

Choose your contributors, ask your question, and receive expert-generated responses using advanced AI.
""")

st.markdown("---")
st.subheader("Ask a Question")
question = st.text_input("What would you like to ask?", value=st.session_state.get("loaded_question", ""))

if st.button("Submit Question"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state["selected_agents"]:
        st.warning("Please select at least one contributor from the sidebar.")
    else:
        answer, context, agents_used = query_collection_openai(question, st.session_state["selected_agents"])
        st.markdown("### 🧠 AI Response:")
        st.write(answer)

        with st.expander("🧠 Debug: Context sent to OpenAI"):
            st.code(context, language="markdown")

        specialist_prediction = classify_specialist(question)

        supabase.table("qa_history").insert({
            "user_id": st.session_state["user"]["id"],
            "question": question,
            "answer": answer,
            "context": context,
            "agent_list": ",".join(agents_used),
            "assigned_specialist": specialist_prediction,
            "timestamp": datetime.utcnow().isoformat()   
        }).execute()

        st.session_state.pop("loaded_question", None)
        st.session_state.pop("loaded_answer", None)

if "loaded_answer" in st.session_state:
    st.markdown("### 🧠 AI Response:")
    st.write(st.session_state["loaded_answer"])








