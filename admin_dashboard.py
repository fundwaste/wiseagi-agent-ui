import streamlit as st
from supabase_config import supabase
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("📈 Admin Dashboard")

# Ensure only the admin can access
if st.session_state.get("user", {}).get("email") != "rashid@wahab.eu":
    st.error("🚫 You are not authorised to view this page.")
    st.stop()

# Load data from Supabase
qa_data = supabase.table("qa_history").select("*").execute()
if not qa_data.data:
    st.warning("No QA history found.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(qa_data.data)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["num_agents"] = df["agent_list"].apply(lambda x: len(x.split(",")) if x else 0)
df["token_estimate"] = df["context"].apply(lambda x: int(len(x.split()) * 1.3))  # Rough token estimate

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Questions", len(df))
col2.metric("Avg. Agents per Query", f"{df['num_agents'].mean():.2f}")
col3.metric("Est. Tokens Used", df["token_estimate"].sum())

# Charts
st.subheader("🧑‍🤝‍🧑 Agent Collaboration")
agents_split = df["agent_list"].dropna().str.split(",").explode()
agent_counts = agents_split.value_counts()
st.bar_chart(agent_counts)

st.subheader("📅 Questions Over Time")
df_by_day = df.groupby(df["timestamp"].dt.date).size()
st.line_chart(df_by_day)

st.subheader("📉 Token Usage Distribution")
st.area_chart(df["token_estimate"])

st.caption("Token estimates are based on context word count * 1.3 multiplier.")
