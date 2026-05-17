
# app.py

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="IPL Match Result AI Agent",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 IPL Match Result AI Agent")
st.write("Ask anything about IPL matches using LangGraph + Groq + DuckDuckGo Search")

# =========================
# API KEY
# =========================
groq_api_key = st.text_input(
    "Enter GROQ API Key",
    type="password"
)

# =========================
# USER INPUT
# =========================
query = st.text_input(
    "Enter your question",
    value="give me latest GT vs KKR ipl match result"
)

# =========================
# RUN BUTTON
# =========================
if st.button("Get Result"):

    if not groq_api_key:
        st.warning("Please enter your GROQ API key")
        st.stop()

    try:
        # Set environment variable
        os.environ["GROQ_API_KEY"] = groq_api_key

        # Initialize model
        model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

        # Search Tool
        search_tool = DuckDuckGoSearchRun()

        # Create ReAct Agent
        agent = create_react_agent(
            model=model,
            tools=[search_tool],
            prompt="You are a helpful AI assistant."
        )

        with st.spinner("Searching latest information..."):

            # Invoke agent
            response = agent.invoke(
                {"messages": [("user", query)]}
            )

            # Final answer
            final_answer = response["messages"][-1].content

        st.success("Result Generated!")

        st.subheader("AI Response")
        st.write(final_answer)

    except Exception as e:
        st.error(f"Error: {e}")
