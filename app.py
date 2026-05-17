# app.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Ask from Tathagata",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Ask from Tathagata Nath")

# =========================
# LOAD API KEY
# =========================
groq_api_key = st.secrets["GROQ_API_KEY"]

# =========================
# MODEL
# =========================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# =========================
# SEARCH TOOL
# =========================
search = DuckDuckGoSearchRun()

# =========================
# USER INPUT
# =========================
query = st.text_input(
    "Ask a question",
    value="give me latest GT vs KKR ipl match result"
)

# =========================
# BUTTON
# =========================
if st.button("Search"):

    try:
        with st.spinner("Searching latest information..."):

            # Step 1: Search Web
            search_result = search.invoke(query)

            # Step 2: Send to LLM
            prompt = f"""
            User Question:
            {query}

            Web Search Result:
            {search_result}

            Give a clean and helpful answer.
            """

            response = llm.invoke(prompt)

        st.subheader("Result")
        st.write(response.content)

    except Exception as e:
        st.error(f"Error: {e}")
