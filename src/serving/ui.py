"""Streamlit frontend: database selector → question input → SQL output → execute results."""
import requests
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(page_title="NL2SQL Assistant", page_icon="🗄️", layout="wide")
st.title("NL2SQL Assistant")
st.caption("Powered by fine-tuned Phi-3 mini + RAG schema retrieval")

# Sidebar — database selector
with st.sidebar:
    st.header("Configuration")
    try:
        resp = requests.get(f"{API_URL}/databases", timeout=5)
        databases = resp.json().get("databases", [])
    except Exception:
        databases = []
        st.warning("API not reachable. Start the FastAPI server first.")

    selected_db = st.selectbox("Database", options=["(auto-detect)"] + databases)
    db_id = selected_db if selected_db != "(auto-detect)" else None
    execute_query = st.checkbox("Execute SQL and show results", value=True)

    st.markdown("---")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API: {health['status']} | Ollama: {'✓' if health['ollama'] else '✗'}")
    except Exception:
        st.error("API offline")

# Main — question input
question = st.text_area(
    "Natural language question",
    placeholder="How many singers performed in more than 2 concerts?",
    height=100,
)

if st.button("Generate SQL", type="primary", disabled=not question.strip()):
    with st.spinner("Generating SQL..."):
        try:
            payload = {"question": question, "db_id": db_id, "execute": execute_query}
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()

            st.subheader("Generated SQL")
            st.code(result["sql"], language="sql")

            if result.get("error"):
                st.error(f"Execution error: {result['error']}")
            elif result.get("results"):
                st.subheader("Query Results")
                cols = result["results"]["columns"]
                rows = result["results"]["rows"]
                if rows:
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=cols)
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(rows)} rows returned")
                else:
                    st.info("Query returned no rows.")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: `uvicorn src.serving.api:app --port 8000`")
        except Exception as e:
            st.error(f"Error: {e}")

# Example queries
with st.expander("Example questions"):
    examples = [
        "How many singers are there?",
        "What are the names of all concert venues?",
        "List the top 5 highest paid employees.",
        "How many students are enrolled in each course?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["example"] = ex
