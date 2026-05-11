"""Streamlit frontend with mode toggle: Local (phi3-nl2sql via ngrok) or Groq (cloud)."""
import requests
import sqlparse
import streamlit as st

st.set_page_config(page_title="NL2SQL Assistant", page_icon="🗄️", layout="wide")
st.title("NL2SQL Assistant")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    mode = st.radio("Backend", ["Local (phi3-nl2sql)", "Groq (cloud)"], horizontal=True)
    st.markdown("---")

    if mode == "Local (phi3-nl2sql)":
        saved_url = st.session_state.get("local_api_url", "http://localhost:8000")
        local_url = st.text_input("API URL (local or ngrok)", value=saved_url)
        st.session_state["local_api_url"] = local_url
        api_url = local_url.rstrip("/")

        try:
            resp = requests.get(f"{api_url}/databases", timeout=5)
            databases = resp.json().get("databases", [])
        except Exception:
            databases = []
            st.warning("API not reachable. Start local services or enter your ngrok URL.")

        selected_db = st.selectbox("Database", ["(auto-detect)"] + databases)
        db_id = selected_db if selected_db != "(auto-detect)" else None
        execute_query = st.checkbox("Execute SQL and show results", value=True)

        st.markdown("---")
        try:
            health = requests.get(f"{api_url}/health", timeout=5).json()
            st.success(f"API: {health['status']} | Ollama: {'✓' if health['ollama'] else '✗'}")
        except Exception:
            st.error("API offline")

        st.markdown("---")
        st.caption("Run locally:\n```\npython -m src.serving.local_startup\n```")

    else:  # Groq mode
        default_key = ""
        try:
            default_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

        groq_key = st.text_input("Groq API Key", value=default_key, type="password",
                                  help="Get a free key at console.groq.com")
        groq_model = st.selectbox("Model", [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ])
        db_id_input = st.text_input("Database ID (optional)", placeholder="e.g. concert_singer",
                                     help="Hint for the model about which schema to use")
        db_id = db_id_input.strip() or None
        execute_query = False

        st.markdown("---")
        if groq_key:
            st.success("Groq API key set")
        else:
            st.warning("Enter your Groq API key above")

        st.caption("SQL execution is unavailable in Groq mode\n(requires local database access).")

# ── Main ─────────────────────────────────────────────────────────────────────
st.caption(f"Mode: **{'Local phi3-nl2sql' if mode.startswith('Local') else 'Groq — ' + groq_model}**")

question = st.text_area(
    "Natural language question",
    placeholder="How many singers performed in more than 2 concerts?",
    height=100,
)

can_submit = bool(question.strip()) and (
    mode.startswith("Local") or (mode.startswith("Groq") and groq_key)
)

if st.button("Generate SQL", type="primary", disabled=not can_submit):
    with st.spinner("Generating SQL..."):

        if mode.startswith("Local"):
            try:
                payload = {"question": question, "db_id": db_id, "execute": execute_query}
                resp = requests.post(f"{api_url}/query", json=payload, timeout=120)
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
                st.error("Cannot connect to API. Start local services or update the ngrok URL.")
            except Exception as e:
                st.error(f"Error: {e}")

        else:  # Groq mode
            try:
                from groq import Groq

                client = Groq(api_key=groq_key)
                system_prompt = (
                    "You are an expert SQL assistant. "
                    "Given a natural language question, generate a correct SQL query. "
                    "Output only the SQL with no explanation."
                )
                db_hint = f"Database: {db_id}\n\n" if db_id else ""
                user_msg = f"{db_hint}Question: {question}"

                response = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0,
                    max_tokens=256,
                )
                raw_sql = response.choices[0].message.content.strip()
                formatted_sql = sqlparse.format(raw_sql, reindent=True, keyword_case="upper")

                st.subheader("Generated SQL")
                st.code(formatted_sql, language="sql")
                st.info("SQL execution not available in Groq mode (requires local database access).")

            except Exception as e:
                st.error(f"Groq error: {e}")

# ── Examples ─────────────────────────────────────────────────────────────────
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
