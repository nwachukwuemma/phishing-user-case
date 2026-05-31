import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM


# 
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env file")
    st.stop()


st.set_page_config(page_title="BNS SOC AI", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #00ffcc;
}
.stButton>button {
    background-color: #00ffcc;
    color: black;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}
.stTextInput>div>div>input {
    background-color: #1c1f26;
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
# BNS AI Agent  
### AI Agent for Cybersecurity Usecases  

🌐 www.bnscyberlab.com  
📧 support@bnscyberlab.com  
""")

st.write("Upload logs, investigate instantly with BNS Agent.")


col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Upload CSV Log File", type=["csv"])

with col2:
    st.info("""
**Supported Use Cases:**
- Login analysis  
- Threat detection  
- IP investigation  
- Activity summarization  
""")


# 
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    with st.expander("📊 Log Preview"):
        st.dataframe(df.head(10))


    # 
    question = st.text_input(" Ask a question about the logs")


    
    if st.button("Analyze Logs") and question:

        with st.spinner("🔍 BNS AI is analyzing logs..."):

            # =========================
            # 9. LLM (OPENROUTER)
            # =========================
            llm = LLM(
                model="openrouter/openai/gpt-4o-mini",
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0
            )


            # =========================
            # 10. AGENT
            # =========================
            agent = Agent(
                role="Senior SOC Analyst",
                goal="Analyze cybersecurity logs and identify threats and anomalies",
                backstory=(
                    "You are a highl experienced SOC analyst at BNS Cyberlab. "
                    "You specialize in detecting threats, analyzing logs, and "
                    "explaining security incidents in simple language."
                ),
                verbose=True,
                allow_delegation=False,
                llm=llm
            )


            # =========================
            # 11. TASK
            # =========================
            task = Task(
                description=f"""
You are analyzing cybersecurity CSV logs.

DATA COLUMNS:
{list(df.columns)}

SAMPLE DATA:
{df.head(10).to_string()}

USER QUESTION:
{question}

INSTRUCTIONS:
- Explain in simple English
- Identify anomalies or suspicious behavior
- Highlight potential threats
- Summarize like a SOC report
""",
                expected_output="Clear SOC-style cybersecurity analysis",
                agent=agent
            )


            # =========================
            # 12. CREW EXECUTION
            # =========================
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True
            )

            result = crew.kickoff()


            # =========================
            # 13. OUTPUT
            # =========================
            st.markdown("## Agent Findings")
            st.success(result)


# =========================
# 14. FOOTER
# =========================
st.markdown("---")
st.markdown("""
### BNS Cyberlab   

🌐 www.bnscyberlab.com  
📧 support@bnscyberlab.com  
""")