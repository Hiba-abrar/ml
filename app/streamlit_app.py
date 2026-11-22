"""F1 Grid vs Race Reality - final presentation app.

Run from the project root:   streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="F1 Grid vs Race Reality", page_icon="🏁", layout="wide")

pg = st.navigation([
    st.Page("pages/0_Overview.py", title="Overview", icon="🏁", default=True),
    st.Page("pages/1_Data_Explorer.py", title="Data & EDA", icon="📊"),
    st.Page("pages/2_Pipeline_and_Features.py", title="Pipeline & Features", icon="🧬"),
    st.Page("pages/3_Model_Lab.py", title="Model Lab", icon="🧪"),
    st.Page("pages/4_Test_Evaluation.py", title="Test Evaluation", icon="📈"),
    st.Page("pages/5_Race_Predictor.py", title="Race Predictor", icon="🔮"),
    st.Page("pages/6_Surprise_Analysis.py", title="Surprise Analysis", icon="⚡"),
])
pg.run()
