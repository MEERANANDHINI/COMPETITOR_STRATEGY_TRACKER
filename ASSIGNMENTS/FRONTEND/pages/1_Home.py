import streamlit as st

st.title("Welcome to Competitor Strategy Tracker")
st.write("Track and analyze competitor strategies with the help of AI.")

if st.button("Ask Bot"):
    st.switch_page("pages/2_Competitor_Bot.py")
