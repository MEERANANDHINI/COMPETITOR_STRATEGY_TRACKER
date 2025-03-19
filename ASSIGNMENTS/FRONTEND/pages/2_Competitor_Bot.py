import streamlit as st
import groq

# Set up Groq API key
GROQ_API_KEY = "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"
client = groq.Client(api_key=GROQ_API_KEY)

st.title("AI Competitor Strategy Bot 🤖")
st.write("Ask your questions related to competitor strategies.")

# Sidebar for navigation and settings
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select Option:", ["Previous Chat", "Bot Features"])
    
    if page == "Previous Chat":
        if "chat_history" in st.session_state:
            for chat in st.session_state.chat_history:
                st.write(chat)
        else:
            st.write("No previous chats available.")
    
    elif page == "Bot Features":
        st.header("Bot Settings")
        temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        json_mode = st.checkbox("Enable JSON Mode")
        streaming = st.checkbox("Enable Streaming")

user_input = st.text_input("Ask your question:")

if st.button("Ask"):
    if user_input.strip():
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": user_input}],
            temperature=temp  
        )
        bot_reply = response.choices[0].message.content

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append(f"**You:** {user_input}")
        st.session_state.chat_history.append(f"**Bot:** {bot_reply}")

        st.text_area("Bot's Reply:", bot_reply, height=100)
    else:
        st.warning("Please enter a question!")
