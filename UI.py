from grpc import generate_text
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st



# app config
st.set_page_config(page_title="GraphRAG", page_icon="🤖")
st.title("Entity")
st.subheader('AI chatbot responds based on Graph DB')

# st.sidebar.write("Ask From Your Document :gear:")
# document = st.sidebar.file_uploader("Upload Here", type=["pdf","txt"])

def get_response(user_query):
    
    return generate_text(user_query)

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm Entity. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner('Wait for it...'):
            response = get_response(user_query)
            st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))
