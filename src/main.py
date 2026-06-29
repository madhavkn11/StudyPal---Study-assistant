import os
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from chatbot_utility import get_chapter_list
from get_yt_video import get_yt_video_link


# Load environment variables
load_dotenv()
DEVICE = "cpu"  # Default to 'cpu' if not set

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

subjects_list = [
    "Biology (Class 12)",
    "Operating Systems (B.Tech)"
]

# Helper to setup vectorstore and chat_chain
def get_vector_db_path(chapter, subject):
    if subject == "Operating Systems (B.Tech)":
        return f"{parent_dir}/vector_db/operating_systems_vector_db"

    if chapter == "All Chapters":
        return f"{parent_dir}/vector_db/class_12_biology_vector_db"

    return f"{parent_dir}/chapters_vector_db/{chapter}"

def setup_chain(selected_chapter, selected_subject):
    vector_db_path = get_vector_db_path(selected_chapter, selected_subject)

    print("Vector DB Path:", vector_db_path)
    print("Folder Exists:", os.path.exists(vector_db_path))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    memory = ConversationBufferMemory(llm=llm, output_key='answer', memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=True
    )
    return chain


st.set_page_config(
    page_title="StudyPal",
    page_icon="🌀",
    layout="wide"
)

st.title("📚 StudyPal AI")

st.caption(
    "AI-powered RAG Study Assistant with YouTube Recommendations"
)

# Initialize the chat history and video history as session state in Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_history" not in st.session_state:
    st.session_state.video_history = []
with st.sidebar:
    st.header("📖 Study Settings")

    selected_subject = st.selectbox(
        "Select Subject",
        subjects_list,
        index=None
    )

    if selected_subject:
        chapter_list = get_chapter_list(selected_subject)

        if selected_subject != "Operating Systems (B.Tech)":
            chapter_list.append("All Chapters")

        selected_chapter = st.selectbox(
            "Select Topic",
            chapter_list,
            index=0
        )
        if (
                st.session_state.get("selected_subject") != selected_subject
                or st.session_state.get("selected_chapter") != selected_chapter
        ):
            st.session_state.chat_chain = setup_chain(
                selected_chapter,
                selected_subject
            )

            st.session_state.chat_history = []
            st.session_state.video_history = []

        st.session_state.selected_subject = selected_subject
        st.session_state.selected_chapter = selected_chapter
# Display previous messages
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show video references if present for assistant messages
        if message["role"] == "assistant" and idx < len(st.session_state.video_history):
            video_refs = st.session_state.video_history[idx]
            if video_refs:
                st.subheader("🎥 Recommended Videos")

                for title, link in video_refs:
                    st.markdown(f"**{title}**")
                    st.video(link)
                    st.link_button(
                        "▶ Watch on YouTube",
                        link,
                        use_container_width=True
                    )

# Input field for user's message
user_input = st.chat_input("Ask AI")

if user_input and "chat_chain" in st.session_state:

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    st.session_state.video_history.append(None)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):
            response = st.session_state.chat_chain(
                {"question": user_input}
            )

        st.markdown(response["answer"])

        search_query = user_input

        try:
            video_titles, video_links = get_yt_video_link(search_query)

        except Exception:
            video_titles = []
            video_links = []

        video_refs = []

        if video_titles:

            st.divider()
            st.subheader("🎥 Recommended Videos")
            for i, (title, link) in enumerate(zip(video_titles, video_links)):
                st.markdown(f"**{title}**")

                st.video(link)

                st.link_button(
                    "▶ Watch on YouTube",
                    link,
                    use_container_width=True,
                    
                )

                video_refs.append((title, link))

        else:

            st.info("No related videos found.")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response["answer"]
            }
        )

        st.session_state.video_history.append(video_refs)

elif user_input:

    st.warning("⚠️ Please select a subject and topic first.")