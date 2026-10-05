import os
import re
import streamlit as st

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥"
)

st.title("🎥 YouTube RAG Chatbot")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()


def extract_video_id(url):
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def create_rag(video_url):
    video_id = extract_video_id(video_url)

    if not video_id:
        raise Exception("Invalid YouTube URL")

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en"])
        transcript = " ".join(
            chunk.text for chunk in transcript_list
        )

    except TranscriptsDisabled:
        raise Exception("Captions are not available for this video")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
you are a helpful assistant.
Answer only from the provided transcript context.
if the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain, video_id


if "main_chain" not in st.session_state:
    st.session_state.main_chain = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


video_url = st.text_input(
    "YouTube Video URL",
    placeholder="Paste YouTube video link here"
)


if st.button("Load Video", use_container_width=True):

    if not video_url:
        st.warning("Please enter a YouTube URL")

    else:

        try:

            with st.spinner("Creating RAG system..."):

                main_chain, video_id = create_rag(video_url)

                st.session_state.main_chain = main_chain
                st.session_state.video_id = video_id
                st.session_state.messages = []

            st.success("Video loaded successfully!")

        except Exception as e:

            st.error(str(e))


if st.session_state.video_id:

    st.video(
        f"https://www.youtube.com/watch?v={st.session_state.video_id}"
    )

    st.divider()

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about the video..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = st.session_state.main_chain.invoke(
                    question
                )

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })