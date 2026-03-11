import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

st.set_page_config(
    page_title="AskDocX",
    page_icon="📚",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.title("📚 AskDocX")
st.caption("A Document Question-Answering System")
st.info("Upload your PDFs, ask questions, and get detailed responses from your documents!")

prompt = ChatPromptTemplate.from_template("""
Answer the question as detailed as possible from the provided context, make sure to provide all the details from the given context only.
Break your answer up into nicely readable paragraphs.
If the answer is not in the context, say "I couldn't find this in the uploaded documents."
Context: {context}
Question: {question}
Answer:""")


def extract_documents_with_metadata(uploaded_pdfs):
    documents = []
    for uploaded_pdf in uploaded_pdfs:
        pdf_reader = PdfReader(uploaded_pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_pdf.name,
                        "page": page_num,
                        "total_pages": len(pdf_reader.pages)
                    }
                ))
    return documents


def embed(uploaded_pdfs):
    raw_docs = extract_documents_with_metadata(uploaded_pdfs)

    if not raw_docs:
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_documents(raw_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=api_key
    )

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_pdf(question, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }) | prompt | ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.4, google_api_key=api_key) | StrOutputParser()
    )

    return rag_chain.invoke(question)

def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "doc_names" not in st.session_state:
        st.session_state.doc_names = []

    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_pdfs = st.file_uploader(
            "Upload your PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDF files"
        )

        if uploaded_pdfs:
            new_names = sorted([f.name for f in uploaded_pdfs])
            if new_names != st.session_state.doc_names:
                with st.spinner("Processing PDFs..."):
                    try:
                        st.session_state.vector_store = None
                        st.session_state.vector_store = embed(uploaded_pdfs)
                        st.session_state.doc_names = new_names
                        st.session_state.chat_history = []
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")

    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_input(
            "🤔 Ask your question",
            placeholder="Type your question here...",
            key="input"
        )

    with col2:
        st.write("")
        st.write("")
        submit = st.button("Ask Question", type="primary", use_container_width=True)

    if submit and st.session_state.vector_store:
        try:
            with st.spinner("Generating response..."):
                response = query_pdf(question, st.session_state.vector_store)

            st.subheader("💡 Response")
            st.write(response)

            st.session_state.chat_history.insert(0, ("Bot 🤖", response))
            st.session_state.chat_history.insert(0, ("You 🙋‍♂️", question))

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    with st.expander("📝 Chat History"):
        for i, (role, text) in enumerate(st.session_state.chat_history):
            st.write(f"**{role}**")
            st.write(text)
            if i < len(st.session_state.chat_history) - 1:
                st.divider()


if __name__ == "__main__":
    main()