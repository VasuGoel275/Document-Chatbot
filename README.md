# 📚 AskDocX — Document Question-Answering System

AskDocX is an AI-powered document question-answering system that allows users to upload PDF files and ask natural language questions about their content. It is built using LangChain, Google Gemini, ChromaDB, and Streamlit, and follows a **Retrieval-Augmented Generation (RAG)** architecture to provide context-aware responses grounded in uploaded documents.

![AskDocX Interface](https://github.com/user-attachments/assets/4b569db6-1eea-4b59-98f0-b27de29acab7)

---

## ✨ Features

- 📄 Upload one or multiple PDF documents
- 🤖 Ask natural language questions about your documents
- 🔎 Semantic search using vector embeddings
- 💡 Detailed, context-aware answers powered by Google Gemini
- 🗂️ Persistent vector storage via ChromaDB
- 💬 Chat history tracking within the session
- ⚡ Efficient document chunking with overlap handling

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini |
| Embeddings | Google Generative AI Embeddings |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| PDF Parsing | PyPDF2 |
| Environment Management | python-dotenv |

---

## 📖 How to Use

1. **Upload PDFs** — Use the sidebar to upload one or more PDF files.
2. **Wait for processing** — The app will extract text, chunk it, and store embeddings.
3. **Ask a question** — Type your question in the input box and click **Ask Question**.
4. **View the answer** — The response is displayed below, with full **chat history** in the expander.

---

## 📁 Project Structure

```
AskDocX/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
└── README.md
```

---

## 📦 Requirements

```
langchain
streamlit
google-generativeai
python-dotenv
PyPDF2
chromadb
langchain_google_genai
```
## ⚙️ Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/vasug27/Document_Chatbot.git
cd AskDocX
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Run the app

```bash
streamlit run app.py
```



---

## 🧑 Author

**Vasu Goel**

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:vasugoel2754@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vasugoel503/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vasug27)

---
