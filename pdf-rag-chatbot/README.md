---
title: PDF RAG Chatbot
sdk: gradio
emoji: 📚
colorFrom: blue
colorTo: purple
pinned: false
sdk_version: 5.23.1
app_file: app.py
license: mit
---

# 📚 PDF RAG Chatbot

An intelligent chatbot that allows you to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## 🌟 Features

- **PDF Upload & Processing**: Extract text from any PDF document
- **Semantic Search**: Find relevant information using sentence embeddings
- **Conversational Interface**: Ask questions naturally and get accurate answers
- **Source-Based Responses**: Answers are grounded in your document content
- **Clean UI**: Built with Gradio for an intuitive user experience

## 🚀 How It Works

This application implements a RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Processing**: Extracts text from uploaded PDFs and splits it into manageable chunks
2. **Embedding Creation**: Converts text chunks into vector embeddings using Sentence Transformers
3. **Vector Storage**: Stores embeddings in a FAISS index for fast similarity search
4. **Query Processing**: When you ask a question, it:
   - Converts your question to an embedding
   - Finds the most relevant chunks from the document
   - Sends the context to an LLM (Mistral-7B) to generate an answer

## 🛠️ Technical Stack

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` - Efficient model for semantic similarity
- **Vector Database**: FAISS - Fast similarity search and clustering
- **LLM**: Llama-3.2-1B-Instruct - Via HuggingFace Inference API (free tier)
- **UI Framework**: Gradio - Interactive web interface
- **PDF Processing**: PyPDF2 - Text extraction from PDFs

## 📋 Usage

1. **Upload a PDF**: Click the upload button and select your PDF file
2. **Process Document**: Click "Process PDF" to analyze the document
3. **Ask Questions**: Type your questions in the chat interface
4. **Get Answers**: Receive contextual answers based on the document content

## 💡 Example Questions

- "What is the main topic of this document?"
- "Can you summarize the key points?"
- "What does it say about [specific topic]?"
- "Are there any statistics or numbers mentioned?"

## ⚙️ Local Setup (Optional)

If you want to run this locally:

```bash
# Clone the repository
git clone https://huggingface.co/spaces/SivaSai8143/pdf-rag-chatbot
cd pdf-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🎯 Use Cases

- Research paper analysis
- Legal document review
- Technical documentation Q&A
- Educational material exploration
- Business report summarization

## 📝 Limitations

- Works best with text-based PDFs (not scanned images)
- Document processing time depends on PDF size
- Answer quality depends on the clarity of the source material
- Uses free-tier Inference API (may have rate limits)

## 🔮 Future Enhancements

- [ ] Support for multiple PDFs simultaneously
- [ ] Citation with page numbers
- [ ] Conversation history persistence
- [ ] Support for scanned PDFs (OCR)
- [ ] Export chat history
- [ ] Advanced chunking strategies

## 🤝 Contributing

Feedback and suggestions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the MIT License.

---

**Built with ❤️ using HuggingFace, Gradio, and Open-Source AI models**
