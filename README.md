# AI_RAG

AI_RAG is an AI-powered document assistant that allows you to upload PDF documents and ask questions about their content. It uses HuggingFace embeddings, FAISS vector search, and Google's Gemini LLM to provide accurate answers based on your documents.

## Features

- Upload and process PDF documents
- Extract and chunk text for efficient retrieval
- Generate embeddings using HuggingFace models
- Store and search document chunks with FAISS vector database
- Use Gemini LLM to answer user queries based on document context
- Streamlit web interface for easy interaction

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/AI_RAG.git
   cd AI_RAG/AI_RAG
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up your environment variables:**
   - Create a `.env` file in the root directory with your Google API key:
     ```
     GOOGLE_API_1='your-google-api-key'
     ```

## Usage

1. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

2. **Open your browser and follow the Streamlit link.**

3. **Upload a PDF document and ask questions about its content.**

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for Python dependencies

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [HuggingFace](https://huggingface.co/)
- [Google Generative AI](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)