import os
import pdfplumber
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import Together
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv
import tempfile
import logging
from collections import deque
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Verify API key is loaded
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("⚠️  WARNING: TOGETHER_API_KEY not found in environment variables!")
    print("Please create a .env file with: TOGETHER_API_KEY=your_api_key_here")

# Global variables for multi-document memory
MAX_PDFS = 5
pdf_memory = deque(maxlen=MAX_PDFS)  # Store last 5 PDFs
combined_db = None
chat_history = []
current_documents = {}

class PDFDocument:
    def __init__(self, name, file_path, db, chunk_count, upload_time):
        self.name = name
        self.file_path = file_path
        self.db = db
        self.chunk_count = chunk_count
        self.upload_time = upload_time

def validate_pdf(file):
    """Validate uploaded PDF file"""
    if file is None:
        return False, "No file uploaded"
    
    if not file.name.lower().endswith('.pdf'):
        return False, "Please upload a PDF file"
    
    # Check file size (limit to 50MB)
    try:
        file_size = os.path.getsize(file.name)
        if file_size > 50 * 1024 * 1024:
            return False, "File too large. Please upload a PDF smaller than 50MB"
    except:
        pass  # If we can't get file size, continue anyway
    
    return True, "Valid PDF"

def extract_text_from_pdf(file_path):
    """Extract text from PDF with better error handling"""
    try:
        text_content = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(f"Page {page_num + 1}:\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
        
        if not text_content:
            return None, "No readable text found in the PDF"
        
        return "\n\n".join(text_content), None
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return None, f"Error processing PDF: {str(e)}"

def combine_databases():
    """Combine all PDF databases into one searchable database"""
    global combined_db
    
    if not pdf_memory:
        combined_db = None
        return
    
    # Get all documents from all PDFs
    all_documents = []
    for pdf_doc in pdf_memory:
        # Extract documents from FAISS database
        docs = pdf_doc.db.docstore._dict.values()
        for doc in docs:
            # Add PDF name to metadata (use the display name)
            doc.metadata['pdf_name'] = pdf_doc.name
            doc.metadata['upload_time'] = pdf_doc.upload_time
            all_documents.append(doc)
    
    if all_documents:
        # Create new embeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create combined database
        combined_db = FAISS.from_documents(all_documents, embeddings)

def process_pdf_automatically(file):
    """Automatically process uploaded PDF"""
    global pdf_memory, combined_db, chat_history, current_documents
    
    if file is None:
        return "Ready to upload PDF documents", get_documents_info(), gr.update(interactive=False)
    
    try:
        # Check API key first
        if not TOGETHER_API_KEY:
            return "❌ API key not found! Please check your configuration.", get_documents_info(), gr.update(interactive=False)
        
        # Validate file
        is_valid, message = validate_pdf(file)
        if not is_valid:
            return f"❌ {message}", get_documents_info(), gr.update(interactive=False)
        
        # Extract text
        text_content, error = extract_text_from_pdf(file.name)
        if error:
            return f"❌ {error}", get_documents_info(), gr.update(interactive=False)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        text_chunks = text_splitter.split_text(text_content)
        
        if not text_chunks:
            return "❌ No text chunks created from PDF", get_documents_info(), gr.update(interactive=False)
        
        # Create documents with unique identifiers
        pdf_name = os.path.basename(file.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_pdf_id = f"{pdf_name}_{timestamp}"
        
        # Check if document with same name already exists
        existing_names = [doc.name for doc in pdf_memory]
        if pdf_name in existing_names:
            # Add timestamp to make it unique
            display_name = f"{os.path.splitext(pdf_name)[0]}_{timestamp}{os.path.splitext(pdf_name)[1]}"
        else:
            display_name = pdf_name
        
        documents = [Document(
            page_content=chunk, 
            metadata={
                "source": file.name,
                "pdf_name": display_name,
                "unique_id": unique_pdf_id,
                "chunk_id": i
            }
        ) for i, chunk in enumerate(text_chunks)]
        
        # Create embeddings and vector database for this PDF
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        pdf_db = FAISS.from_documents(documents, embeddings)
        
        # Create PDF document object
        pdf_doc = PDFDocument(
            name=display_name,
            file_path=file.name,
            db=pdf_db,
            chunk_count=len(text_chunks),
            upload_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Add to memory (this will automatically remove oldest if at max capacity)
        pdf_memory.append(pdf_doc)
        
        # Combine all databases
        combine_databases()
        
        # Create status message
        status_msg = f"✅ Successfully processed '{display_name}'"
        
        # Create document info
        doc_info = get_documents_info()
        
        return (
            status_msg,
            doc_info,
            gr.update(interactive=True)
        )
        
    except Exception as e:
        logger.error(f"Error in process_pdf_automatically: {e}")
        return f"❌ Error processing PDF: {str(e)}", get_documents_info(), gr.update(interactive=False)

def chat_with_pdf(user_question, history):
    """Handle chat interaction with all PDFs in memory"""
    global combined_db, chat_history
    
    if not combined_db:
        bot_response = "Please upload a PDF document first to start chatting."
        history.append((user_question, bot_response))
        return history, ""
    
    if not user_question.strip():
        bot_response = "Please enter a valid question."
        history.append((user_question, bot_response))
        return history, ""
    
    try:
        # Get relevant documents from combined database
        retriever = combined_db.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(user_question)
        
        # Combine context from relevant documents
        context_parts = []
        source_pdfs = set()
        
        for doc in relevant_docs:
            pdf_name = doc.metadata.get('pdf_name', 'Unknown')
            source_pdfs.add(pdf_name)
            context_parts.append(f"[From {pdf_name}]: {doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt manually
        prompt_text = f"""Use the following context from PDF documents to answer the question. Provide a clear, helpful answer based on the information available. When referencing information, mention which document it came from when relevant.

Context from PDF documents:
{context}

Question: {user_question}

Answer:"""
        
        # Initialize LLM and get response
        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=512,
            together_api_key=TOGETHER_API_KEY
        )
        
        # Get response directly from LLM
        bot_response = llm.invoke(prompt_text)
        
        # Add source information
        if source_pdfs:
            source_info = f"\n\n📚 *Sources: {', '.join(source_pdfs)}*"
            bot_response += source_info
        
        # Update conversation history
        chat_history.append((user_question, bot_response))
        history.append((user_question, bot_response))
        
        return history, ""
        
    except Exception as e:
        logger.error(f"Error in chat_with_pdf: {e}")
        error_message = f"Error generating response: {str(e)}"
        history.append((user_question, error_message))
        return history, ""

def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return [], get_documents_info()

def get_documents_info():
    """Get information about all PDFs in memory"""
    global pdf_memory
    
    if not pdf_memory:
        return "No documents loaded"
    
    info_parts = [f"**Loaded Documents ({len(pdf_memory)}/{MAX_PDFS})**\n"]
    
    for i, pdf_doc in enumerate(reversed(pdf_memory), 1):
        info_parts.append(f"**{i}. {pdf_doc.name}**")
        info_parts.append(f"   Added: {pdf_doc.upload_time}")
        info_parts.append("")
    
    return "\n".join(info_parts)

def remove_oldest_pdf():
    """Remove the oldest PDF from memory"""
    global pdf_memory, combined_db
    
    if pdf_memory:
        removed = pdf_memory.popleft()  # Remove from left (oldest)
        combine_databases()  # Recreate combined database
        return f"Removed: {removed.name}", get_documents_info()
    
    return "No documents to remove", get_documents_info()

# Full-width modern CSS styling
custom_css = """
/* Full-width modern design */
* {
    box-sizing: border-box;
}

body, html {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    height: 100vh !important;
}

:root {
    --primary-color: #2563eb;
    --primary-dark: #1d4ed8;
    --secondary-color: #64748b;
    --success-color: #059669;
    --danger-color: #dc2626;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-tertiary: #f1f5f9;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --border-light: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
}

/* Force full width */
.gradio-container, 
.gradio-container .main,
.gradio-container .wrap {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    min-height: 100vh !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Header full width */
#header {
    width: 100% !important;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    color: white;
    padding: 2rem 3rem;
    margin: 0 !important;
    border-radius: 0 !important;
    box-shadow: var(--shadow-lg);
}

#header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    text-align: center;
}

#header p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0.25rem 0;
    text-align: center;
}

/* Main container full width */
#main-container {
    width: 100% !important;
    max-width: 100% !important;
    padding: 2rem 3rem !important;
    margin: 0 !important;
    min-height: calc(100vh - 200px);
    background: var(--bg-secondary);
}

/* Sidebar styling */
#sidebar {
    background: var(--bg-primary);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
    height: fit-content;
    min-height: 600px;
}

/* Upload area */
#upload-area {
    border: 2px dashed var(--border-light);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    background: var(--bg-tertiary);
    margin-bottom: 2rem;
}

#upload-area:hover {
    border-color: var(--primary-color);
    background: rgba(37, 99, 235, 0.05);
    transform: translateY(-2px);
}

/* Chat section full width */
#chat-container {
    background: var(--bg-primary);
    border-radius: 16px;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
    display: flex;
    flex-direction: column;
    height: 80vh;
    min-height: 700px;
    overflow: hidden;
}

/* Chat header */
#chat-header {
    background: var(--bg-tertiary);
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border-light);
}

#chat-header h3 {
    margin: 0;
    color: var(--text-primary);
    font-size: 1.25rem;
    font-weight: 600;
}

/* Chatbot full height */
#chatbot {
    flex: 1 !important;
    height: 100% !important;
    border: none !important;
    border-radius: 0 !important;
    background: var(--bg-primary) !important;
    padding: 1rem !important;
}

/* Chat input area */
#chat-input {
    background: var(--bg-primary);
    padding: 1.5rem 2rem;
    border-top: 1px solid var(--border-light);
}

/* Modern buttons */
.btn-primary {
    background: var(--primary-color) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
    font-size: 0.95rem !important;
}

.btn-primary:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

.btn-secondary {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    font-size: 0.9rem !important;
}

.btn-secondary:hover {
    background: var(--bg-secondary) !important;
    border-color: var(--secondary-color) !important;
    color: var(--text-primary) !important;
}

/* Status messages */
.status-success {
    color: var(--success-color) !important;
    background: rgba(5, 150, 105, 0.1) !important;
    border: 1px solid rgba(5, 150, 105, 0.3) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    font-weight: 500 !important;
}

.status-error {
    color: var(--danger-color) !important;
    background: rgba(220, 38, 38, 0.1) !important;
    border: 1px solid rgba(220, 38, 38, 0.3) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    font-weight: 500 !important;
}

/* Input styling */
.input-modern {
    border: 2px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.input-modern:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    outline: none !important;
}

/* Document info */
#doc-info {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 10px !important;
    padding: 1.5rem !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    color: var(--text-secondary) !important;
}

/* Message styling */
.message.user {
    background: var(--primary-color) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    margin: 8px 0 8px auto !important;
    max-width: 75% !important;
    box-shadow: var(--shadow-sm) !important;
}

.message.bot {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 18px !important;
    margin: 8px auto 8px 0 !important;
    max-width: 80% !important;
    border: 1px solid var(--border-light) !important;
}

/* Responsive full width */
@media (max-width: 1200px) {
    #main-container {
        padding: 1.5rem 2rem !important;
    }
    
    #header {
        padding: 1.5rem 2rem;
    }
}

@media (max-width: 768px) {
    #main-container {
        padding: 1rem !important;
    }
    
    #header {
        padding: 1rem;
    }
    
    #header h1 {
        font-size: 2rem;
    }
    
    #chat-container {
        height: 70vh;
        min-height: 500px;
    }
    
    .message.user, .message.bot {
        max-width: 90% !important;
    }
    
    #sidebar {
        padding: 1.5rem;
    }
}

/* Smooth scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-tertiary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--secondary-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}

/* Animation */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: slideIn 0.3s ease-out;
}
"""

# Create the full-width modern Gradio interface
with gr.Blocks(
    css=custom_css,
    title="AI PDF Chat Assistant",
    theme=gr.themes.Default().set(
        body_background_fill="#f8fafc",
        block_background_fill="white"
    )
) as demo:
    
    # Full-width Header
    gr.HTML("""
    <div id="header">
        <h1>AI PDF Chat Assistant</h1>
        <p>Upload PDF documents and chat with them using advanced AI technology</p>
        <p><em>Powered by Mistral-7B • Multi-document memory • Intelligent search</em></p>
    </div>
    """)
    
    # Main Container - Full Width
    with gr.Row(elem_id="main-container"):
        # Sidebar - Document Management
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.HTML("<h3 style='margin: 0 0 1.5rem 0; color: #0f172a; font-weight: 600;'>📁 Document Upload</h3>")
            
            # Auto-processing file upload
            with gr.Group(elem_id="upload-area"):
                file_uploader = gr.File(
                    label="Drop your PDF here or click to browse",
                    file_types=[".pdf"],
                    type="filepath"
                )
            
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready to upload PDF documents",
                lines=3,
                elem_classes=["status-success"]
            )
            
            gr.HTML("<h3 style='margin: 2rem 0 1rem 0; color: #0f172a; font-weight: 600;'>📚 Document Library</h3>")
            doc_info = gr.Textbox(
                label="Loaded Documents",
                interactive=False,
                value="No documents loaded",
                lines=8,
                elem_id="doc-info"
            )
            
            with gr.Row():
                refresh_btn = gr.Button(
                    "🔄 Refresh",
                    size="sm",
                    elem_classes=["btn-secondary"]
                )
                remove_btn = gr.Button(
                    "🗑️ Remove Oldest",
                    size="sm",
                    elem_classes=["btn-secondary"]
                )
        
        # Main Chat Area - Full Height
        with gr.Column(scale=3, elem_id="chat-container"):
            # Chat Header
            gr.HTML("""
            <div id="chat-header">
                <h3>💬 Chat with your Documents</h3>
            </div>
            """)
            
            # Enhanced Chat Interface
            chatbot = gr.Chatbot(
                show_label=False,
                avatar_images=None,
                bubble_full_width=False,
                elem_id="chatbot",
                container=False
            )
            
            # Chat Input Area
            with gr.Group(elem_id="chat-input"):
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask me anything about your uploaded documents...",
                        scale=6,
                        show_label=False,
                        lines=1,
                        elem_classes=["input-modern"]
                    )
                    
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        elem_classes=["btn-primary"]
                    )
                
                # Action Buttons
                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️ Clear Chat",
                        variant="secondary",
                        elem_classes=["btn-secondary"]
                    )
                    example_btn = gr.Button(
                        "📊 Summarize Documents",
                        variant="secondary",
                        elem_classes=["btn-secondary"]
                    )
    
    # Event Handlers
    
    # Auto-process PDF on upload
    file_uploader.change(
        fn=process_pdf_automatically,
        inputs=[file_uploader],
        outputs=[upload_status, doc_info, send_btn]
    )
    
    # Chat functionality
    send_btn.click(
        fn=chat_with_pdf,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )
    
    user_input.submit(
        fn=chat_with_pdf,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )
    
    # Clear chat
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, doc_info]
    )
    
    # Refresh documents
    refresh_btn.click(
        fn=get_documents_info,
        outputs=[doc_info]
    )
    
    # Remove oldest document
    remove_btn.click(
        fn=remove_oldest_pdf,
        outputs=[upload_status, doc_info]
    )
    
    # Example button
    def ask_summary():
        return "Please provide a comprehensive summary of all the documents I have uploaded."
    
    example_btn.click(
        fn=ask_summary,
        outputs=[user_input]
    )
    
    # Initialize document info on load
    demo.load(
        fn=get_documents_info,
        outputs=[doc_info]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )