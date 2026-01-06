import os
import re
import logging
import tempfile
from datetime import datetime
from pathlib import Path

# Flask imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Document processing imports
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

# Database imports
import chromadb
from chromadb.config import Settings

# AI imports
try:
    import ollama
except ImportError:
    ollama = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('vector_search_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Application configuration"""
    
    # Load environment variables
    load_dotenv()
    
    # Flask Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Ollama Configuration
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3:latest')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
    MAX_RESPONSE_TOKENS = int(os.getenv('MAX_RESPONSE_TOKENS', '500'))
    
    # Database Configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'documents')


class DocumentProcessor:
    """Handles document processing and chunking"""
    
    def __init__(self):
        try:
            if tiktoken:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            else:
                self.encoding = None
        except Exception as e:
            logger.warning(f"Failed to load encoding: {e}")
            self.encoding = None
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        """Split text into overlapping chunks with improved algorithm"""
        try:
            if not text or not text.strip():
                return []
            
            # Clean and normalize text
            text = re.sub(r'\s+', ' ', text.strip())
            words = text.split()
            chunks = []
            
            if len(words) <= chunk_size:
                return [text]
            
            # Create overlapping chunks
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk = ' '.join(chunk_words).strip()
                if chunk:
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []
    
    def process_pdf(self, file_path: str):
        """Process PDF file and return chunks"""
        if not PdfReader:
            return []
        
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                chunks = []
                
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text.strip():
                        page_chunks = self.chunk_text(text)
                        for chunk_idx, chunk in enumerate(page_chunks):
                            chunks.append({
                                'text': chunk,
                                'page': page_num,
                                'page_string': f'Page {page_num}',
                                'chunk_index': chunk_idx,
                                'file': os.path.basename(file_path),
                                'source': os.path.basename(file_path)
                            })
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def process_docx(self, file_path: str):
        """Process DOCX file and return chunks"""
        if not Document:
            return []
        
        try:
            doc = Document(file_path)
            chunks = []
            
            for para_idx, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    para_chunks = self.chunk_text(paragraph.text)
                    for chunk_idx, chunk in enumerate(para_chunks):
                        chunks.append({
                            'text': chunk,
                            'page': 1,
                            'page_string': 'Page 1',
                            'chunk_index': chunk_idx,
                            'file': os.path.basename(file_path),
                            'source': os.path.basename(file_path)
                        })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return []
    
    def process_txt(self, file_path: str):
        """Process TXT file and return chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks_data = self.chunk_text(text)
                chunks = []
                
                for chunk_idx, chunk in enumerate(chunks_data):
                    chunks.append({
                        'text': chunk,
                        'page': 1,
                        'page_string': 'Page 1',
                        'chunk_index': chunk_idx,
                        'file': os.path.basename(file_path),
                        'source': os.path.basename(file_path)
                    })
                
                return chunks
                
        except Exception as e:
            logger.error(f"Error processing TXT {file_path}: {e}")
            return []


class VectorSearchApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        logger.info("Initializing Vector Search Document Q&A Application")
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
        CORS(self.app)
        
        # Initialize services
        self.ollama_client = None
        self.chroma_client = None
        self.collection = None
        self.document_processor = DocumentProcessor()
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        """Setup all Flask routes"""
        
        @self.app.route('/')
        def index():
            try:
                return render_template('index.html')
            except Exception as e:
                logger.error(f"Error rendering index: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            try:
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'services': {
                        'ollama': self.ollama_client is not None,
                        'chromadb': self.collection is not None
                    }
                })
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/process_folder_files', methods=['POST'])
        def process_folder_files():
            """Process uploaded files from folder selection"""
            try:
                if not request.files:
                    return jsonify({'error': 'No files provided'}), 400
                
                if self.collection is None:
                    return jsonify({'error': 'Database not initialized'}), 500
                
                uploaded_files = request.files.to_dict()
                folder_name = request.form.get('folder_name', 'Uploaded Files')
                
                processed_files = []
                total_chunks = 0
                
                # Process each file
                for file_key, file in uploaded_files.items():
                    if file and hasattr(file, 'filename'):
                        filename = file.filename
                        if filename:
                            file_ext = os.path.splitext(filename)[1].lower()
                            
                            # Save file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                file.save(tmp_file.name)
                                
                                # Process based on file type
                                if file_ext == '.pdf':
                                    chunks = self.document_processor.process_pdf(tmp_file.name)
                                elif file_ext == '.docx':
                                    chunks = self.document_processor.process_docx(tmp_file.name)
                                elif file_ext == '.txt':
                                    chunks = self.document_processor.process_txt(tmp_file.name)
                                else:
                                    continue
                                
                                # Add embeddings to ChromaDB
                                for chunk_idx, chunk in enumerate(chunks):
                                    embedding = self.create_embedding(chunk['text'])
                                    if embedding and self.collection:
                                        try:
                                            self.collection.add(
                                                embeddings=[embedding],
                                                documents=[chunk['text']],
                                                metadatas=[{
                                                    'file': filename,
                                                    'folder': folder_name,
                                                    'page': chunk.get('page', 1),
                                                    'page_string': chunk.get('page_string', 'Page 1'),
                                                    'chunk_index': chunk_idx,
                                                    'source': filename
                                                }],
                                                ids=[f"{os.path.splitext(filename)[0]}_{folder_name}_{chunk_idx}"]
                                            )
                                        except Exception as embed_error:
                                            logger.error(f"Error adding embedding: {embed_error}")
                                    
                                    total_chunks += len(chunks)
                                
                                processed_files.append({
                                    'filename': filename,
                                    'status': 'success',
                                    'chunks': len(chunks)
                                })
                                
                                # Clean up temporary file
                                try:
                                    os.unlink(tmp_file.name)
                                except:
                                    pass
                
                return jsonify({
                    'message': f'Successfully processed {len(processed_files)} files from {folder_name}',
                    'processed_files': processed_files,
                    'total_chunks': total_chunks
                })
                
            except Exception as e:
                logger.error(f"Error in process_folder_files: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/query', methods=['POST'])
        def query_documents():
            """Query documents"""
            try:
                if self.collection is None:
                    return jsonify({'error': 'Database not initialized'}), 500
                
                if self.ollama_client is None:
                    return jsonify({'error': 'AI service not available'}), 500
                
                data = request.get_json()
                if not data or not data.get('query'):
                    return jsonify({'error': 'Query is required'}), 400
                
                query = data.get('query', '').strip()
                if not query:
                    return jsonify({'error': 'Query cannot be empty'}), 400
                
                # Perform vector search
                search_results = self.query_vector_search(query)
                
                if not search_results:
                    return jsonify({
                        'answer': 'I apologize, but I could not find any relevant information in your documents to answer your question.',
                        'sources': []
                    })
                
                # Prepare context for AI
                context_parts = []
                sources = []
                
                documents = search_results.get('documents', [[]]) or [[]]
                metadatas = search_results.get('metadatas', [[]]) or [[]]
                
                # Debug: log search results structure
                logger.info(f"Search results keys: {list(search_results.keys()) if search_results else 'None'}")
                logger.info(f"Metadatas structure: {metadatas[:2] if metadatas else 'None'}")
                
                # Use top 3 results
                for i in range(min(3, len(documents[0]) if documents and documents[0] else 0)):
                    if i < len(documents[0]) and i < len(metadatas) and i < len(metadatas[0]):
                        document = documents[0][i] if documents and documents[0] else ""
                        metadata = metadatas[0][i] if metadatas and metadatas[0] else {}
                        
                        # Debug: log metadata structure
                        logger.info(f"Metadata {i}: {metadata}")
                        
                        # Get filename from metadata, extract basename to remove folder path
                        file_path = metadata.get('file') or metadata.get('source', 'Unknown')
                        file_name = os.path.basename(str(file_path)) if file_path and file_path != 'Unknown' else 'Unknown'
                        
                        context_parts.append(f"[{file_name} - {metadata.get('page_string', 'Page 1')}]: {document}")
                        sources.append({
                            'file': file_name,
                            'page_string': metadata.get('page_string', 'Page 1')
                        })
                
                context = "\n\n".join(context_parts)
                
                # Generate response using Ollama
                answer = self.generate_ai_response(query, context)
                
                return jsonify({
                    'answer': answer,
                    'sources': sources
                })
                
            except Exception as e:
                logger.error(f"Error in query_documents: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/get_inventory', methods=['GET'])
        def get_inventory():
            """Get document inventory"""
            try:
                logger.info("Getting inventory")
                
                if self.collection is None:
                    logger.warning("Collection not initialized")
                    return jsonify({'folders': {}})
                
                results = self.collection.get()
                
                if not results or not results.get('metadatas'):
                    logger.info("No documents found in inventory")
                    return jsonify({'folders': {}})
                
                inventory = {}
                metadatas = results.get('metadatas', [])
                
                for metadata in metadatas or []:
                    if isinstance(metadata, dict) and metadata.get('source'):
                        folder = metadata.get('folder', 'Unknown')
                        file_path = metadata.get('file', 'Unknown')
                        file = os.path.basename(str(file_path)) if file_path else 'Unknown'
                        page = metadata.get('page_string', 'Page 1')
                        
                        if folder not in inventory:
                            inventory[folder] = {
                                'files': {},
                                'chunk_count': 0
                            }
                        
                        if file not in inventory[folder]['files']:
                            inventory[folder]['files'][file] = {
                                'page_numbers': [],
                                'chunks': []
                            }
                        
                        if metadata.get('chunk_index') is not None:
                            inventory[folder]['files'][file]['page_numbers'].append(metadata.get('page', 1))
                            inventory[folder]['files'][file]['chunks'].append(metadata.get('chunk_index', 0))
                            inventory[folder]['chunk_count'] += 1
                
                return jsonify({'folders': inventory})
                
            except Exception as e:
                logger.error(f"Error getting inventory: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/clear_db', methods=['POST'])
        def clear_database():
            """Clear entire database"""
            try:
                if self.collection is None and self.chroma_client is not None:
                    self.chroma_client.delete_collection(name=Config.COLLECTION_NAME)
                    self.collection = self.chroma_client.get_or_create_collection(name=Config.COLLECTION_NAME)
                    logger.info("Database cleared successfully")
                return jsonify({'message': 'Database cleared successfully'})
            except Exception as e:
                logger.error(f"Error clearing database: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/clear_folder', methods=['POST'])
        def clear_folder():
            """Clear specific folder from database"""
            try:
                if self.collection is None:
                    return jsonify({'error': 'Collection not initialized'}), 500
                
                data = request.get_json()
                if not data or not data.get('folder_path'):
                    return jsonify({'error': 'folder_path is required'}), 400
                
                folder_path = data.get('folder_path')
                
                # Get all documents with matching folder
                results = self.collection.get()
                if not results or not results.get('metadatas'):
                    return jsonify({'message': f'No documents found in folder: {folder_path}'})
                
                ids_to_delete = []
                metadatas = results.get('metadatas', [])
                
                for i, metadata in enumerate(metadatas or []):
                    if isinstance(metadata, dict) and metadata.get('folder') == folder_path:
                        ids_to_delete.append(results.get('ids', [])[i])
                
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    logger.info(f"Cleared {len(ids_to_delete)} documents from folder: {folder_path}")
                    return jsonify({'message': f'Successfully cleared {len(ids_to_delete)} documents from folder: {folder_path}'})
                else:
                    return jsonify({'message': f'No documents found in folder: {folder_path}'})
                    
            except Exception as e:
                logger.error(f"Error clearing folder: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/delete_folder', methods=['POST'])
        def delete_folder():
            """Delete specific folder from database (alias for clear_folder)"""
            return clear_folder()
        
        @self.app.route('/delete_file', methods=['POST'])
        def delete_file():
            """Clear specific file from database"""
            try:
                if self.collection is None:
                    return jsonify({'error': 'Collection not initialized'}), 500
                
                data = request.get_json()
                if not data or not data.get('folder_path') or not data.get('file_name'):
                    return jsonify({'error': 'folder_path and file_name are required'}), 400
                
                folder_path = data.get('folder_path')
                file_name = data.get('file_name')
                
                # Get all documents with matching folder and file
                results = self.collection.get()
                if not results or not results.get('metadatas'):
                    return jsonify({'message': f'No documents found for file: {file_name}'})
                
                ids_to_delete = []
                metadatas = results.get('metadatas', [])
                
                for i, metadata in enumerate(metadatas or []):
                    if isinstance(metadata, dict) and metadata.get('folder') == folder_path and os.path.basename(metadata.get('file', '')) == file_name:
                        ids_to_delete.append(results.get('ids', [])[i])
                
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    logger.info(f"Cleared {len(ids_to_delete)} documents from file: {file_name}")
                    return jsonify({'message': f'Successfully cleared {len(ids_to_delete)} chunks from file: {file_name}'})
                else:
                    return jsonify({'message': f'No documents found for file: {file_name}'})
                    
            except Exception as e:
                logger.error(f"Error clearing file: {e}")
                return jsonify({'error': str(e)}), 500
    
    def initialize_services(self):
        """Initialize Ollama and ChromaDB"""
        try:
            # Initialize Ollama
            if ollama:
                logger.info(f"Connecting to Ollama at {Config.OLLAMA_HOST}")
                self.ollama_client = ollama.Client(host=Config.OLLAMA_HOST)
                
                # Test Ollama connection
                try:
                    response = self.ollama_client.list()
                    logger.info("Ollama client initialized successfully")
                    
                    # Check if required models are available
                    available_models = [model['model'] for model in response['models']]
                    logger.info(f"Available models: {available_models}")
                    
                    if Config.OLLAMA_MODEL not in available_models:
                        logger.warning(f"Model {Config.OLLAMA_MODEL} not found. Available models: {available_models}")
                    if Config.EMBEDDING_MODEL not in available_models:
                        logger.warning(f"Embedding model {Config.EMBEDDING_MODEL} not found. Available models: {available_models}")
                        
                except Exception as ollama_test_error:
                    logger.error(f"Ollama connection test failed: {ollama_test_error}")
                    raise ConnectionError(f"Failed to connect to Ollama: {ollama_test_error}")
            
            # Initialize ChromaDB
            logger.info(f"Initializing ChromaDB at {Config.CHROMA_DB_PATH}")
            os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_or_create_collection(name=Config.COLLECTION_NAME)
            logger.info(f"ChromaDB initialized successfully with collection: {Config.COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
    
    def create_embedding(self, text: str):
        """Create embedding"""
        if not text or not text.strip():
            return None
        
        try:
            if self.ollama_client:
                response = self.ollama_client.embeddings(
                    model=Config.EMBEDDING_MODEL,
                    prompt=text.strip()
                )
                return response['embedding']
            else:
                logger.error("Ollama client not initialized")
                return None
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None
    
    def query_vector_search(self, query: str, n_results: int = 5):
        """Perform vector search"""
        try:
            if self.collection is None:
                logger.error("Collection not initialized")
                return []
            
            # Create embedding for query
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                logger.error("Failed to create query embedding")
                return []
            
            # Search vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def generate_ai_response(self, query: str, context: str):
        """Generate AI response using Ollama"""
        try:
            if self.ollama_client:
                response = self.ollama_client.chat(
                    model=Config.OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a helpful assistant that answers questions based on provided document context. 
                            Follow these guidelines:
                            1. Answer based only on the provided context
                            2. If the answer is not found in the context, clearly state that
                            3. Be concise but thorough
                            4. Do not mention source numbers in your response - the sources will be displayed separately
                            5. If information conflicts across sources, mention the discrepancy"""
                        },
                        {
                            "role": "user", 
                            "content": f"""Based on the following context, please answer the user's question. 

Context:
{context}

Question: {query}

Answer:"""
                        }
                    ],
                    options={
                        'temperature': Config.TEMPERATURE,
                        'num_predict': Config.MAX_RESPONSE_TOKENS
                    }
                )
            
                return response['message']['content'].strip()
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the application"""
        try:
            self.initialize_services()
            
            # Use config values if not provided
            host = host or Config.HOST
            port = port or Config.PORT
            debug = debug if debug is not None else Config.FLASK_DEBUG
            
            logger.info(f"Starting Vector Search Document Q&A Application on {host}:{port}")
            logger.info(f"Debug mode: {debug}")
            if ollama:
                logger.info(f"Ollama model: {Config.OLLAMA_MODEL}")
                logger.info(f"Embedding model: {Config.EMBEDDING_MODEL}")
                logger.info(f"Ollama host: {Config.OLLAMA_HOST}")
            
            self.app.run(host=host or '0.0.0.0', port=port or 5000, debug=debug or False)
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            raise


# Initialize and run application
if __name__ == '__main__':
    app = VectorSearchApp()
    app.run()