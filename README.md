# Vector Search Document Q&A Application

A powerful Flask-based web application that allows you to upload, process, and query documents using AI-powered vector search. Built with ChromaDB for vector storage and Ollama for AI responses.

## Features

- 📄 **Multi-format Support**: Process PDF, DOCX, and TXT files
- 🔍 **AI-Powered Search**: Query documents using natural language
- 🗂️ **Folder Management**: Organize documents by folders
- 💾 **Vector Storage**: Fast and efficient document retrieval with ChromaDB
- 🤖 **Local AI**: Uses Ollama for privacy-focused document processing
- 🎨 **Modern UI**: Beautiful glassmorphic interface with responsive design

## Architecture

- **Backend**: Flask (Python)
- **Vector Database**: ChromaDB
- **AI Service**: Ollama (local LLM)
- **Frontend**: HTML5, CSS3, JavaScript with Bootstrap
- **Document Processing**: PyPDF2, python-docx

## Prerequisites

- Python 3.8+
- Ollama installed and running
- Sufficient RAM for document processing (recommended: 8GB+)

## Quick Start (Local Development)

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### 2. Start Ollama and Pull Models

```bash
# Start Ollama service
ollama serve

# Pull required models (in separate terminal)
ollama pull llama3:latest
ollama pull nomic-embed-text:latest
```

### 3. Set Up the Application

```bash
# Clone the repository
git clone <your-repo-url>
cd aidp4

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (optional)
cp .env.example .env
# Edit .env with your preferences
```

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment Options

### 🏆 Recommended: Docker Compose (Production)

**Best for**: Production deployments, easy scaling, consistent environments

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - HOST=0.0.0.0
      - PORT=5000
      - OLLAMA_HOST=http://ollama:11434
      - FLASK_DEBUG=False
    volumes:
      - ./chroma_db:/app/chroma_db
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p chroma_db

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
```

**Deploy with Docker Compose**:
```bash
docker-compose up -d
```

### ☁️ Cloud Deployment Options

#### 1. Railway (Easiest)

**Best for**: Quick deployment, hobby projects, minimal configuration

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Environment Variables** (set in Railway dashboard):
```
HOST=0.0.0.0
PORT=5000
OLLAMA_HOST=https://your-ollama-instance.com
FLASK_DEBUG=False
```

#### 2. Render (Great for Free Tier)

**Best for**: Free hosting, good performance, easy GitHub integration

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set these environment variables:
   ```
   HOST=0.0.0.0
   PORT=5000
   OLLAMA_HOST=https://your-ollama-instance.com
   FLASK_DEBUG=False
   PYTHON_VERSION=3.11
   ```
4. Deploy

#### 3. Heroku (Reliable Option)

**Best for**: Scalable deployments, add-ons support

```bash
# Install Heroku CLI
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create your-app-name
heroku config:set HOST=0.0.0.0 PORT=5000 FLASK_DEBUG=False
git push heroku main
```

#### 4. AWS EC2 (Full Control)

**Best for**: Enterprise, custom configurations, high performance

```bash
# On EC2 instance (Ubuntu 22.04)
sudo apt update
sudo apt install python3-pip nginx

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Deploy with Docker Compose
git clone <your-repo>
cd <your-repo>
docker-compose up -d

# Configure Nginx as reverse proxy
sudo nano /etc/nginx/sites-available/your-app
```

#### 5. Google Cloud Run (Serverless)

**Best for**: Serverless, pay-per-use, auto-scaling

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/vector-search-app

# Deploy to Cloud Run
gcloud run deploy vector-search-app \
  --image gcr.io/PROJECT-ID/vector-search-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars HOST=0.0.0.0,PORT=8080,OLLAMA_HOST=https://your-ollama-instance.com
```

### 🏢 Enterprise Deployment

#### Kubernetes (Helm Chart)

```yaml
# values.yaml
replicaCount: 2

image:
  repository: your-registry/vector-search-app
  tag: latest

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: your-domain.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

env:
  - name: HOST
    value: "0.0.0.0"
  - name: PORT
    value: "5000"
  - name: OLLAMA_HOST
    value: "http://ollama-service:11434"
```

## Environment Variables

Create a `.env` file in the project root:

```env
# Flask Configuration
HOST=0.0.0.0
PORT=5000
FLASK_DEBUG=False

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3:latest
EMBEDDING_MODEL=nomic-embed-text:latest
TEMPERATURE=0.1
MAX_RESPONSE_TOKENS=500

# Database Configuration
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=documents
```

## Production Considerations

### Security

1. **HTTPS**: Always use SSL in production
2. **Firewall**: Restrict access to Ollama service
3. **Authentication**: Add user authentication (JWT, OAuth)
4. **Rate Limiting**: Implement API rate limiting
5. **Input Validation**: Sanitize all user inputs

### Performance

1. **Load Balancing**: Use Nginx or cloud load balancers
2. **Caching**: Implement Redis for session storage
3. **Database Optimization**: Configure ChromaDB for production
4. **Resource Limits**: Set appropriate memory and CPU limits

### Monitoring

1. **Health Checks**: Use `/health` endpoint
2. **Logging**: Monitor application logs
3. **Metrics**: Add Prometheus metrics
4. **Alerts**: Set up alerting for failures

## Scaling Ollama

### Single Instance (Small Scale)

```bash
# Increase Ollama memory limits
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_NUM_PARALLEL=4
ollama serve
```

### Multiple Instances (Large Scale)

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  ollama-1:
    image: ollama/ollama:latest
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    volumes:
      - ollama_data_1:/root/.ollama

  ollama-2:
    image: ollama/ollama:latest
    environment:
      - OLLAMA_HOST=0.0.0.0:11435
    volumes:
      - ollama_data_2:/root/.ollama

  load-balancer:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "11434:80"
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Restart Ollama
   ollama serve
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   htop
   
   # Increase swap space
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :5000
   
   # Kill process on port
   sudo kill -9 <PID>
   ```

### Health Check

The application provides a health check endpoint:

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-06T12:00:00",
  "services": {
    "ollama": true,
    "chromadb": true
  }
}
```

## API Documentation

### Endpoints

- `GET /` - Main application interface
- `GET /health` - Health check
- `POST /query` - Query documents
- `POST /process_folder_files` - Process uploaded files
- `GET /get_inventory` - Get document inventory
- `POST /clear_db` - Clear database
- `POST /clear_folder` - Clear specific folder
- `POST /delete_file` - Delete specific file

### Example Query

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics in the documents?"}'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the application logs for errors

---

**Note**: This application requires significant computational resources for AI processing. Ensure your deployment environment has adequate CPU, RAM, and optionally GPU support for optimal performance.