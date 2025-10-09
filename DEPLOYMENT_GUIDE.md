# AI SpillGuard Deployment Guide

## 🚀 Deployment Options

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements_deploy.txt

# Run Streamlit app
streamlit run app.py

# Run FastAPI backend
uvicorn api:app --reload
```

### 2. Streamlit Cloud Deployment
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy the app (main file: `app.py`)

### 3. Heroku Deployment

#### Setup Files:
Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

#### Deploy Commands:
```bash
heroku create your-app-name
heroku buildpacks:add heroku/python
git push heroku main
```

### 4. Docker Deployment
```bash
# Build image
docker build -t ai-spillguard .

# Run container
docker run -p 8501:8501 ai-spillguard

# Docker Compose (if needed)
docker-compose up
```

### 5. AWS EC2 Deployment
```bash
# SSH into EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install dependencies
sudo yum update
sudo yum install python3 pip3 git

# Clone repository
git clone https://github.com/YourUsername/AI_SpillGuard_OSD.git
cd AI_SpillGuard_OSD

# Install requirements
pip3 install -r requirements_deploy.txt

# Run with nohup for persistent execution
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
```

### 6. Azure Container Instances
```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image ai-spillguard .

# Deploy to Container Instances
az container create \
    --resource-group myResourceGroup \
    --name ai-spillguard \
    --image myregistry.azurecr.io/ai-spillguard:latest \
    --ports 8501 \
    --dns-name-label ai-spillguard
```

### 7. Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/ai-spillguard
gcloud run deploy --image gcr.io/PROJECT-ID/ai-spillguard --platform managed
```

## 📦 Production Considerations

### Environment Variables
```bash
export MODEL_PATH="/path/to/best_model.pth"
export UPLOAD_FOLDER="/tmp/uploads"
export MAX_FILE_SIZE="10MB"
```

### Performance Optimization
- Use GPU instances for faster inference
- Implement caching for frequently used models
- Add request rate limiting
- Use CDN for static assets

### Security
- Add authentication and authorization
- Implement input validation
- Use HTTPS in production
- Sanitize file uploads

### Monitoring
- Add logging and error tracking
- Implement health checks
- Monitor resource usage
- Set up alerts for failures

## 🔧 Configuration Files

### requirements_deploy.txt (Production)
```
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
opencv-python-headless==4.8.0.76
albumentations==1.3.1
numpy==1.24.3
Pillow==10.0.0
plotly==5.15.0
fastapi==0.103.0
uvicorn==0.23.2
python-multipart==0.0.6
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  ai-spillguard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - MODEL_PATH=/app/models/best_model.pth
    volumes:
      - ./models:/app/models
```

Choose the deployment option that best fits your needs!