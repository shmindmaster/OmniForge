# Copilot Instructions for OmniForge

## Project Overview

**OmniForge** is a computer vision platform using YOLO models for object detection.

## Azure Infrastructure

### Subscription

- **Subscription ID**: 44e77ffe-2c39-4726-b6f0-2c733c7ffe78 (mahumtech)

### Resource Groups

- **rg-shared-web**: Container Apps (Python backend)
- **rg-shared-dns**: DNS zones (shtrial.com)
- **rg-shared-ai**: Shared AI resources

### Domain Configuration

- **Custom Domain**: https://omniforge.shtrial.com (pending deployment)
- **DNS Zone**: shtrial.com (in rg-shared-dns)
- **Deployment**: Azure Container Apps (Python + YOLO models)

### Shared AI Resources

- **Azure OpenAI**: shared-openai-eastus2 (https://shared-openai-eastus2.openai.azure.com/)
- **Azure AI Search**: shared-search-eastus2 (https://shared-search-eastus2.search.windows.net)

## GitHub Repository

- **Owner**: shmindmaster
- **Repository**: https://github.com/shmindmaster/OmniForge

## Technology Stack

- **Backend**: Python 3.11+
- **Computer Vision**: YOLO11 (yolo11n.pt model)
- **Deployment**: Azure Container Apps (containerized Python)

## Build & Deployment

### Docker Build

```bash
# Build Docker image
docker build -t omniforge .

# Deploy to Azure Container Apps
az containerapp up --name omniforge \
  --resource-group rg-shared-web \
  --subscription mahumtech \
  --image <your-acr>.azurecr.io/omniforge:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 --memory 2.0Gi  # YOLO models need more resources
```

## Environment Variables (Container Apps)

```bash
AZURE_OPENAI_ENDPOINT=https://shared-openai-eastus2.openai.azure.com/
AZURE_OPENAI_KEY=<from-key-vault>
AZURE_SEARCH_ENDPOINT=https://shared-search-eastus2.search.windows.net
MODEL_PATH=/app/models/yolo11n.pt
```

## Important Notes

- **Resource Requirements**: YOLO models need at least 1.0 CPU and 2.0 GiB memory
- **Model Storage**: yolo11n.pt file must be included in Docker image or mounted from Azure Storage
- **GPU Optional**: Can use CPU-only deployment for basic inference

## Agent Behavior Rules

1. Deploy to Azure Container Apps (NOT Static Web Apps)
2. Allocate sufficient CPU/memory for YOLO models
3. Use shared AI resources for non-CV AI features
4. Never create per-app resource groups

## Additional Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/)
