import os, time, hashlib, threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Path
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from azure.storage.blob import BlobServiceClient, ContentSettings
from .pipeline import run_pipeline

BLOB_CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.environ.get("BLOB_CONTAINER") or os.environ.get("STORAGE_CONTAINER", "im2fit-outputs")
PUBLIC_ARTIFACT_PROXY = os.environ.get("PUBLIC_ARTIFACT_PROXY", "0") == "1"

# Don't hard-fail at import for dev; enforce at first upload
if not BLOB_CONN:
    print("WARNING: AZURE_STORAGE_CONNECTION_STRING not set; uploads will fail", flush=True)

MODEL_HASH = None
model_path = os.environ.get("ONNX_MODEL_PATH", "model/best.onnx")
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            MODEL_HASH = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception as e:
        print(f"Warning: Could not read model file {model_path}: {e}")
        MODEL_HASH = "no-model"
else:
    print(f"Warning: Model file {model_path} not found")
    MODEL_HASH = "no-model"

app = FastAPI(title="im2fit Prototype", version="0.1")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# CORS for same-origin and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"], 
    allow_headers=["*"],
)

def _blob_client():
    if not BLOB_CONN:
        raise RuntimeError("Blob connection string not configured")
    return BlobServiceClient.from_connection_string(BLOB_CONN).get_container_client(BLOB_CONTAINER)

def upload_bytes(name: str, data: bytes, content_type: str) -> str:
    cc = _blob_client()
    cc.upload_blob(name, data, overwrite=True,
                   content_settings=ContentSettings(content_type=content_type))
    if PUBLIC_ARTIFACT_PROXY:
        return f"/artifact/{name}"
    blob = cc.get_blob_client(name)
    return blob.url

@app.get("/health")
def health():
        return {"status": "ok"}

@app.get("/version")
def version():
        return {"model_hash": MODEL_HASH, "ts": int(time.time())}

@app.get("/")
def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

@app.get("/artifact/{name}")
async def get_artifact(name: str = Path(...)):
    """Server-side proxy to stream blobs when PUBLIC_ARTIFACT_PROXY=1"""
    if not PUBLIC_ARTIFACT_PROXY:
        raise HTTPException(status_code=404, detail="Artifact proxy disabled")
    
    try:
        cc = _blob_client()
        blob_client = cc.get_blob_client(name)
        blob_data = blob_client.download_blob()
        
        # Get content type from blob properties
        props = blob_client.get_blob_properties()
        content_type = props.content_settings.content_type or "application/octet-stream"
        
        def iterfile():
            for chunk in blob_data.chunks():
                yield chunk
                
        return StreamingResponse(iterfile(), media_type=content_type)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {e}")

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    
    img_bytes = await file.read()
    
    # Enforce file size ≤ 5 MB
    if len(img_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be ≤ 5 MB")

    try:
        result = run_pipeline(img_bytes)  # overlay_png, csv_bytes, stl_bytes, metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    ts = int(time.time())
    base = f"{ts}"
    overlay_url = upload_bytes(f"{base}_overlay.png", result["overlay_png"], "image/png")
    csv_url     = upload_bytes(f"{base}_measures.csv", result["csv_bytes"], "text/csv")
    stl_url     = upload_bytes(f"{base}_nail.stl", result["stl_bytes"], "model/stl")

    return JSONResponse({
        "overlay_url": overlay_url,
        "csv_url": csv_url,
        "stl_url": stl_url,
        "backend": "onnx" if os.environ.get("USE_ONNX") == "1" else "hosted",
        "scale_confidence": result.get("metrics", {}).get("scale_confidence"),
        "metrics": result.get("metrics", {})
    })
