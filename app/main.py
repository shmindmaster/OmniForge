import os, time, hashlib, threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient, ContentSettings
from .pipeline import run_pipeline

BLOB_CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.environ.get("BLOB_CONTAINER", "im2fit-outputs")

if not BLOB_CONN:
    raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set")

MODEL_HASH = None
if os.path.exists(os.environ.get("ONNX_MODEL_PATH", "model/best.onnx")):
    with open(os.environ.get("ONNX_MODEL_PATH", "model/best.onnx"), 'rb') as f:
        MODEL_HASH = hashlib.sha256(f.read()).hexdigest()[:16]

app = FastAPI(title="im2fit Prototype", version="0.1")
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
    blob = cc.get_blob_client(name)
    return blob.url

@app.get("/health")
def health():
        return {"status": "ok"}

@app.get("/version")
def version():
        return {"model_hash": MODEL_HASH, "ts": int(time.time())}

@app.get("/")
def index():
        html = """
    <html><head><title>im2fit</title></head>
        <body>
    <h1>im2fit Demo</h1>
        <form id='f'>
            <input type='file' name='file' accept='image/*'/>
            <button type='submit'>Upload</button>
        </form>
        <pre id='out'></pre>
        <script>
        const f = document.getElementById('f');
        f.addEventListener('submit', async (e)=>{
            e.preventDefault();
            const fd = new FormData(f);
            const r = await fetch('/process', {method:'POST', body:fd});
            const j = await r.json();
            document.getElementById('out').textContent = JSON.stringify(j, null, 2);
        });
        </script>
        <p><a href='/docs'>Swagger Docs</a></p>
        </body></html>
        """
        return HTMLResponse(html)

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    img_bytes = await file.read()

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
