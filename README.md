# im2fit Prototype – FastAPI on Azure App Service

**What this does**
Upload a hand photo (with a printed 20 mm ArUco marker) → get back:

- **Overlay PNG** showing segmentation + measurements (mm)
- **CSV** with accurate millimeter metrics
- **STL** 3D file of the nail contour

All results are stored in **Azure Blob Storage** and returned as public URLs via the web response.

---

## Features & Tech Stack

| Component             | Description                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Web Framework**     | FastAPI, served via Gunicorn + Uvicorn                                                                                                           |
| **Image Storage**     | Azure Blob Storage                                                                                                                               |
| **Nail Segmentation** | Roboflow Hosted Instance Segmentation API (or swap to ONNX later)                                                                                |
| **Scaling to mm**     | OpenCV ArUco marker detection ([OpenCV ArUco tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html?utm_source=chatgpt.com)) |
| **STL Generation**    | Trimesh — trusted for 3D mesh exports ([Trimesh export docs](https://trimesh.org/trimesh.exchange.export.html?utm_source=chatgpt.com))           |
| **Deployment**        | Azure App Service (Linux, Python) — fast and reliable                                                                                            |
| **CI/CD**             | GitHub Actions (auto-deploy) or zip deployment                                                                                                   |

---

## 🆕 Quick Deploy (under 15 min)

1. **Set up Azure resources**:

   - Create **App Service (Linux, Python)**.
   - Create **Storage Account** + **Blob container** (e.g., `im2fit-outputs`).

2. **Configure environment settings** in Azure Portal → _Configuration_:

   - `AZURE_STORAGE_CONNECTION_STRING` (from Storage Account).
   - `BLOB_CONTAINER` (e.g., `im2fit-outputs`).
   - If using Roboflow: `ROBOFLOW_API_KEY`, `ROBOFLOW_INFER_URL`.
   - (Optional) For local inference later: `USE_ONNX=1`, `ONNX_MODEL_PATH`.

3. **Deploy code**:

   - Connect Azure App Service to your GitHub repo or use zip deploy. Azure will handle installing dependencies via `requirements.txt`, and run the app with the startup command from `startup.txt`.

4. **Test the live endpoint**:
   - Visit `https://<your-app-name>.azurewebsites.net/docs`.
   - Execute `POST /process` and upload a photo containing a **20 mm ArUco marker** printed at 100 %.

---

## Why This Works

- **Quick setup**: App Service auto-deploy + auto-detect Python apps. Gunicorn/Uvicorn setup is Azure-recommended. ([Microsoft Quickstart sample](https://github.com/Azure-Samples/msdocs-python-fastapi-webapp-quickstart?utm_source=chatgpt.com))
- **Reliable segmentation**: Roboflow’s Hosted Instance Segmentation API gives instant polygon masks—no model ops or GPU. ([Roboflow hosting guide](https://docs.roboflow.com/deploy/serverless/instance-segmentation?utm_source=chatgpt.com))
- **Accurate scaling**: Using OpenCV’s ArUco marker tutorial ensures precise mm-per-pixel measurements.
- **STL export**: Trimesh reliably converts nail outlines into real-scale 3D models.
- **Swappable backend**: You can switch to an ONNX-based segmentation pipeline (`USE_ONNX=1`) without changing the API or infrastructure.

---

## Running Locally

Set environment variables (one-time):

Windows PowerShell:

```powershell
setx KAGGLE_USERNAME "<your>"
setx KAGGLE_KEY "<key>"
```

Linux / macOS:

```bash
export KAGGLE_USERNAME=<your>
export KAGGLE_KEY=<key>
```

Install deps & run:

```bash
make setup
uvicorn app.main:app --reload --port 8000
```

### Segmentation Backend Options

Option A (Fine-tune YOLO11 on Kaggle dataset): convert dataset → train → export ONNX → set `USE_ONNX=1`.

Option B (Pretrained Weights Shortcut): download public pretrained nails segmentation weights (YOLOv8/YOLO11), export to ONNX, place at `model/best.onnx`, skip training.

Both options produce the same runtime API.
