# Copilot Instructions — im2fit

Purpose: Give AI coding agents the minimum, project-specific context to be productive here.

## Big Picture
- Web API: FastAPI app (`app/main.py`) exposes `POST /process`, `GET /health`, `GET /version`, plus index/Swagger. Served by Gunicorn+Uvicorn (`Dockerfile`, `app/startup.txt`).
- Flow: Client uploads an image → `pipeline.run_pipeline(bytes)` → compute overlay PNG, CSV (metrics), STL → upload to Azure Blob → return public URLs.
- Segmentation backends (switchable): Hosted Roboflow (`ROBOFLOW_*`) or local ONNX YOLO (`USE_ONNX=1`, model at `ONNX_MODEL_PATH`, default `model/best.onnx`).
- Scaling: ArUco marker detection (`scale_aruco.py`) returns `mm_per_px` and a confidence; pipeline falls back to a default if not found.

## Environments & Secrets
- Required for uploads: `AZURE_STORAGE_CONNECTION_STRING`; optional `BLOB_CONTAINER` (default `im2fit-outputs`). App fails fast at import if missing.
- Backend selection: Hosted needs `ROBOFLOW_API_KEY` + `ROBOFLOW_INFER_URL`; local needs `USE_ONNX=1` + `ONNX_MODEL_PATH`.
- Version route: `GET /version` includes an ONNX model SHA-256 short hash when present.
- Optional: Kaggle creds (`KAGGLE_USERNAME`, `KAGGLE_KEY`) used only for dataset download/training.

## Dev Workflows
- Install deps: `make setup` (uses `app/requirements.txt`).
- Run API (dev): `uvicorn app.main:app --reload --port 8000` or `make dev`.
- Run tests: `pytest -q` (see `pytest.ini`). Tests mock ONNX/Aruco for determinism.
- Build/run container: see `Dockerfile`; healthcheck hits `/health`.
- Azure App Service: Startup matches `app/startup.txt`; set env vars in Portal.

## Code Structure & Patterns (under `app/`)
- `main.py`: Routes, CORS, Blob upload helper (`upload_bytes`). Keep imports light; environment check at import.
- `pipeline.py`: Orchestrates segmentation (hosted vs ONNX), scaling, metrics, overlay, STL, CSV. Pure functions on bytes/arrays.
- `onnx_infer.py`: Lazy `onnxruntime` session, 640×640 RGB preproc, heuristic YOLO-seg postproc.
- `scale_aruco.py`: OpenCV contrib ArUco DICT_4X4_50 → `(mm_per_px, confidence)`.
- `overlay.py`: Draw contour + metric text → PNG bytes.
- `to_3d.py`: Densify polygon → simple prism extrusion + parabolic crown → STL via `trimesh`.

## Data Contracts
- `run_pipeline(image_bytes)` returns `{ overlay_png: bytes, csv_bytes: bytes, stl_bytes: bytes, metrics: dict }`.
- `metrics` includes: `length_mm`, `width_prox_mm`, `width_mid_mm`, `width_dist_mm`, `mm_per_px`, `scale_confidence`, `mask_area_px`, `sharpness` (+ PCA artifacts).

## Testing Guidance
- Unit tests: `tests/` (see `test_pipeline.py`). Use monkeypatch to stub heavy/external deps:
  - `app.onnx_infer.infer_mask` (avoid real ONNX).
  - `app.scale_aruco.mm_per_pixel_from_aruco` (control scaling).

## Gotchas
- Import-time guard: `AZURE_STORAGE_CONNECTION_STRING` required by `main.py`. For local unit tests, import `app.pipeline` directly or set a dummy value.
- ONNX variance: `onnx_infer._postprocess` uses heuristics; adjust `mask_dim` logic if swapping models.
- ArUco requires `opencv-contrib-python` (already in `requirements.txt`).
