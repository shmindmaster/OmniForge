# Copilot Instructions — im2fit

Purpose: Equip AI coding agents with focused, project-specific context.

## Big Picture
- FastAPI app (`app/main.py`) serves `POST /process`, `GET /health`, `GET /version`, and index; run via Gunicorn+Uvicorn (`Dockerfile`, `app/startup.txt`).
- Flow: upload image → `pipeline.run_pipeline(bytes)` → overlay PNG, CSV (metrics), STL → upload to Azure Blob → return public URLs. Optional proxy `GET /artifact/{name}` when `PUBLIC_ARTIFACT_PROXY=1`.
- Segmentation backends: Roboflow Hosted (`ROBOFLOW_*`) or local ONNX YOLO (`USE_ONNX=1`, `ONNX_MODEL_PATH`, default `model/best.onnx`).
- Scaling to mm: ArUco 4×4 (DICT_4X4_50) 20 mm marker (`scale_aruco.py`) → `mm_per_px`; fallback to `0.25 mm/px` and low confidence.

## Endpoints & Contracts
- `POST /process` (multipart image ≤ 5 MB): returns `{ overlay_url, csv_url, stl_url, backend, scale_confidence, metrics }`.
- `GET /version`: includes short SHA-256 of ONNX model when present. `GET /health`. `GET /`: Jinja2 index at `app/templates/index.html` with static in `app/static/`.

## Environments & Secrets
- Required: `AZURE_STORAGE_CONNECTION_STRING` for blob uploads; optional `BLOB_CONTAINER` (default `im2fit-outputs`).
- Backend choice: Hosted → `ROBOFLOW_API_KEY`, `ROBOFLOW_INFER_URL`; Local ONNX → `USE_ONNX=1`, `ONNX_MODEL_PATH`.
- Optional: `PUBLIC_ARTIFACT_PROXY=1` to enable `/artifact/{name}`. Dev-only: Kaggle (`KAGGLE_USERNAME`, `KAGGLE_KEY`) for dataset/scripts.

## Dev Workflows
- Install deps: `make setup` (uses `app/requirements.txt`).
- Run API (dev): `uvicorn app.main:app --reload --port 8000` or `make dev`.
- Quick client: `python demo_local.py <image.png>` (starts server if needed, posts to `/process`).
- Tests: `pytest -q` (see `pytest.ini`; `pythonpath=app`).

## Code Structure & Patterns
- `main.py`: Routes, CORS, blob upload (`upload_bytes`), import-time guard for `AZURE_STORAGE_CONNECTION_STRING`, optional `/artifact/{name}` proxy.
- `pipeline.py`: Orchestrates segmentation (hosted vs ONNX), scaling, measurements, overlay, STL, CSV. Pure functions over bytes/arrays; returns bytes + metrics dict.
- `onnx_infer.py`: Lazy `onnxruntime` (CPUExecutionProvider). 640×640 RGB preproc; heuristic YOLO-seg mask reconstruction (assumes 32/64 coeffs).
- `scale_aruco.py`: OpenCV contrib ArUco DICT_4X4_50 → `(mm_per_px, confidence)`; `None, 0.0` if not found.
- `overlay.py`: Draw contour + metric text → PNG bytes. `to_3d.py`: densify polygon; simple prism + parabolic crown → STL via `trimesh`.

## Data Contracts & Metrics
- `run_pipeline(image_bytes)` → `{ overlay_png: bytes, csv_bytes: bytes, stl_bytes: bytes, metrics: dict }`.
- `metrics`: `length_mm`, `width_prox_mm`, `width_mid_mm`, `width_dist_mm`, `mm_per_px`, `scale_confidence`, `mask_area_px`, `sharpness`, `axis_origin`, `axis_vec`. CSV includes only scalar metrics.

## Testing Guidance
- See `tests/test_pipeline.py`: monkeypatch `app.onnx_infer.infer_mask` and `app.scale_aruco.mm_per_pixel_from_aruco` to avoid heavy/external deps.
- For the import-time blob guard, import `app.pipeline` directly or set a dummy `AZURE_STORAGE_CONNECTION_STRING` in test env.

## Gotchas & Tips
- Import-time failure if `AZURE_STORAGE_CONNECTION_STRING` missing; affects `uvicorn` and `demo_local.py`.
- Env name mismatch: infra/azd use `STORAGE_CONTAINER`; app uses `BLOB_CONTAINER` (default `im2fit-outputs`). Prefer `BLOB_CONTAINER` at runtime.
- `/process` enforces `image/*` content type and ≤ 5 MB. ArUco detection needs `opencv-contrib-python` (included).

## Azure Deploy Notes
- This project is demo-only; use non-production settings. `APP_ENV=demo` is set in `azure.yaml` and `infra/main.bicep`.
- `azure.yaml` + `infra/main.bicep` provision ACR, Storage, App Service, Insights, roles. App still expects a blob connection string—configure `AZURE_STORAGE_CONNECTION_STRING` even if MSI/roles are set.
- Env mismatch reminder: infra uses `STORAGE_CONTAINER`, app uses `BLOB_CONTAINER` (now explicitly set to `im2fit-outputs`).
- Container image served by App Service; startup follows `app/startup.txt`/`Dockerfile`. Healthcheck hits `/health`.
