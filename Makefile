.PHONY: all setup train validate export-onnx run-evaluation qc

# Use PowerShell's ability to resolve absolute paths for robustness
# This ensures 'make' can be run from any directory within the project
DATA_YAML = $(shell pwsh -Command "(Resolve-Path data/roboflow/data.yaml).Path")
RUN_NAME = im2fit_final_seg

# Run everything from setup to final deliverable and quality check
all: setup train validate export-onnx run-evaluation qc

# 1. Install/verify all necessary Python dependencies with GPU support
setup:
	@echo "--- Upgrading Dependencies & Verifying Environment ---"
	pip install -U ultralytics
	pip install onnxruntime-gpu opencv-contrib-python pandas "transformers[torch]"

# 2. Train the robust YOLOv11-m-seg model with the final optimized recipe
train:
	@echo "--- Starting Final Optimized Training Run: $(RUN_NAME) ---"
	yolo segment train model=yolov11m-seg.pt data="$(DATA_YAML)" ^
	imgsz=640 epochs=40 patience=10 batch=-1 device=0 workers=10 ^
	cache=disk optimizer=AdamW cos_lr=True close_mosaic=15 deterministic=False ^
	mixup=0.10 erasing=0.20 seed=42 plots=True name=$(RUN_NAME) project="runs/segment"

# 3. Validate the trained model's performance on both val and test sets
validate:
	@echo "--- Validating model on 'val' and 'test' splits ---"
	yolo segment val model="runs/segment/$(RUN_NAME)/weights/best.pt" data="$(DATA_YAML)" split=val retina_masks=True
	yolo segment val model="runs/segment/$(RUN_NAME)/weights/best.pt" data="$(DATA_YAML)" split=test retina_masks=True

# 4. Export the final model to ONNX and perform verification
export-onnx:
	@echo "--- Exporting model to ONNX and Verifying CUDA Provider ---"
	yolo export model="runs/segment/$(RUN_NAME)/weights/best.pt" format=onnx dynamic=True simplify=True opset=14
	if not exist model mkdir model
	copy "runs\segment\$(RUN_NAME)\weights\best.onnx" "model\best.onnx" /Y
	python -c "from hashlib import sha256; import onnxruntime as ort; p='model/best.onnx'; print(f'ONNX_SHA256= {sha256(open(p,\"rb\").read()).hexdigest()}'); sess = ort.InferenceSession(p, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); print(f'Providers: {sess.get_providers()}')"

# 5. Generate the final client deliverables
run-evaluation:
	@echo "--- Generating client artifacts in 'handsy_out' folder ---"
	python scripts/run_handsy_evaluation.py
	python scripts/write_methods_note.py --run-name $(RUN_NAME)

# 6. Perform final automated Quality Control checks
qc:
	@echo "--- Performing Final Quality Control Checks ---"
	python -c "import os; assert len(list((p for p in os.listdir('handsy_out') if p.endswith('_mask.png')))) == 5, 'QC FAILED: Expected 5 mask files.'; print('✅ QC PASSED: 5 mask files found.')"
	python -c "import os; assert len(list((p for p in os.listdir('handsy_out') if p.endswith('_overlay.png')))) == 5, 'QC FAILED: Expected 5 overlay files.'; print('✅ QC PASSED: 5 overlay files found.')"
	python -c "import pandas as pd; df = pd.read_csv('handsy_out/measurements.csv'); assert len(df) >= 20, f'QC FAILED: Expected >= 20 rows in CSV, found {len(df)}.'; print(f'✅ QC PASSED: {len(df)} rows found in measurements.csv.')"
	python -c "import pandas as pd; df = pd.read_csv('handsy_out/measurements.csv'); cols = ['image', 'nail_index', 'length_mm', 'width_prox_mm', 'scale_method', 'scale_confidence']; assert all(c in df.columns for c in cols), f'QC FAILED: Missing one or more required columns in {cols}.'; print('✅ QC PASSED: All required CSV columns are present.')"