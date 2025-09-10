PYTHON=python
PIP=python -m pip

setup:
	$(PIP) install --upgrade pip && pip install -r app/requirements.txt

kaggle:
	@echo "Creating .kaggle directory if it doesn't exist..."
	@mkdir -p .kaggle
	@echo "Generating kaggle.json from env vars"
	@echo '{"username":"'$(KAGGLE_USERNAME)'","key":"'$(KAGGLE_KEY)'"}' > .kaggle/kaggle.json
	$(PYTHON) -c "import os, json; print('Wrote .kaggle/kaggle.json')"
	kaggle datasets download -d muhammadhammad261/nail-segmentation-dataset -p data/raw --unzip

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

.PHONY: train
train:
	@echo --- Training YOLO11-m-seg on Roboflow dataset ---
	yolo segment train model=yolo11m-seg.pt data="data/roboflow/data.yaml" imgsz=640 epochs=100 batch=16 patience=25 device=0 name=im2fit_rf_y11m

.PHONY: val
val:
	yolo segment val model="runs/segment/im2fit_rf_y11m/weights/best.pt" data="data/roboflow/data.yaml"

.PHONY: export-onnx
export-onnx:
	yolo export model="runs/segment/im2fit_rf_y11m/weights/best.pt" format=onnx dynamic=True simplify=True opset=13
	if not exist model mkdir model
	copy runs\segment\im2fit_rf_y11m\weights\best.onnx model\best.onnx

.PHONY: phase0
phase0:
	@echo --- Generating Phase-0 artifacts from handsy_in to handsy_out ---
	python scripts\handsy_phase0.py
