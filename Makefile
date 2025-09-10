PYTHON=python
PIP=$(PYTHON) -m pip
RUN_NAME?=im2fit_rf_y11m_final

.PHONY: setup
setup:
	@echo --- Ensuring base dependencies ---
	$(PIP) install --upgrade pip
	$(PIP) install -r app/requirements.txt
	$(PIP) install -U ultralytics

.PHONY: dev
dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

.PHONY: test
test:
	pytest -q

.PHONY: train
train:
	@echo --- Optimized training run: $(RUN_NAME) ---
	yolo segment train model=yolo11m-seg.pt data="data/roboflow/data.yaml" imgsz=640 epochs=120 patience=25 batch=-1 device=0 workers=8 cache=disk cos_lr=True close_mosaic=20 name=$(RUN_NAME) project="runs/segment"

.PHONY: val
val:
	yolo segment val model="runs/segment/$(RUN_NAME)/weights/best.pt" data="data/roboflow/data.yaml"

.PHONY: export-onnx
export-onnx:
	yolo export model="runs/segment/$(RUN_NAME)/weights/best.pt" format=onnx dynamic=True simplify=True opset=14
	if not exist model mkdir model
	copy runs\segment\$(RUN_NAME)\weights\best.onnx model\best.onnx /Y

.PHONY: evaluate
evaluate:
	$(PYTHON) scripts\run_handsy_evaluation.py --run-name $(RUN_NAME)

.PHONY: methods-note
methods-note:
	$(PYTHON) scripts\write_methods_note.py --run-name $(RUN_NAME)

.PHONY: phase0
phase0:
	$(PYTHON) scripts\handsy_phase0.py

.PHONY: all
all: setup train val export-onnx evaluate

.PHONY: monitor
monitor:
	$(PYTHON) scripts\monitor_training.py --run-name $(RUN_NAME) --once
