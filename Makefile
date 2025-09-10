PYTHON=python
PIP=$(PYTHON) -m pip

# Definitive Phase-0 backbone (locked to YOLOv11 as per final plan)
MODEL?=yolo11m-seg.pt

RUN_NAME?=im2fit_final_seg

.PHONY: setup
setup:
	@echo --- Hardening environment & upgrading core deps ---
	$(PIP) install --upgrade pip
	$(PIP) install -U ultralytics
	$(PIP) install onnxruntime-gpu "transformers[torch]" opencv-contrib-python pandas
	$(PIP) install -r app/requirements.txt

.PHONY: dev
dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

.PHONY: test
test:
	pytest -q

.PHONY: train
train:
	@echo --- Enabling TF32 + cuDNN benchmark (set via inline Python) ---
	$(PYTHON) - <<"PY"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
print("TF32 enabled:", torch.backends.cuda.matmul.allow_tf32)
PY
	@echo --- Training YOLOv11 segmentation (lean 40-epoch recipe) $(RUN_NAME) ---
	yolo segment train model=$(MODEL) data="$(shell $(PYTHON) -c "from pathlib import Path;print(Path('data/roboflow/data.yaml').resolve())")" imgsz=640 \
	  epochs=40 patience=10 batch=-1 device=0 workers=10 cache=disk \
	  optimizer=AdamW cos_lr=True close_mosaic=15 deterministic=False \
	  mixup=0.10 erasing=0.20 seed=42 plots=True name=$(RUN_NAME) project="runs/segment"

.PHONY: validate val
validate val:
	yolo segment val model="runs/segment/$(RUN_NAME)/weights/best.pt" data="$(shell $(PYTHON) -c "from pathlib import Path;print(Path('data/roboflow/data.yaml').resolve())")" retina_masks=True
	yolo segment val model="runs/segment/$(RUN_NAME)/weights/best.pt" data="$(shell $(PYTHON) -c "from pathlib import Path;print(Path('data/roboflow/data.yaml').resolve())")" split=test retina_masks=True

.PHONY: fine_tune_768
fine_tune_768:
	@echo --- Optional 768px fine-tune ---
	yolo segment train model="runs/segment/$(RUN_NAME)/weights/best.pt" data="data/roboflow/data.yaml" imgsz=768 \
	  epochs=8 patience=3 batch=-1 device=0 workers=6 cache=disk optimizer=AdamW cos_lr=True \
	  mosaic=0 mixup=0.05 erasing=0.10 close_mosaic=0 deterministic=False \
	  name=$(RUN_NAME)_768 project="runs/segment"

.PHONY: export-onnx export
export-onnx export:
	@echo --- Exporting ONNX (dynamic, simplify, opset=14) ---
	yolo export model="runs/segment/$(RUN_NAME)/weights/best.pt" format=onnx dynamic=True simplify=True opset=14
	if not exist model mkdir model
	copy runs\segment\$(RUN_NAME)\weights\best.onnx model\best.onnx /Y
	@echo --- Computing SHA256 + provider check ---
	$(PYTHON) - <<"PY"
from hashlib import sha256
import onnxruntime as ort, json
import os
p='model/best.onnx'
h=sha256(open(p,'rb').read()).hexdigest()
sess=ort.InferenceSession(p, providers=[('CUDAExecutionProvider',{'device_id':0}), 'CPUExecutionProvider'])
print('ONNX_SHA256='+h)
print('Providers='+json.dumps(sess.get_providers()))
open('model/onnx_sha256.txt','w').write(h+'\n')
PY

.PHONY: run-evaluation evaluate
run-evaluation evaluate:
	@echo --- Generating masks, overlays, CSV (scale_method included) ---
	set USE_ID1_FALLBACK=1 && $(PYTHON) scripts\run_handsy_evaluation.py --run-name $(RUN_NAME)
	@echo --- Quick QC: expecting 5 mask + 5 overlay files ---
	$(PYTHON) - <<"PY"
from pathlib import Path, PurePath
import sys, csv
out=Path('handsy_out')
masks=list(out.glob('*_mask.png'))
over=list(out.glob('*_overlay.png'))
assert len(masks)==5, f"Expected 5 mask images, got {len(masks)}"
assert len(over)==5, f"Expected 5 overlay images, got {len(over)}"
csv_path=out/'measurements.csv'
rows=list(csv.reader(open(csv_path,'r',encoding='utf-8')))
header=rows[0]
needed=['image','nail_index','length_mm','width_prox_mm','width_mid_mm','width_dist_mm','mask_area_px','mm_per_px','scale_confidence','scale_method']
missing=[c for c in needed if c not in header]
assert not missing, f"Missing columns: {missing}"
data_rows=rows[1:]
assert len(data_rows)>=20, f"Expected >=20 data rows, got {len(data_rows)}"
print('QC OK: artifacts present and CSV schema valid.')
PY
	@echo --- Writing methods note ---
	$(PYTHON) scripts\write_methods_note.py --run-name $(RUN_NAME)

.PHONY: methods-note
methods-note:
	$(PYTHON) scripts\write_methods_note.py --run-name $(RUN_NAME)

.PHONY: phase0
phase0:
	$(PYTHON) scripts\handsy_phase0.py

.PHONY: all
all: setup train validate export-onnx run-evaluation

.PHONY: monitor
monitor:
	$(PYTHON) scripts\monitor_training.py --run-name $(RUN_NAME) --once
