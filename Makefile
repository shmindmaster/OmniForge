PYTHON=python
PIP=python -m pip

setup:
	$(PIP) install -r app/requirements.txt

kaggle:
	@if not exist .kaggle mkdir .kaggle
	@echo Generating kaggle.json from env vars
	@echo {{"username":"$${KAGGLE_USERNAME}","key":"$${KAGGLE_KEY}"}}> .kaggle/kaggle.json
	$(PYTHON) -c "import os, json; print('Wrote .kaggle/kaggle.json')"
	kaggle datasets download -d muhammadhammad261/nail-segmentation-dataset -p data/raw --unzip

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q
