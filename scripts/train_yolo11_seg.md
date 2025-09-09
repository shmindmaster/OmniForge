## YOLO11 Segmentation Training (Small Dataset)

Config file: `ultralytics.yaml`

Contents:

```
model: yolo11n-seg.pt
data: data/yolo/data.yaml
epochs: 50
imgsz: 640
batch: 16
patience: 10
save_period: 5
```

### Commands

```bash
# Train
yolo task=segment mode=train model=yolo11n-seg.pt data="data/yolo/data.yaml" epochs=50 imgsz=640 batch=16 device=cpu

# Validate
yolo mode=val task=segment model=runs/segment/train/weights/best.pt data="data/yolo/data.yaml"

# Export ONNX
yolo mode=export task=segment model=runs/segment/train/weights/best.pt format=onnx opset=12 dynamic=True
```

After export, move ONNX:

```bash
mkdir -p model
copy runs/segment/train/weights/best.onnx model/best.onnx  # Windows
```

### GPU Switch

If you have a GPU: append `device=0` to the train command. For multi-GPU, set `device=0,1`.

### Notes

- `patience=10` provides early stopping.
- `save_period=5` keeps checkpoints every 5 epochs.
- Adjust `batch` based on memory; on CPU you may need `batch=8`.
