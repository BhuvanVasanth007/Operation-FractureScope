# Operation FractureScope

**Industrial Anomaly Detection in Harsh Environments**  
EdgeX AI Challenge – Bridge inspection cracks, corrosion & leak detection

---

## 📂 Repository Structure


---

## 🚀 Quickstart

1. **Create & activate** a Python venv  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate.bat   # Windows
2. **Install deps**
pip install -r requirements.txt
3. **Regenerate full synthetic data**
python src/generate_synthetic_data.py \
  --out_dir data/full_images \
  --train_count 1000 \
  --val_count 200
4. **Train YOLOv8**
python src/train_yolo.py \
  --data data/full_images/data.yaml \
  --epochs 50 \
  --imgsz 512 \
  --batch 16
5. **Export & quantize**
python src/export_and_quantize.py
6. **Benchmark CPU‐only latency**
python src/benchmark.py
7. **Visualize labels & preds**
python src/visualize.py
8. **Edge demo**
python deploy/edge_inference.py \
  --source data/sample_images \
  --model weights/final_synthetic_defect_q.onnx
9. **Secure logging scaffold** 
python deploy/device_logger.py

