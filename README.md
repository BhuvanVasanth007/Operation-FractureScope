# Operation-FractureScope

## Structure
- src/      Python scripts  
- data/     sample_images/ (20 images + labels)  
- report/   Combined_Report.tex (+ .pdf)  
- deploy/   edge_inference.py, device_logger.py  

## Quickstart
```bash
pip install -r requirements.txt
python src/generate_synthetic_data.py
python src/train_yolo.py
python src/export_and_quantize.py
python src/benchmark.py
python deploy/edge_inference.py
python deploy/device_logger.py
