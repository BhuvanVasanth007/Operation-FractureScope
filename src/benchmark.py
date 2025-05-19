import os, glob, random, cv2, numpy as np, time, onnxruntime as ort

# simulate 4 cores
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[v] = "4"

sess = ort.InferenceSession("weights/final_synthetic_defect.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

paths = glob.glob("data/images/val/*.jpg")
img = cv2.resize(cv2.imread(random.choice(paths)), (512,512)).astype(np.float32)/255.0
tensor = np.transpose(img, (2,0,1))[None,...]

# warm-up
for _ in range(5):
    sess.run(None, {inp: tensor})

# timing
runs=50; t0=time.time()
for _ in range(runs):
    sess.run(None, {inp: tensor})
print(f"Avg FP32 ONNX latency: {(time.time()-t0)/runs*1000:.1f} ms")
