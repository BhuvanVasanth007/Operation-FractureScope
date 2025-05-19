import os, cv2, numpy as np, onnxruntime as ort

sess = ort.InferenceSession("weights/final_synthetic_defect_q.onnx",
                            providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def preprocess(path):
    img = cv2.imread(path); img = cv2.resize(img,(512,512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    rgb3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    multi = np.concatenate([img, rgb3, thermal], axis=2).astype(np.float32)/255.0
    return np.transpose(multi,(2,0,1))[None,...]

if __name__ == "__main__":
    folder = "data/images/val"
    for f in os.listdir(folder):
        if not f.lower().endswith((".jpg",".png")): continue
        inp_tensor = preprocess(f"{folder}/{f}")
        outs = sess.run(None, {inp: inp_tensor})[0]
        # … print or visualize …
        print(f, "→", len(outs), "detections")
