import os, random, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# 1) Ground-truth preview
def preview_labels(img_dir, lbl_dir, N=5, img_size=512):
    class_names = {0: "Crack", 1: "Corrosion", 2: "Leak"}
    imgs = sorted(glob.glob(f"{img_dir}/*.jpg"))
    lbls = sorted(glob.glob(f"{lbl_dir}/*.txt"))
    for idx in random.sample(range(len(imgs)), N):
        img = cv2.cvtColor(cv2.imread(imgs[idx]), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, figsize=(5,5))
        ax.imshow(img)
        with open(lbls[idx]) as f:
            for line in f:
                cls, xc,yc,w,h = map(float, line.split())
                x = int((xc - w/2)*img_size)
                y = int((yc - h/2)*img_size)
                w = int(w*img_size); h = int(h*img_size)
                rect = patches.Rectangle((x,y),w,h,edgecolor='r',facecolor='none',linewidth=2)
                ax.add_patch(rect)
                ax.text(x,y-5,class_names[int(cls)],color='yellow',weight='bold')
        ax.axis('off'); plt.show()

# 2) Model prediction preview
def preview_preds(model_path, img_dir, N=5):
    model = YOLO(model_path)
    imgs = glob.glob(f"{img_dir}/*.jpg")
    for path in random.sample(imgs, N):
        res = model.predict(path, conf=0.25, imgsz=512)[0]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, figsize=(5,5))
        ax.imshow(img)
        for *box, conf, cls in res.boxes.data.tolist():
            x1,y1,x2,y2 = map(int,box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,
                                     edgecolor='lime',facecolor='none',linewidth=2)
            ax.add_patch(rect)
            ax.text(x1,y1-5,label,color='yellow',weight='bold')
        ax.axis('off'); plt.show()

if __name__ == "__main__":
    preview_labels("data/sample_images", "data/sample_images", N=3, img_size=512)
    preview_preds("weights/final_synthetic_defect.pt", "data/sample_images", N=3)
