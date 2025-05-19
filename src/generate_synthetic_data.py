import cv2
import numpy as np
import random
import os

IMG_SIZE = 512
CLASSES = {0: "crack", 1: "corrosion", 2: "leak"}
NUM_IMAGES = 200
TRAIN_SPLIT = 0.8

def create_background():
    bg = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    noise = np.random.normal(loc=128, scale=25, size=bg.shape).astype(np.uint8)
    blur = cv2.GaussianBlur(noise, (5, 5), 0)
    return blur

def draw_crack(img):
    x1, y1 = random.randint(50, 460), random.randint(50, 460)
    length = random.randint(30, 100)
    thickness = 1
    points = [(x1, y1)]
    for _ in range(random.randint(3, 6)):
        x1 += random.randint(-10, 10)
        y1 += random.randint(-10, 10)
        points.append((x1, y1))
    for i in range(len(points)-1):
        cv2.line(img, points[i], points[i+1], (0, 0, 0), thickness)
    x, y, w, h = cv2.boundingRect(np.array(points))
    return x, y, w, h, 0

def draw_corrosion(img):
    x, y = random.randint(50, 400), random.randint(50, 400)
    radius = random.randint(10, 30)
    color = (
        random.randint(80, 120),
        random.randint(30, 50),
        random.randint(0, 20)
    )
    center = (x, y)
    cv2.circle(img, center, radius, color, -1)
    return x-radius, y-radius, 2*radius, 2*radius, 1


def draw_leak(img):
    x, y = random.randint(50, 400), random.randint(50, 400)
    radius = random.randint(15, 30)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    leak_blob = cv2.GaussianBlur(mask, (21, 21), 10)
    leak_blob = cv2.merge([leak_blob]*3)
    img[:] = np.where(leak_blob > 0, img // 2, img)
    return x-radius, y-radius, 2*radius, 2*radius, 2

def draw_random_defect(img):
    defect_type = random.choice([draw_crack, draw_corrosion, draw_leak])
    return defect_type(img)

def save_yolo_format(label_path, boxes):
    with open(label_path, 'w') as f:
        for x, y, w, h, cls in boxes:
            xc = (x + w / 2) / IMG_SIZE
            yc = (y + h / 2) / IMG_SIZE
            wn = w / IMG_SIZE
            hn = h / IMG_SIZE
            f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

base_dir = "/kaggle/working/synthetic_dataset"
for i in range(NUM_IMAGES):
    img = create_background()
    n_defects = random.randint(1, 2)
    boxes = [draw_random_defect(img) for _ in range(n_defects)]

    # Generate grayscale and thermal
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    stacked = np.concatenate([img, gray_rgb, thermal], axis=2)
    name = f"{i:04d}.jpg"
    subset = "train" if i < int(NUM_IMAGES * TRAIN_SPLIT) else "val"

    # Save visible image
    cv2.imwrite(f"{base_dir}/images/{subset}/{name}", img)
    save_yolo_format(f"{base_dir}/labels/{subset}/{name.replace('.jpg', '.txt')}", boxes)

print("✅ Dataset generated:", NUM_IMAGES, "images")


if __name__ == "__main__":
    os.makedirs("data/images/train", exist_ok=True)
    os.makedirs("data/images/val",   exist_ok=True)
    generate_dataset("data/images/train", 1000)
    generate_dataset("data/images/val",   200)
