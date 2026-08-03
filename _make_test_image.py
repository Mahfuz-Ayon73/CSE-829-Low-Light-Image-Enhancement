import cv2
import numpy as np

rng = np.random.default_rng(0)

img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.rectangle(img, (0, 0), (600, 400), (60, 90, 120), -1)
cv2.circle(img, (150, 150), 80, (30, 200, 220), -1)
cv2.rectangle(img, (300, 200), (550, 380), (200, 150, 40), -1)
for i in range(0, 600, 20):
    cv2.line(img, (i, 0), (i, 400), (255, 255, 255), 1)
cv2.putText(img, "TEST", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)

low = img.astype(np.float64) * 0.18
noise = rng.normal(0, 6, low.shape)
low = np.clip(low + noise, 0, 255).astype(np.uint8)

cv2.imwrite("test_bright.png", img)
cv2.imwrite("test_lowlight.png", low)
print("wrote test_bright.png, test_lowlight.png")
print("lowlight mean/std:", low.mean(), low.std())
