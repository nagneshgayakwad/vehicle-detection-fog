import cv2
import numpy as np
from skimage import img_as_float

def dark_channel(img, window_size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    return cv2.erode(min_channel, kernel)

def estimate_atmospheric_light(img, dark):
    h, w = dark.shape
    flat_dark = dark.ravel()
    flat_img = img.reshape(h * w, 3)
    indices = flat_dark.argsort()[-max(int(h*w*0.001),1):]
    return np.max(flat_img[indices], axis=0)

def estimate_transmission(img, A):
    normed = img / A
    return np.clip(1 - 0.95 * dark_channel(normed), 0.1, 1)

def recover_image(img, A, t):
    J = np.empty_like(img)
    for c in range(3):
        J[..., c] = (img[..., c] - A[c]) / t + A[c]
    return np.clip(J, 0, 1)

def enhance_array(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_as_float(img_rgb)

    dark = dark_channel(img_float)
    A = estimate_atmospheric_light(img_float, dark)
    t = estimate_transmission(img_float, A)

    dcp = recover_image(img_float, A, t)
    dcp = (dcp * 255).astype(np.uint8)

    return cv2.cvtColor(dcp, cv2.COLOR_RGB2BGR)