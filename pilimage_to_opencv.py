import cv2
import numpy as np

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def crop_signature_opencv(pil_image, bbox):
    """
    bbox = { "x": int, "y": int, "width": int, "height": int }
    """
    img = pil_to_cv2(pil_image)

    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["width"])
    h = int(bbox["height"])

    cropped = img[y:y+h, x:x+w]

    return cropped

def denormalize_bbox(bbox, img_width, img_height):
    return {
        "x": int(bbox["x"] * img_width),
        "y": int(bbox["y"] * img_height),
        "width": int(bbox["width"] * img_width),
        "height": int(bbox["height"] * img_height)
    }


def safe_crop(img, x, y, w, h):
    H, W, _ = img.shape

    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)

    return img[y:y+h, x:x+w]
