import fitz  # PyMuPDF
from PIL import Image
import io
import cv2
import numpy as np
import base64
import os
import json
from openai import AzureOpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SIGNATURE_ANALYSIS_PROMPT = """
You are a forensic handwriting analyst.

Analyze the handwritten signature image and describe:
- Stroke continuity (smooth / broken)
- Curve style and loops
- Slant direction
- Line thickness and pressure
- Unique identifying features

Keep the response under 120 words.
Do NOT guess the person's name.
"""



openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def analyze_signature_with_gpt4o(signature_b64: str) -> str:
    """
    signature_b64: base64 encoded PNG/JPG (NO data:image prefix)
    returns: descriptive text
    """

    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {
                "role": "system",
                "content": "You are an expert forensic handwriting analyst."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SIGNATURE_ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{signature_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# Embedding Config
AZURE_EMBED_API_KEY = os.getenv("AZURE_EMBED_API_KEY")
AZURE_EMBED_ENDPOINT = os.getenv("AZURE_EMBED_ENDPOINT")
AZURE_EMBED_API_VERSION = os.getenv("AZURE_EMBED_API_VERSION")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT") 

client = AzureOpenAI(
    api_key=AZURE_EMBED_API_KEY,
    api_version=AZURE_EMBED_API_VERSION,
    azure_endpoint=AZURE_EMBED_ENDPOINT
)

def get_embedding(text: str):
    response = client.embeddings.create(
        model=AZURE_EMBED_DEPLOYMENT,
        input=text
    )
    return np.array(response.data[0].embedding)

def pdf_to_images_fitz(pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)

    zoom = dpi / 72  # 72 DPI is default
    mat = fitz.Matrix(zoom, zoom)

    images = []

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    return images

def preprocess_for_signature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (best for scanned docs)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )

    return thresh

def find_signature_bbox(thresh):
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter small noise
    contours = [c for c in contours if cv2.contourArea(c) > 500]

    if not contours:
        return None

    # Take largest contour (usually signature)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    return x, y, w, h


def crop_signature(img):
    thresh = preprocess_for_signature(img)
    bbox = find_signature_bbox(thresh)

    if bbox is None:
        raise ValueError("No signature detected")

    x, y, w, h = bbox

    # Add padding
    pad = 10
    h_img, w_img = img.shape[:2]

    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, w_img)
    y2 = min(y + h + pad, h_img)

    return img[y1:y2, x1:x2]

def opencv_to_base64(cv_img, fmt=".png") -> str:
    success, buffer = cv2.imencode(fmt, cv_img)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")

def pil_to_opencv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
    
# arun sign -------------
images = pdf_to_images_fitz("sign2.pdf", dpi=300)
images[0].save("page_1.png", "PNG")



cv_img = pil_to_opencv(images[0])
signature = crop_signature(cv_img)
cv2.imwrite("signature.png", signature)



signature_b64 = opencv_to_base64(signature)

arun=signature_b64  # preview


#--------------------siva sign

images1 = pdf_to_images_fitz("siva sign.pdf", dpi=300)
images1[0].save("page_1.png", "PNG")



cv_img1 = pil_to_opencv(images1[0])
signature1 = crop_signature(cv_img1)
cv2.imwrite("signature.png", signature1)



signature_b641 = opencv_to_base64(signature1)

siva=signature_b641  # preview
#-------------------------------------------------------

sig1_text = analyze_signature_with_gpt4o(signature_b64)
sig2_text = analyze_signature_with_gpt4o(signature_b641)

print("Signature 1 analysis:\n", sig1_text)
print("Signature 2 analysis:\n", sig2_text)

emb1 = get_embedding(sig1_text)
emb2 = get_embedding(sig2_text)

similarity = cosine_similarity(emb1, emb2)

print(similarity)
