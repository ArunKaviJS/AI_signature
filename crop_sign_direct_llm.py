import os
import io
import base64
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from openai import AzureOpenAI

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# --------------------------------------------------
# PDF → PIL Image
# --------------------------------------------------
def pdf_to_pil_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    images = []

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)

    return images

# --------------------------------------------------
# Detect handwritten / signature region
# --------------------------------------------------
def detect_signature_region(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold works better than OTSU for documents
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 15
    )

    # Remove horizontal/vertical lines (tables)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape
    candidates = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect_ratio = cw / max(ch, 1)

        roi = gray[y:y+ch, x:x+cw]
        ink_ratio = np.sum(roi < 200) / max(area, 1)

        if (
            400 < area < 25000 and           # not too big (tables)
            cw > 120 and ch < 150 and        # signature-like
            aspect_ratio > 3 and
            ink_ratio < 0.35 and             # sparse ink
            y > h * 0.55                     # near bottom
        ):
            candidates.append((x, y, cw, ch))

    if not candidates:
        return None

    # Pick widest region (signatures are wide)
    return max(candidates, key=lambda b: b[2])

# --------------------------------------------------
# Crop signature
# --------------------------------------------------
def crop_signature(pil_img, bbox, padding=20):
    img = np.array(pil_img)
    x, y, w, h = bbox

    h_img, w_img, _ = img.shape

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    cropped = img[y1:y2, x1:x2]
    return Image.fromarray(cropped)

# --------------------------------------------------
# PIL → Base64
# --------------------------------------------------
def pil_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# --------------------------------------------------
# Compare two signatures with GPT-4o
# --------------------------------------------------
def compare_signatures_with_llm(sig1_b64, sig2_b64):
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a forensic document examiner specialized in handwritten signature verification."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Compare the following two handwritten signatures.\n"
                            "Return:\n"
                            "1. Match Percentage (0–100%)\n"
                            "2. Verdict: MATCH or NOT MATCH\n"
                            "3. Short technical explanation\n\n"
                            "Base your judgment on stroke style, shape, flow, slant, and consistency."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sig1_b64}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sig2_b64}"}
                    }
                ]
            }
        ],
        max_tokens=400
    )

    return response.choices[0].message.content

# --------------------------------------------------
# Extract signature from PDF
# --------------------------------------------------
def extract_signature_from_pdf(pdf_path):
    images = pdf_to_pil_images(pdf_path)

    for page_img in images:
        bbox = detect_signature_region(page_img)
        if bbox:
            return crop_signature(page_img, bbox)

    return None

def save_signature_image(signature_img, output_path):
    if signature_img is None:
        print(f"❌ No signature to save: {output_path}")
        return
    signature_img.save(output_path, format="PNG")
    print(f"✅ Saved signature: {output_path}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    pdf1 = "signwithdocument.pdf"
    pdf2 = "sign2.pdf"

    sig1 = extract_signature_from_pdf(pdf1)
    sig2 = extract_signature_from_pdf(pdf2)
    
    save_signature_image(sig1, "signature_1.png")
    save_signature_image(sig2, "signature_2.png")

    if sig1 is None or sig2 is None:
        print("❌ Signature not found in one or both PDFs")
        exit()

    sig1_b64 = pil_to_base64(sig1)
    sig2_b64 = pil_to_base64(sig2)

    result = compare_signatures_with_llm(sig1_b64, sig2_b64)

    print("\n===== SIGNATURE COMPARISON RESULT =====")
    print(result)
