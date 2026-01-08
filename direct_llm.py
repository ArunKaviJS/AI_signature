import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import os
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()


SYSTEM_PROMPT = """
You are a forensic document and cheque examination expert.

You specialize in:
- Bank cheque background authenticity
- Ink color and ink type analysis (blue pen vs digital ink)
- Detection of pasted or overlaid signatures
- Signature reuse and simulation detection
- Document tampering and compositing artifacts

Image-1 is a SUSPECT document.
Image-2 is a KNOWN ORIGINAL reference.

Base all conclusions strictly on visible evidence.
If evidence is insufficient, state Inconclusive.
"""

USER_PROMPT = """
Analyze Image-1 (suspect cheque document) and Image-2 (original reference).

Step 1: Cheque Background & Ink Check (Image-1 only)
- Determine whether the background resembles a professional bank cheque
- Look for micro-patterns, guilloche lines, security printing
- Check whether the signature ink appears blue pen ink or digitally inserted
- Identify layering, edge halos, uniform color, or resolution mismatch

Step 2: Signature Comparison
- Compare the Image-1 signature with Image-2 (original)
- Evaluate stroke flow, slant, proportions, pen lifts
- Detect signs of reuse, tracing, or copy-paste

Step 3: Fraud & Tampering Assessment
- Identify any cut-paste, overlay, or digital manipulation indicators
- Assess stamp or seal misuse if present

Return results in the following EXACT format:

Cheque & Signature Fraud Assessment:

1. Cheque Background Authenticity (Image-1):
   - Assessment:
   - Confidence (%):
   - Evidence:

2. Ink Analysis (Image-1):
   - Ink Type (Blue Pen / Printed / Digital Overlay):
   - Confidence (%):
   - Evidence:

3. Signature Match with Original (Image-2):
   - Match Percentage (%):
   - Confidence (%):
   - Evidence:

4. Fraud Flags:
   - Flag Name:
     - Risk Level:
     - Confidence:
     - Evidence:

5. Final Verdict:
   - Likely Authentic / Likely Fraudulent / Inconclusive
   - Overall Confidence (%)
"""




client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def pil_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pdf_to_pil_images(pdf_path, dpi=300):
    """
    Convert PDF pages to PIL Images using fitz
    """
    doc = fitz.open(pdf_path)
    images = []

    zoom = dpi / 72  # PDF default DPI is 72
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)

        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(pil_img)

    return images



def compare_images_with_llm(image1_b64, image2_b64):
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": USER_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image1_b64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image2_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=900
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    pdf1 = "signwithdocument.pdf"
    pdf2 = "oeigin doopp.pdf"

    # Convert PDFs to images (taking first page for simplicity)
    img1 = pdf_to_pil_images(pdf1)[0]
    img2 = pdf_to_pil_images(pdf2)[0]

    # Convert to Base64
    img1_b64 = pil_to_base64(img1)
    img2_b64 = pil_to_base64(img2)

    # Compare via LLM
    result = compare_images_with_llm(img1_b64, img2_b64)
    print("\n--- Signature Comparison Result ---")
    print(result)
