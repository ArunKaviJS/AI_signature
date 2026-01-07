import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import os
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

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
    """
    Sends two images to GPT-4o (vision) and asks for:
    1. Whether the signatures match
    2. A match percentage (0-100%)
    3. Short reasoning
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document verification assistant specialized in signature analysis."
                    " Your task is to compare two signatures and provide a match assessment."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Compare these two signatures and provide a match percentage between 0% (completely different) "
                            "and 100% (identical). Give a short explanation of your reasoning."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image1_b64}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image2_b64}"}
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    pdf1 = "priya1.pdf"
    pdf2 = "priya2.pdf"

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
