import json
import cv2

from image_to_llm import image_to_base64, detect_signature_with_azure_gpt4o
from pdf_to_image import pdf_to_images
from pilimage_to_opencv import (
    crop_signature_opencv,
    denormalize_bbox,
)
print(cv2.__version__)

# Load PDF → images
pdf_images = pdf_to_images("arun sign.pdf")
#pdf_images.save("debug_page.png")

def is_normalized_bbox(bbox):
    return all(
        isinstance(bbox.get(k), (int, float)) and 0 <= bbox[k] <= 1
        for k in ["x", "y", "width", "height"]
    )

for page_idx, img in enumerate(pdf_images):
    print(f"\nProcessing page {page_idx + 1}")

    img_b64 = image_to_base64(img)
    llm_result = detect_signature_with_azure_gpt4o(img_b64)

    # Convert LLM JSON string → dict
    result = json.loads(llm_result)

    signatures = result.get("signatures", [])

    if not signatures:
        print("No signatures detected")
        continue

    img_width, img_height = img.size

    for i, sig in enumerate(signatures):
        bbox = sig["bounding_box"]

        print("Raw bbox:", bbox)

        # ✅ Auto-detect normalization
        if is_normalized_bbox(bbox):
            print("Detected normalized bbox → denormalizing")
            bbox = denormalize_bbox(bbox, img_width, img_height)

        cropped = crop_signature_opencv(img, bbox)

        output_name = f"signature_page{page_idx+1}_{i+1}.png"
        cv2.imwrite(output_name, cropped)

        print(f"Saved: {output_name}")
