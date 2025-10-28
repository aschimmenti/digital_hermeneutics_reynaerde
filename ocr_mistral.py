from mistralai import Mistral
import os
import json

# Configure client
api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# Input PDF
pdf_path = "kenneth-2000.pdf"
base_name = os.path.splitext(os.path.basename(pdf_path))[0]
md_out = f"{base_name}.md"
json_out = f"{base_name}.ocr.json"

# Upload file to Mistral, then reference by file_id
uploaded = client.files.upload(
    file={
        "file_name": os.path.basename(pdf_path),
        "content": open(pdf_path, "rb"),
    },
    purpose="ocr",
)

# Run OCR using uploaded file reference
ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "file",
        "file_id": uploaded.id,
    },
    include_image_base64=False,
)

# Persist raw response for debugging/inspection
try:
    # Try to convert SDK object to dict if supported
    if hasattr(ocr_response, "to_dict"):
        raw = ocr_response.to_dict()
    elif isinstance(ocr_response, dict):
        raw = ocr_response
    else:
        # Fallback best-effort serialization
        raw = json.loads(json.dumps(ocr_response, default=lambda o: getattr(o, "__dict__", str(o))))
except Exception:
    raw = {"raw": str(ocr_response)}

with open(json_out, "w", encoding="utf-8") as f:
    json.dump(raw, f, ensure_ascii=False, indent=2)

# Extract markdown from likely fields
markdown_text = None

# Common shapes to try in order
if isinstance(raw, dict):
    if "markdown" in raw and isinstance(raw["markdown"], str):
        markdown_text = raw["markdown"]
    elif "content" in raw and isinstance(raw["content"], str):
        markdown_text = raw["content"]
    elif "pages" in raw and isinstance(raw["pages"], list):
        page_md = []
        for p in raw["pages"]:
            if isinstance(p, dict) and "markdown" in p and isinstance(p["markdown"], str):
                page_md.append(p["markdown"])
        if page_md:
            markdown_text = "\n\n".join(page_md)

# Final fallback to stringified response
if markdown_text is None:
    markdown_text = str(ocr_response)

with open(md_out, "w", encoding="utf-8") as f:
    f.write(markdown_text)

print(f"Saved OCR markdown to: {md_out}")
print(f"Saved raw OCR JSON to: {json_out}")