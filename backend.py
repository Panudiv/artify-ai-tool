import os
import json
from io import BytesIO
import requests
import cv2 
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, send_from_directory
from rembg import remove, new_session

# --- 1. INITIALIZE THE FLASK APP ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=APP_DIR, static_url_path="")

# --- 2. INITIALIZE THE AI MODEL ---
MODEL_NAME = "isnet-general-use" 
session = new_session(model_name=MODEL_NAME)


# --- 3. DEFINE ALL APP ROUTES ---

@app.route("/")
def serve_index():
    return send_from_directory(APP_DIR, "index.html")

@app.route("/remover.html")
def serve_remover():
    return send_from_directory(APP_DIR, "remover.html")

@app.route("/remove-background", methods=["POST"])
def remove_background_api():
    if "image" not in request.files:
        return "No file uploaded", 400
    f = request.files["image"]
    try:
        inp = Image.open(f.stream).convert("RGBA")
    except Exception as e:
        return f"Could not open image: {e}", 400

    out = remove(inp, session=session, alpha_matting=True)
    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/refine-mask", methods=["POST"])
def refine_mask():
    if "original_image" not in request.files or "mask_image" not in request.files or "edit_strokes" not in request.files:
        return "Missing files for refinement", 400
    try:
        original_image_bytes = request.files['original_image'].read()
        original_pil = Image.open(BytesIO(original_image_bytes)).convert("RGBA")
        original_np = np.array(original_pil)
        mask_pil = Image.open(request.files['mask_image'].read()).convert("L")
        mask_np = np.array(mask_pil)
        edit_strokes_pil = Image.open(request.files['edit_strokes'].read()).convert("RGBA")
        edit_strokes_np = np.array(edit_strokes_pil)

        keep_mask = (edit_strokes_np[:, :, 1] > 200) & (edit_strokes_np[:, :, 0] < 100) & (edit_strokes_np[:, :, 2] < 100)
        remove_mask = (edit_strokes_np[:, :, 0] > 200) & (edit_strokes_np[:, :, 1] < 100) & (edit_strokes_np[:, :, 2] < 100)

        final_mask_np = np.maximum(mask_np, keep_mask * 255)
        final_mask_np[remove_mask] = 0
        final_image_np = original_np.copy()
        final_image_np[:, :, 3] = final_mask_np

        final_image_pil = Image.fromarray(final_image_np)
        buf = BytesIO()
        final_image_pil.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return f"Error during refinement: {e}", 500

@app.route("/generate-background", methods=["POST"])
def generate_background():
    prompt = request.form.get("prompt")
    
    # --- IMPORTANT: PASTE YOUR API KEY HERE ---
    api_key = "sk-Zh6ghRuwZulKbeLGfLtCy4iR6zKMHcKNJuRToWEfo9fkS3qr" # Replace with your key

    if not prompt:
        return "Missing prompt", 400

    if api_key == "your_stability_api_key_goes_here":
        return "API key has not been set in the backend.py file.", 500

    api_host = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {"authorization": f"Bearer {api_key}", "accept": "image/*"}
    files = {"prompt": (None, prompt), "output_format": (None, "png"), "aspect_ratio": (None, "16:9")}

    try:
        response = requests.post(api_host, headers=headers, files=files)
        if response.status_code == 200:
            image_bytes = BytesIO(response.content)
            return send_file(image_bytes, mimetype='image/png')
        else:
            error_message = response.json().get("errors", ["Unknown error"])[0]
            return f"AI Generation Failed: {error_message}", response.status_code
    except Exception as e:
        return f"An error occurred: {e}", 500

# The if __name__ == "__main__": block has been removed.
# Gunicorn will now have full control.
