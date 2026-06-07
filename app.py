from flask import Flask, render_template, request, jsonify
from gradio_client import Client, handle_file
import os
import shutil
import uuid
import requests
from flask import send_file
import numpy as np

last_unet_overlay = None
last_rcnn_overlay = None
last_heatmap = None
last_heatmap_rcnn = None
last_area = 0
last_conf = 0

app = Flask(__name__)

# 🔥 GRADIO API
client = Client("sevendiyora/brain-tumor-detection-api")

# App routes 
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/how")
def how():
    return render_template("how.html")

@app.route("/future")
def future():
    return render_template("future.html")

# function to see if tumor is detected or not 
CONF_THRESHOLD = 0.5

def get_diagnosis(area, conf_dict):
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    # Handle dict or single value safely
    if isinstance(conf_dict, dict):
        unet_conf = safe_float(conf_dict.get("unet"))
        mask_conf = safe_float(conf_dict.get("rcnn"))
    else:
        unet_conf = safe_float(conf_dict)
        mask_conf = 0.0

    try:
        area = float(area)
    except:
        area = 0.0

    # FINAL DECISION
    if area > 0 and (unet_conf >= CONF_THRESHOLD or mask_conf >= CONF_THRESHOLD):
        return "Tumor Detected"
    else:
        return "No Significant Tumor Detected"
    
# funtion to maintain the image size in PDF report 
from reportlab.platypus import Image
from PIL import Image as PILImage

def get_resized_image(path, max_width=160):
    img = PILImage.open(path)
    width, height = img.size

    aspect = height / width
    new_width = max_width
    new_height = new_width * aspect

    return Image(path, width=new_width, height=new_height)


# function to download report 
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from flask import send_file
import datetime

@app.route("/download_report")
def download_report():

    global last_unet_overlay, last_rcnn_overlay, last_heatmap, last_area, last_conf

    # Safety check
    if last_unet_overlay is None and last_rcnn_overlay is None:
        return "⚠️ Please run detection before downloading report."

    # Create PDF
    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    # ---------------- HELPER ----------------
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    # Extract areas safely
    if isinstance(last_area, dict):
        area_unet = safe_float(last_area.get("unet", 0))
        area_rcnn = safe_float(last_area.get("rcnn", 0))
    else:
        area_unet = safe_float(last_area)
        area_rcnn = 0

    # Extract confidence safely
    if isinstance(last_conf, dict):
        unet_conf = safe_float(last_conf.get("unet", 0))
        mask_conf = safe_float(last_conf.get("rcnn", 0))
    else:
        unet_conf = safe_float(last_conf)
        mask_conf = 0

    # ---------------- HEADER ----------------
    content.append(Paragraph("<b>AI Tumor Detection System</b>", styles["Heading1"]))
    content.append(Paragraph("<b>Brain MRI Analysis Report</b>", styles["Title"]))
    content.append(Spacer(1, 8))

    content.append(Paragraph("Patient: Anonymous", styles["Normal"]))
    content.append(Paragraph(
        f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    content.append(Spacer(1, 15))

    # ---------------- EXAM ----------------
    content.append(Paragraph("<b>1. Exam Type</b>", styles["Heading2"]))
    content.append(Paragraph("MRI Brain Scan (AI-assisted analysis)", styles["Normal"]))
    content.append(Spacer(1, 10))

    # ---------------- TECHNIQUE ----------------
    content.append(Paragraph("<b>2. Technique</b>", styles["Heading2"]))
    content.append(Paragraph(
        "Automated tumor segmentation performed using deep learning models (U-Net and Mask R-CNN).",
        styles["Normal"]
    ))
    content.append(Spacer(1, 10))

    # ---------------- FINDINGS ----------------
    content.append(Paragraph("<b>3. Findings</b>", styles["Heading2"]))

    content.append(Paragraph(
        f"U-Net detected region area: {area_unet:.2f} mm²",
        styles["Normal"]
    ))

    if last_rcnn_overlay:
        content.append(Paragraph(
            f"Mask R-CNN detected region area: {area_rcnn:.2f} mm²",
            styles["Normal"]
        ))

    content.append(Spacer(1, 10))

    # ---------------- MODEL PERFORMANCE ----------------
    content.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))

    # Safe extraction
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    # Handle dict or single
    if isinstance(last_conf, dict):
        unet_conf = safe_float(last_conf.get("unet"))
        mask_conf = safe_float(last_conf.get("rcnn"))
    else:
        unet_conf = safe_float(last_conf)
        mask_conf = 0.0

    # Area handling
    if isinstance(last_area, dict):
        unet_area = safe_float(last_area.get("unet"))
        mask_area = safe_float(last_area.get("rcnn"))
    else:
        unet_area = safe_float(last_area)
        mask_area = 0.0


    # -------- CONDITIONAL DISPLAY --------

    if last_model == "U-Net":
        content.append(Paragraph(
            f"U-Net → Confidence: {unet_conf:.3f} | Area: {unet_area:.2f} mm²",
            styles["Normal"]
        ))

    elif last_model == "Mask R-CNN":
        content.append(Paragraph(
            f"Mask R-CNN → Confidence: {mask_conf:.3f} | Area: {mask_area:.2f} mm²",
            styles["Normal"]
        ))

    elif last_model == "Both":
        content.append(Paragraph(
            f"U-Net → Confidence: {unet_conf:.3f} | Area: {unet_area:.2f} mm²",
            styles["Normal"]
        ))

        content.append(Paragraph(
            f"Mask R-CNN → Confidence: {mask_conf:.3f} | Area: {mask_area:.2f} mm²",
            styles["Normal"]
        ))

    # ---------------- DIAGNOSIS ----------------
    area_for_diag = max(area_unet, area_rcnn)
    diagnosis = get_diagnosis(area_for_diag, last_conf)

    if diagnosis == "Tumor Detected":
        color = "red"
    else:
        color = "green"

    content.append(Paragraph("<b>IMPRESSION</b>", styles["Heading2"]))

    content.append(Paragraph(
        f"<font color='{color}'><b>{diagnosis.upper()}</b></font>",
        styles["Normal"]
    ))

    # ---------------- IMAGES ----------------
    from reportlab.platypus import Table

    image_data = []

    row = []

    if last_unet_overlay:
        row.append(get_resized_image(last_unet_overlay, 250))
    else:
        row.append("")

    if last_rcnn_overlay:
        row.append(get_resized_image(last_rcnn_overlay, 250))
    else:
        row.append("")

    image_data.append(row)

    table = Table(image_data)
    content.append(Paragraph("<b>AI Visualization</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))
    content.append(table)
    content.append(Spacer(1, 10))   

    # Heatmap
    content.append(Paragraph("<b>Heatmap Visualization</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    if last_model == "U-Net":
        if last_heatmap:
            content.append(Paragraph("U-Net Heatmap", styles["Normal"]))
            content.append(get_resized_image(last_heatmap))

    elif last_model == "Mask R-CNN":
        if last_heatmap_rcnn:
            content.append(Paragraph("Mask R-CNN Heatmap", styles["Normal"]))
            content.append(get_resized_image(last_heatmap_rcnn))

    elif last_model == "Both":
        if last_heatmap:
            content.append(Paragraph("U-Net Heatmap", styles["Normal"]))
            content.append(get_resized_image(last_heatmap))

        if last_heatmap_rcnn:
            content.append(Paragraph("Mask R-CNN Heatmap", styles["Normal"]))
            content.append(get_resized_image(last_heatmap_rcnn))

    # ---------------- DISCLAIMER ----------------
    content.append(Paragraph("<b>Disclaimer</b>", styles["Heading3"]))
    content.append(Paragraph(
        "This AI-generated report is for research and educational purposes only. "
        "It should not be used as a substitute for professional medical diagnosis.",
        styles["Normal"]
    ))

    # footer 
    def add_footer(canvas, doc):
        canvas.drawString(200, 20, "AI Tumor Detection System | Confidential Report")

    # Build PDF
    doc.build(content, onFirstPage=add_footer, onLaterPages=add_footer)

    return send_file(file_path, as_attachment=True)

# function to calculate tumor area 
def calculate_tumor_area(mask_path):
    import cv2
    import numpy as np

    mask_path = mask_path.lstrip("/")  

    mask = cv2.imread(mask_path, 0)

    if mask is None:
        raise ValueError(f"Mask not found at path: {mask_path}")

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    pixel_count = np.sum(binary == 255)

    pixel_spacing_mm = 0.5
    area_mm2 = pixel_count * (pixel_spacing_mm ** 2)

    return round(area_mm2, 2)

# function to generate the heat map     
def create_heatmap(image_path, mask_path):
    import cv2
    import os

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("ERROR: Image not found:", image_path)
        return None

    if mask is None:
        print("ERROR: Mask not found:", mask_path)
        return None

    #Resize mask to match image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    #Apply heatmap
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    #Ensure same type
    heatmap = heatmap.astype(image.dtype)
    #Blend safely
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    output_path = os.path.join("static", "heatmap_" + os.path.basename(mask_path))
    cv2.imwrite(output_path, overlay)

    return "/" + output_path.replace("\\", "/")


# 🔥 MAIN API CALL ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    model = request.form.get("model")

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # -------- helper to download gradio file --------
    def save_file(gradio_output, name):
        out_path = f"static/output_{name}_{uuid.uuid4().hex}.png"

        if isinstance(gradio_output, dict):
            src = gradio_output.get("url") or gradio_output.get("path")
        else:
            src = gradio_output

        if os.path.exists(src):
            shutil.copy(src, out_path)
        elif str(src).startswith("http"):
            response = requests.get(src)
            with open(out_path, "wb") as f:
                f.write(response.content)
        else:
            raise ValueError(f"Unsupported file format: {src}")

        return out_path.replace("\\", "/")

    # ---------------- SAFE CONF ----------------
    import re
    def safe_conf(x):
        if isinstance(x, dict):
            x = x.get("label", 0)

        if isinstance(x, str):
            match = re.search(r"\d+\.\d+", x)
            if match:
                return float(match.group())

        try:
            return float(x)
        except:
            return 0.0

    # ---------------- RUN MODELS ----------------
    overlay_u_path = mask_u_path = heatmap_u = None
    overlay_r_path = mask_r_path = heatmap_r = None
    area_u = area_r = None
    conf_u_val = conf_r_val = 0.0

    # ---- U-NET ----
    if model in ["U-Net", "Both"]:
        result_unet = client.predict(
            handle_file(filepath)
        )

        overlay_unet, mask_unet = result_unet

        conf_unet = 1.0

        overlay_u_path = save_file(overlay_unet, "overlay_unet")
        mask_u_path = save_file(mask_unet, "mask_unet")

        real_mask_u = os.path.join("static", os.path.basename(mask_u_path))
        area_u = float(calculate_tumor_area(real_mask_u))
        conf_u_val = safe_conf(conf_unet)

        heatmap_u = create_heatmap(filepath, real_mask_u)

    # ---- MASK R-CNN ----
    if False:
        result_rcnn = client.predict(
            image=handle_file(filepath),
            model_choice="Mask R-CNN",
            api_name="/segment_image"
        )
        overlay_rcnn, mask_rcnn, conf_rcnn = result_rcnn

        overlay_r_path = save_file(overlay_rcnn, "overlay_rcnn")
        mask_r_path = save_file(mask_rcnn, "mask_rcnn")

        real_mask_r = os.path.join("static", os.path.basename(mask_r_path))
        area_r = float(calculate_tumor_area(real_mask_r))
        conf_r_val = safe_conf(conf_rcnn)

        heatmap_r = create_heatmap(filepath, real_mask_r)

    # ---------------- STORE GLOBALS ----------------
    global last_unet_overlay, last_rcnn_overlay
    global last_heatmap, last_heatmap_rcnn
    global last_area, last_conf, last_model

    last_unet_overlay = overlay_u_path.lstrip("/") if overlay_u_path else None
    last_rcnn_overlay = overlay_r_path.lstrip("/") if overlay_r_path else None

    last_heatmap = heatmap_u.lstrip("/") if heatmap_u else None
    last_heatmap_rcnn = heatmap_r.lstrip("/") if heatmap_r else None

    last_area = {"unet": area_u, "rcnn": area_r}
    last_conf = {"unet": conf_u_val, "rcnn": conf_r_val}

    last_model = model

    # ---------------- RESPONSE ----------------
    return jsonify({
        "model": model,

        "unet_overlay": overlay_u_path,
        "rcnn_overlay": overlay_r_path,

        "unet_heatmap": heatmap_u,
        "rcnn_heatmap": heatmap_r,

        "confidence": {
            "unet": conf_u_val,
            "rcnn": conf_r_val
        },

        "area": {
            "unet": area_u,
            "rcnn": area_r
        }
    })
if __name__ == "__main__":
    app.run(debug=True)
