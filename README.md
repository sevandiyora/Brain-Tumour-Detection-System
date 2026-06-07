# 🧠 AI Brain Tumor Detection Web App

This project is a Flask-based web application that detects and segments brain tumors from MRI images using deep learning models (**U-Net** and **Mask R-CNN**). It provides visualization, heatmaps, and generates a professional PDF medical report.

---

## Live Demo
https://brain-tumour-detection-system-xzyz.onrender.com

## AI Model
https://huggingface.co/spaces/sevendiyora/brain-tumor-detection-api

## Features
- Brain MRI tumor segmentation
- U-Net deep learning model
- Heatmap visualization
- PDF report generation
- Flask web application
- Hugging Face AI deployment

---

## Project Architecture

User Uploads MRI
        ↓
Flask Web Application
        ↓
U-Net / Mask R-CNN
        ↓
Tumor Segmentation
        ↓
Heatmap Generation
        ↓
PDF Medical Report

## 🚀 Features

* Upload MRI scan
* Select model:
  * U-Net
  * Mask R-CNN
  * Both (comparison)
* Tumor segmentation overlay
* Heatmap visualization
* Tumor area calculation
* Confidence score display
* Downloadable PDF medical report

## 🏗️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask (Python)
* **ML Models:** U-Net, Mask R-CNN (via Gradio API)
* **Libraries:**

  * OpenCV
  * NumPy
  * Pillow
  * ReportLab
  * Gradio Client

---

## 📁 Project Structure

```
brain_tumor_webapp/
│
├── app.py
├── requirements.txt
├── static/
│   ├── style.css
│   ├── app.js
│   └── generated images
│
├── templates/
│   ├── index.html
│   ├── detect.html
│   ├── about.html
│   ├── how.html
│   └── future.html
│
└── report.pdf
```
## Live Demo

https://huggingface.co/spaces/sevendiyora/brain-tumor-detection-api

---

## ⚙️ Installation & Setup

### 🔹 Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd brain_tumor_webapp
```

---

### 🔹 Step 2: Create virtual environment

```bash
python -m venv .venv
```

Activate:

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

---

### 🔹 Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 Step 4: Run Gradio Model (IMPORTANT)

Make sure your ML model is running and gives a URL like:

```
https://xxxx.gradio.live/
```

Then update in `app.py`:

```python
client = Client("YOUR_GRADIO_URL")
```

---

### 🔹 Step 5: Run Flask App

```bash
python app.py
```

---

### 🔹 Step 6: Open in browser

```
http://127.0.0.1:5000
```

---

## 🧪 How to Use

1. Upload MRI image
2. Select model
3. Click **Run AI Detection**
4. View:

   * Segmentation results
   * Heatmaps
   * Confidence & tumor area
5. Click **Download Report**

---

## ⚠️ Common Issues

### ❌ Gradio Error

```
Could not fetch config
```

✔ Restart model → update URL

---

### ❌ No Report Generated

✔ Run detection first

---

### ❌ Images not loading

✔ Check `/static` folder paths

---

## 📈 Future Improvements

* Deploy models permanently (HuggingFace)
* Add 3D MRI support
* Real-time inference
* Doctor dashboard

---

## 👨‍💻 Author

Montclair State University
MS Computer Science

---

## 📄 License

This project is for academic and educational use.
