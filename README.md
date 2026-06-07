# 🧠 AI-Powered Brain Tumor Detection & Segmentation System

A full-stack AI-powered medical imaging platform that analyzes brain MRI scans and performs automated tumor segmentation using deep learning. The application provides tumor visualization, heatmap generation, and downloadable PDF reports through an interactive web interface.

## 🌐 Live Demo

**Web Application:**
https://brain-tumour-detection-system-xzyz.onrender.com

**AI Inference Service:**
https://huggingface.co/spaces/sevendiyora/brain-tumor-detection-api

**Source Code:**
https://github.com/sevandiyora/Brain-Tumour-Detection-System

---

## 📌 Overview

This project was developed as part of graduate-level research in Computer Science. It combines modern web development and deep learning techniques to assist in the analysis of brain MRI scans.

The system allows users to:

* Upload MRI brain scans
* Perform automated tumor segmentation
* Visualize segmented tumor regions
* Generate heatmap visualizations
* Download AI-generated PDF reports
* Access results through a publicly deployed web application

---

## 🏗️ System Architecture

```text
User Uploads MRI Scan
          │
          ▼
 Flask Web Application
          │
          ▼
 Hugging Face AI Service
          │
          ▼
 U-Net Deep Learning Model
          │
          ▼
 Tumor Segmentation
          │
          ├──► Overlay Visualization
          ├──► Heatmap Generation
          └──► PDF Report Creation
```

---

## 🚀 Features

### AI-Powered Tumor Segmentation

* Automated MRI image analysis
* Deep learning-based segmentation
* High-quality tumor boundary visualization

### Medical Image Visualization

* Tumor overlay generation
* Heatmap visualization
* Side-by-side result comparison

### Report Generation

* Automated PDF medical reports
* Tumor area estimation
* Prediction confidence information

### Web Application

* Responsive user interface
* Real-time image upload
* Public cloud deployment
* Integrated AI inference pipeline

---

## 🛠️ Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Flask
* Python

### Artificial Intelligence

* U-Net
* Computer Vision
* Medical Image Segmentation

### Libraries & Tools

* OpenCV
* NumPy
* Pillow
* ReportLab
* Gradio Client
* Hugging Face Spaces

### Deployment

* Render
* Hugging Face
* GitHub

---

## 📂 Project Structure

```text
Brain-Tumour-Detection-System/
│
├── app.py
├── requirements.txt
├── src/
│   └── unet_model.py
│
├── templates/
│   ├── index.html
│   ├── detect.html
│   ├── about.html
│   ├── how.html
│   └── future.html
│
├── static/
│   ├── style.css
│   ├── app.js
│   ├── images/
│   └── generated_outputs/
│
└── docs/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/sevandiyora/Brain-Tumour-Detection-System.git
cd Brain-Tumour-Detection-System
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🧪 Usage

1. Upload a brain MRI image
2. Start AI detection
3. View tumor segmentation results
4. Analyze generated heatmaps
5. Download the PDF report

---

## 📊 Research Highlights

* Applied deep learning techniques for medical image segmentation
* Developed a cloud-hosted AI inference pipeline
* Integrated AI services with a production-style web application
* Automated visualization and reporting workflow

---

## 🔮 Future Enhancements

* Mask R-CNN cloud deployment
* Multi-model comparison dashboard
* 3D MRI volume segmentation
* User authentication system
* Patient history management
* Clinical decision-support features

---

## 👨‍💻 Author

**Sevan Diyora**
M.S. Computer Science
Montclair State University

LinkedIn: https://www.linkedin.com/in/sevan-diyora
GitHub: https://github.com/sevandiyora

---

## 📄 License

This project was developed for educational, research, and portfolio purposes.
