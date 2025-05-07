# 🏓 Table Tennis Analyzer

**Table Tennis Analyzer** is an intelligent and fully automated web tool for analyzing table tennis matches using computer vision, OCR, and Streamlit.

This captivating project allows users to effortlessly upload a match video and instantly receive:
- 🎯 **Player Detection & Tracking** using YOLOv8 pose estimation
- 🌡️ **Heatmaps** of player movement and positioning
- 📊 **Live Score Recognition** using EasyOCR
- 🎥 **Annotated Match Video** with real-time overlays
- 📈 **Score Progress Chart** (optional)

---

## 🚀 Features

- 🔍 **YOLO Pose Model Integration** — accurate detection of players in each frame.
- 📐 **Smart Player Selection Logic** — filters and identifies two main players using spatial analysis.
- 🧠 **Score Parsing** — uses OCR with preprocessing and error handling to extract scores reliably.
- 🌈 **Heatmap Generation** — visualizes movement density beautifully.
- 📉 **Score Charts** — elegant visualization of score progression over time.
- 📁 **Downloadable Outputs** — video, heatmap, score chart, and CSV data, all exportable.

---

## 🖥️ How to Use

1. Upload a video (`.mp4`, `.avi`, `.mov`) through the Streamlit interface.
2. Configure optional parameters (start/end frame, OCR interval, etc.).
3. Click **"Run Analysis"**.
4. View and download results: heatmap, chart, annotated video, and CSV.

> Works entirely through a friendly and accessible Streamlit UI — no coding needed!

---

## 📦 Dependencies

- Python 3.8+
- `ultralytics`, `opencv-python`, `matplotlib`, `numpy`, `easyocr`, `pandas`, `streamlit`

Install them via:
```bash
pip install -r requirements.txt