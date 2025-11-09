---
title: Geo-Risk Predictor
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---
[Dont Upload Pictures]

https://huggingface.co/spaces/Pingsz/3rd-hack-nation

# 🌍 Geo-Risk Predictor

Interactive Gradio app that estimates environmental and geographic risk using:
- **Satellite imagery** (Google Earth Engine)
- **Elevation data** (SRTM)
- **Weather data** (Open-Meteo)
- **Custom PyTorch model** for risk inference

while hosting fix pydantic version else it will break in Container after building 

## How to run locally
```bash
pip install -r requirements.txt
python app.py
```
-------------------------------
