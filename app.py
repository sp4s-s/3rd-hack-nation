import os
import io
import json
import tempfile
import torch
import requests
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
import torchvision.transforms as T
from geopy.geocoders import Nominatim
from src.model import CompactGeoEmbed

MODEL_PATH = "model/geo_model.pth"
GEE_KEY_PATH = os.environ.get("GEE_KEY_PATH", "keys/gee_service_account.json")
DEVICE = torch.device("cpu")
TF_SIZE = (120, 120)

def load_model():
    model = CompactGeoEmbed(32, 96)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        print("✅ Model loaded successfully from local path.")
    except Exception as e:
        print("⚠️ Model load failed:", e)
    model.to(DEVICE).eval()
    return model

MODEL = load_model()
tf = T.Compose([T.Resize(TF_SIZE), T.ToTensor()])

def reverse_geocode(lat, lon):
    try:
        geolocator = Nominatim(user_agent="geo-risk-app", timeout=10)
        loc = geolocator.reverse((lat, lon), language="en")
        return loc.address if loc else "Unknown location"
    except Exception:
        return "Unknown location"

def init_gee():
    import ee
    if os.path.exists(GEE_KEY_PATH):
        try:
            with open(GEE_KEY_PATH, "r") as f:
                svc = json.load(f)
            credentials = ee.ServiceAccountCredentials(svc["client_email"], GEE_KEY_PATH)
            ee.Initialize(credentials)
            print("✅ Earth Engine initialized from local key.")
            return True
        except Exception as e:
            print("⚠️ Local GEE init failed:", e)
    gee_secret = os.environ.get("GEE_KEY_JSON")
    if gee_secret:
        try:
            svc = json.loads(gee_secret)
            credentials = ee.ServiceAccountCredentials(svc["client_email"], key_data=gee_secret)
            ee.Initialize(credentials)
            print("✅ Earth Engine initialized from Hugging Face secret.")
            return True
        except Exception as e:
            print("⚠️ Secret GEE init failed:", e)
    print("⚠️ GEE initialization failed.")
    return False

GEE_READY = init_gee()

def fetch_gee_images(lat, lon):
    try:
        if not GEE_READY:
            raise RuntimeError("GEE not initialized")
        import ee
        p = ee.Geometry.Point([lon, lat])
        region = p.buffer(1000).bounds()
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(p)
            .filterDate("2024-01-01", "2024-12-31")
            .sort("CLOUDY_PIXEL_PERCENTAGE")
            .first()
            .select(["B4", "B3", "B2"])
        )
        s2_url = s2.visualize(min=0, max=3000).getThumbURL(
            {"region": region, "dimensions": f"{TF_SIZE[0]}x{TF_SIZE[1]}", "format": "png"}
        )
        s2_img = Image.open(io.BytesIO(requests.get(s2_url, timeout=10).content)).convert("RGB")
        srtm = ee.Image("USGS/SRTMGL1_003")
        elev_url = srtm.visualize(min=0, max=3000).getThumbURL(
            {"region": region, "dimensions": f"{TF_SIZE[0]}x{TF_SIZE[1]}", "format": "png"}
        )
        elev_img = Image.open(io.BytesIO(requests.get(elev_url, timeout=10).content)).convert("L")
        return s2_img, elev_img
    except Exception as e:
        print("⚠️ GEE fetch failed:", e)
        return Image.new("RGB", TF_SIZE, (127, 127, 127)), Image.new("L", TF_SIZE, 127)

def fetch_weather(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,"
            "precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max"
            "&forecast_days=1&timezone=UTC"
        )
        r = requests.get(url, timeout=8)
        d = r.json().get("daily", {})
        return {
            "temp_max": float(d.get("temperature_2m_max", [None])[0]) if d else None,
            "precip": float(d.get("precipitation_sum", [None])[0]) if d else None,
            "humidity": float(d.get("relative_humidity_2m_mean", [None])[0]) if d else None,
            "wind_speed": float(d.get("wind_speed_10m_max", [None])[0]) if d else None,
        }
    except Exception:
        return {"temp_max": None, "precip": None, "humidity": None, "wind_speed": None}

def preprocess(img):
    return tf(img.convert("RGB")).unsqueeze(0)

def get_ip_location():
    try:
        r = requests.get("https://ipapi.co/json", timeout=5)
        data = r.json()
        return round(float(data["latitude"]), 5), round(float(data["longitude"]), 5)
    except Exception:
        return 51.5072, -0.1276

def predict(lat, lon, img):
    if lat is None or lon is None:
        lat, lon = get_ip_location()
    lat, lon = round(float(lat), 5), round(float(lon), 5)
    location_str = reverse_geocode(lat, lon)
    rgb_img, elev_img = fetch_gee_images(lat, lon)
    if img is not None:
        rgb_img = img.resize(TF_SIZE)
    x = preprocess(rgb_img).to(DEVICE)
    e = torch.tensor(np.array(elev_img) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        try:
            _, _, r = MODEL(x, e)
            risk_score = float(r.item())
        except Exception as e:
            print("⚠️ Model inference failed:", e)
            risk_score = None
    weather = fetch_weather(lat, lon)
    result = {
        "Location": location_str,
        "Latitude": lat,
        "Longitude": lon,
        "Predicted_Risk_Score": round(risk_score, 4) if risk_score is not None else None,
        **weather,
    }
    df = pd.DataFrame([result])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp_path = tmp.name
    df.to_csv(tmp_path, index=False)
    return rgb_img, df, tmp_path

with gr.Blocks() as demo:
    gr.Markdown("## 🌍 Geo-Risk Predictor (Local Model + GEE + Location Auto-Detect)")
    with gr.Row():
        lat = gr.Number(value=None, label="Latitude")
        lon = gr.Number(value=None, label="Longitude")
        get_loc_btn = gr.Button("📍 Use My Location")
    img = gr.Image(type="pil", label=f"Optional RGB Tile (auto-resized to {TF_SIZE[0]}×{TF_SIZE[1]})")
    run_btn = gr.Button("Run Prediction")
    rgb_preview = gr.Image(label="Satellite Image Used")
    output_df = gr.DataFrame(label="Predicted Data", interactive=False)
    file_out = gr.File(label="Download CSV")
    run_btn.click(fn=predict, inputs=[lat, lon, img], outputs=[rgb_preview, output_df, file_out])
    get_loc_btn.click(
        None, [], [lat, lon],
        js="""
        async () => {
            if (navigator.geolocation) {
                try {
                    const pos = await new Promise((res, rej) =>
                        navigator.geolocation.getCurrentPosition(res, rej)
                    );
                    return [pos.coords.latitude.toFixed(5), pos.coords.longitude.toFixed(5)];
                } catch {
                    const ip = await fetch("https://ipapi.co/json");
                    const data = await ip.json();
                    return [data.latitude.toFixed(5), data.longitude.toFixed(5)];
                }
            } else {
                const ip = await fetch("https://ipapi.co/json");
                const data = await ip.json();
                return [data.latitude.toFixed(5), data.longitude.toFixed(5)];
            }
        }
        """,
    )

if __name__ == "__main__":
    demo.launch()
