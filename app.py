"""
app.py
Smart Crowd Density Monitoring & Public Safety Early Warning System
Production Deployment Application powered by CSRNet v2 Pretrained A+B Best.
"""

import os
import time
import glob
import numpy as np
import cv2
import torch
import pandas as pd
from PIL import Image
import streamlit as st

from model import CSRNet

st.set_page_config(
    page_title="Smart Crowd Density Monitor | CSRNet v2",
    page_icon="👥",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #007bff;
    }
    .model-badge-v2 {
        background: #e3f2fd;
        color: #0d47a1;
        padding: 6px 12px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .status-safe {
        color: #28a745;
        font-weight: bold;
        font-size: 22px;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
        font-size: 22px;
    }
    .status-danger {
        color: #dc3545;
        font-weight: bold;
        font-size: 22px;
    }
</style>
""", unsafe_allow_html=True)


# ================= MODEL CONFIGURATION =================
MODEL_NAME = "CSRNet v2 (Pretrained A+B Best)"
CHECKPOINT_PATH = "checkpoints/csrnet_v2_pretrained_AB_best.pth"


@st.cache_resource
def load_csrnet_v2(checkpoint_path=CHECKPOINT_PATH):
    """
    Loads CSRNet v2 model weights (Pretrained VGG-16 + Part A & B Combined).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet(pretrained_frontend=False).to(device)
    meta = {}
    
    if os.path.exists(checkpoint_path):
        obj = torch.load(checkpoint_path, map_location=device)
        if isinstance(obj, dict):
            if "model_state_dict" in obj:
                state_dict = obj["model_state_dict"]
                meta = {k: v for k, v in obj.items() if k != "model_state_dict"}
            elif "state_dict" in obj:
                state_dict = obj["state_dict"]
                meta = {k: v for k, v in obj.items() if k != "state_dict"}
            else:
                state_dict = obj
        else:
            state_dict = obj
            
        model.load_state_dict(state_dict)
    
    model.eval()
    return model, device, meta


@st.cache_resource
def load_yolo_model():
    """Loads YOLOv8 Nano for sparse/empty scene verification and phantom noise rejection."""
    try:
        from ultralytics import YOLO
        if os.path.exists("yolov8n.pt"):
            return YOLO("yolov8n.pt")
    except ImportError:
        import sys
        venv_site = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
        if os.path.exists(venv_site) and venv_site not in sys.path:
            sys.path.append(venv_site)
        try:
            from ultralytics import YOLO
            if os.path.exists("yolov8n.pt"):
                return YOLO("yolov8n.pt")
        except Exception:
            pass
    except Exception:
        pass
    return None


def run_yolo_detection(image_rgb, yolo_model):
    """Executes YOLOv8 person detection and returns (count, boxes)."""
    if yolo_model is None:
        return None, []
    try:
        y_res = yolo_model(image_rgb, verbose=False, conf=0.3)[0]
        p_boxes = [b for b in y_res.boxes if int(b.cls[0]) == 0]
        yolo_count = len(p_boxes)
        yolo_boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in p_boxes]
        return yolo_count, yolo_boxes
    except Exception:
        return None, []


def predict_density(model, device, image_rgb, yolo_model=None, use_hybrid=True):
    """
    Hybrid Dual-Engine Crowd Monitoring with CSRNet v2:
    - Rejects blank / solid-color / empty backgrounds with the variance gate.
    - YOLOv8 handles sparse / empty scenes (0 to 25 people).
    - CSRNet v2 handles dense congested crowds (> 25 people) bypassing occlusion collapse.
    """
    orig_h, orig_w = image_rgb.shape[:2]
    
    # 1. Quick check for blank / solid-color background
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    is_solid_color = float(np.var(gray)) < 12.0
    
    # 2. YOLO Person Detection
    if use_hybrid and yolo_model is not None:
        yolo_count, yolo_boxes = run_yolo_detection(image_rgb, yolo_model)
    else:
        yolo_count, yolo_boxes = None, []
        
    # If solid color / blank image, instantly return 0 with zero phantom noise!
    if is_solid_color:
        zero_heatmap = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        return 0.0, zero_heatmap, image_rgb.copy(), 1.0, "Background Saliency Gate (Single-Color / Empty Background Clamped to 0)", (yolo_count, yolo_boxes)
        
    # 3. Preprocessing for CSRNet v2 (ImageNet Normalization)
    img_float = image_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    t0 = time.time()
    with torch.no_grad():
        pred_density = model(img_tensor)
    latency_ms = (time.time() - t0) * 1000.0
    
    density_raw = pred_density.squeeze().cpu().numpy()
    
    # Saliency noise gate: suppress diffuse residual baseline noise
    max_d = float(np.max(density_raw)) if density_raw.size > 0 else 0.0
    std_d = float(np.std(density_raw)) if density_raw.size > 0 else 0.0
    if max_d < 0.015 or std_d < 0.002:
        csrnet_count = 0.0
        density_raw = np.zeros_like(density_raw)
    else:
        cleaned_density = np.where(density_raw < 0.0015, 0.0, density_raw)
        csrnet_count = max(0.0, float(np.sum(cleaned_density)))
        
    # Dual-Regime Decision Engine
    if use_hybrid and yolo_count is not None:
        if yolo_count == 0 and csrnet_count < 8.0:
            pred_count = 0.0
            active_engine = "YOLOv8 Zero-Crowd Gate (Empty Scene Verified)"
        elif yolo_count < 25 and csrnet_count < 25:
            pred_count = float(yolo_count)
            active_engine = f"YOLOv8 Sparse Detector ({yolo_count} discrete bounding boxes)"
        else:
            pred_count = csrnet_count
            active_engine = f"CSRNet v2 Congested Density Engine ({csrnet_count:.1f} integrated people)"
    else:
        pred_count = csrnet_count
        active_engine = "CSRNet v2 (Saliency-Filtered Density)"
        
    # Visual heatmap overlay
    if pred_count == 0.0:
        heatmap_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        blend_overlay = image_rgb.copy()
    else:
        density_resized = cv2.resize(density_raw, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        density_norm = cv2.normalize(density_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        density_uint8 = np.uint8(np.clip(density_norm, 0, 255))
        heatmap_colored = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        blend_overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_rgb, 0.4, 0)
        
    # Draw YOLO bounding boxes on overlay if sparse engine active
    if "YOLOv8" in active_engine and yolo_boxes:
        for (x1, y1, x2, y2) in yolo_boxes:
            cv2.rectangle(blend_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    return pred_count, heatmap_rgb, blend_overlay, latency_ms, active_engine, (yolo_count, yolo_boxes)


# ================= SIDEBAR CONTROLS =================
st.sidebar.title("🎛️ Control Center")

# --- MODEL BANNER ---
st.sidebar.subheader("🤖 AI Model Engine")
st.sidebar.markdown("""
<div style="background:#eef4fb; border-left:4px solid #0066cc; padding:12px; border-radius:8px; margin-bottom:14px;">
    <div style="font-size:13px; font-weight:bold; color:#0066cc;">🏆 CSRNet v2 (Pretrained A+B Best)</div>
    <div style="font-size:11px; color:#444; margin-top:5px; line-height:1.4;">
        Trained on both ShanghaiTech Part A (Congested) and Part B (Surveillance) using ImageNet Pretrained VGG-16 frontend.
    </div>
</div>
""", unsafe_allow_html=True)

# Operating Mode Selector
mode = st.sidebar.radio(
    "Select Operating Mode",
    [
        "Dataset Samples (Part A/B)",
        "Upload Image",
        "Batch Multi-Image Testing",
        "Webcam / Live Stream",
        "Hardware / IoT Monitor"
    ]
)

# Dual-Engine Hybrid Toggle
st.sidebar.subheader("🧠 Dual-Engine AI Architecture")
enable_hybrid_yolo = st.sidebar.checkbox(
    "Enable YOLOv8 + CSRNet Hybrid", 
    value=True, 
    help="Uses YOLOv8 to verify sparse/empty scenes (clamps blank/solid backgrounds to 0), and CSRNet v2 for dense congested crowds."
)
yolo_model = load_yolo_model() if enable_hybrid_yolo else None
if enable_hybrid_yolo and yolo_model:
    st.sidebar.caption("🟢 YOLOv8 Nano Loaded (Zero False-Positives on Blank Scenes)")

# Safety Thresholds
st.sidebar.subheader("🚨 Safety Capacity Thresholds")
thresh_warning = st.sidebar.slider("Congestion Warning Threshold (Yellow)", 20, 500, 100)
thresh_critical = st.sidebar.slider("Critical Danger Threshold (Red)", 50, 1500, 250)

# Hardware Actuator
st.sidebar.subheader("🔌 Hardware Actuator Status")
enable_hardware_buzzer = st.sidebar.checkbox("Simulate IoT Buzzer & Strobe Light", value=True)
serial_port = st.sidebar.text_input("Hardware Serial COM Port (Optional)", value="COM3 (Mock)")

# Load CSRNet v2
model, device, meta = load_csrnet_v2()


# ================= MAIN DASHBOARD HEADER =================
st.title("🛡️ Smart Crowd Density Monitoring & Public Safety System")
st.markdown(
    "**Course**: CSBB 311 - Machine Learning | **Model**: `CSRNet v2 (Pretrained A+B Best)` | "
    "**Hardware**: PyTorch + cuDNN Acceleration | **Objective**: Stampede Prevention & Overcrowding Detection"
)
st.divider()


# ================= MODE 1: DATASET SAMPLES =================
if mode == "Dataset Samples (Part A/B)":
    st.subheader("🖼️ Dataset Sample Evaluation (ShanghaiTech)")
    dataset_choice = st.selectbox(
        "Select ShanghaiTech Subset", 
        ["part_B (Surveillance / Streets - Sparse to Moderate)", "part_A (Congested Crowds - Dense to Extreme)"]
    )
    subset = "part_B" if "part_B" in dataset_choice else "part_A"
    
    test_img_paths = sorted(glob.glob(f"ShanghaiTech/{subset}/test_data/images/*.jpg"))
    if not test_img_paths:
        st.error(f"No images found in ShanghaiTech/{subset}/test_data/images/!")
    else:
        sample_file = st.selectbox("Select Test Image", [os.path.basename(p) for p in test_img_paths[:40]])
        full_img_path = f"ShanghaiTech/{subset}/test_data/images/{sample_file}"
        
        # Load image
        img_bgr = cv2.imread(full_img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Check ground truth
        h5_path = full_img_path.replace(".jpg", ".h5")
        gt_count = None
        if os.path.exists(h5_path):
            import h5py
            with h5py.File(h5_path, "r") as hf:
                gt_count = float(np.sum(np.asarray(hf["density"])))
        else:
            mat_path = full_img_path.replace("images", "ground-truth").replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
            if os.path.exists(mat_path):
                from scipy.io import loadmat
                mat = loadmat(mat_path)
                pts = mat['image_info'][0, 0][0, 0][0] if "image_info" in mat else (mat['annPoints'] if 'annPoints' in mat else [])
                gt_count = float(len(pts))

        # CSRNet v2 Inference
        pred_count, heatmap_rgb, blend_overlay, latency_ms, engine_info, _ = predict_density(model, device, img_rgb, yolo_model, enable_hybrid_yolo)
        st.caption(f"🧠 **Model:** `CSRNet v2 Pretrained A+B Best` | **Active AI Engine:** `{engine_info}`")
        
        err = abs(pred_count - gt_count) if gt_count else None
        acc = max(0.0, (1.0 - err / max(gt_count, 1.0)) * 100.0) if gt_count else None
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Predicted Count", f"{pred_count:.0f} People")
        col2.metric("Ground Truth", f"{gt_count:.0f} People" if gt_count else "N/A", 
                    delta=f"{pred_count - gt_count:+.1f} Error" if gt_count else None)
        col3.metric("Counting Accuracy", f"{acc:.1f}%" if acc is not None else "N/A")
        col4.metric("Latency", f"{latency_ms:.1f} ms", f"{1000/latency_ms:.1f} FPS")
        
        # Safety Status Badge
        if pred_count < thresh_warning:
            status_text = "🟢 SAFE (Normal Flow)"
            alert_class = "status-safe"
            alert_msg = "Pedestrian transit is within safe capacity."
            actuator_state = "IDLE (Buzzer OFF)"
        elif pred_count < thresh_critical:
            status_text = "🟡 MODERATE (Caution)"
            alert_class = "status-warning"
            alert_msg = "Congestion detected at ingress bottlenecks. Monitor flow."
            actuator_state = "WARNING (Amber LED ON)"
        else:
            status_text = "🔴 CRITICAL (Stampede Risk!)"
            alert_class = "status-danger"
            alert_msg = "DANGER! Venue capacity exceeded. Divert pedestrian flow immediately!"
            actuator_state = "ALARM ACTIVE (Siren ON, Turnstiles LOCKED)"
            
        col5.markdown(f"**Safety Status**<br><span class='{alert_class}'>{status_text}</span>", unsafe_allow_html=True)
        
        if pred_count >= thresh_critical:
            st.error(f"🚨 **EMERGENCY ALERT**: {alert_msg}")
        elif pred_count >= thresh_warning:
            st.warning(f"⚠️ **CONGESTION NOTICE**: {alert_msg}")
        else:
            st.success(f"✅ {alert_msg}")
            
        # Visual display
        st.subheader("Visual Spatial Density Distribution")
        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.image(img_rgb, caption="Raw Input Camera Feed", use_container_width=True)
        vcol2.image(heatmap_rgb, caption=f"CSRNet v2 Density Heatmap ({pred_count:.1f} People)", use_container_width=True)
        vcol3.image(blend_overlay, caption="Augmented Safety Overlay", use_container_width=True)


# ================= MODE 2: UPLOAD IMAGE =================
elif mode == "Upload Image":
    st.subheader("📤 Upload Image (CCTV / Drone / Phone / Solid-Color Test)")
    uploaded_file = st.file_uploader("Upload Crowd Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        pred_count, heatmap_rgb, blend_overlay, latency_ms, engine_info, _ = predict_density(model, device, img_rgb, yolo_model, enable_hybrid_yolo)
        st.caption(f"🧠 **Model:** `CSRNet v2 Pretrained A+B Best` | **Active AI Engine:** `{engine_info}`")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Crowd Count", f"{pred_count:.0f} People")
        col2.metric("Inference Latency", f"{latency_ms:.1f} ms", f"{1000/latency_ms:.1f} FPS")
        status_label = "SAFE" if pred_count < thresh_warning else ("WARNING" if pred_count < thresh_critical else "CRITICAL")
        col3.metric("Safety Level", status_label)
        
        vcol1, vcol2 = st.columns(2)
        vcol1.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        vcol2.image(blend_overlay, caption=f"CSRNet v2 Density Overlay ({pred_count:.1f} People)", use_container_width=True)


# ================= MODE 3: BATCH MULTI-IMAGE TESTING =================
elif mode == "Batch Multi-Image Testing":
    st.subheader("📸 Batch Multi-Image Evaluation & Multi-Camera Array")
    st.markdown("Batch evaluate images across the venue using **CSRNet v2 (Pretrained A+B Best)**.")
    
    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
    with b_col1:
        batch_source = st.radio("Select Batch Source", ["ShanghaiTech Part B (Surveillance)", "ShanghaiTech Part A (Congested)", "Upload Multiple Files"], horizontal=True)
    
    total_avail = 316 if "Part B" in batch_source else (182 if "Part A" in batch_source else 50)
    with b_col2:
        num_batch = st.slider("Number of Images to Analyze", min_value=2, max_value=min(total_avail, 100), value=10, step=1)
    with b_col3:
        test_all = st.checkbox(f"Analyze ALL ({total_avail} images)", value=False)
        
    batch_img_paths = []
    uploaded_batch_images = []
    
    if batch_source == "ShanghaiTech Part B (Surveillance)":
        all_b = sorted(glob.glob("ShanghaiTech/part_B/test_data/images/*.jpg"))
        batch_img_paths = all_b if test_all else all_b[:num_batch]
    elif batch_source == "ShanghaiTech Part A (Congested)":
        all_a = sorted(glob.glob("ShanghaiTech/part_A/test_data/images/*.jpg"))
        batch_img_paths = all_a if test_all else all_a[:num_batch]
    else:
        uploaded_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            selected_files = uploaded_files if test_all else uploaded_files[:num_batch]
            for uf in selected_files:
                file_bytes = np.asarray(bytearray(uf.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                uploaded_batch_images.append((uf.name, img_rgb))

    run_batch = st.button("🚀 Run Batch Evaluation", type="primary")
    
    if run_batch:
        records = []
        gallery = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        items_to_process = []
        if uploaded_batch_images:
            items_to_process = [(name, img, None) for (name, img) in uploaded_batch_images]
        elif batch_img_paths:
            for p in batch_img_paths:
                name = os.path.basename(p)
                img_bgr = cv2.imread(p)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                gt_cnt = None
                h5_p = p.replace(".jpg", ".h5")
                if os.path.exists(h5_p):
                    import h5py
                    with h5py.File(h5_p, "r") as hf:
                        gt_cnt = float(np.sum(np.asarray(hf["density"])))
                else:
                    mat_p = p.replace("images", "ground-truth").replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
                    if os.path.exists(mat_p):
                        from scipy.io import loadmat
                        mat = loadmat(mat_p)
                        pts = mat['image_info'][0, 0][0, 0][0] if "image_info" in mat else (mat['annPoints'] if 'annPoints' in mat else [])
                        gt_cnt = float(len(pts))
                items_to_process.append((name, img_rgb, gt_cnt))
                
        for idx, (name, img_rgb, gt_cnt) in enumerate(items_to_process):
            status_text.text(f"Processing camera feed [{idx+1}/{len(items_to_process)}]: {name}...")
            
            pred_count, heatmap_rgb, blend_overlay, latency_ms, engine_info, _ = predict_density(
                model, device, img_rgb, yolo_model, enable_hybrid_yolo
            )
            
            status_badge = "🟢 SAFE" if pred_count < thresh_warning else ("🟡 WARNING" if pred_count < thresh_critical else "🔴 CRITICAL")
            border_color = (0, 220, 0) if pred_count < thresh_warning else ((255, 180, 0) if pred_count < thresh_critical else (220, 0, 0))
            
            err = abs(pred_count - gt_cnt) if gt_cnt else None
            acc = max(0.0, (1.0 - err / max(gt_cnt, 1.0)) * 100.0) if gt_cnt else None
            
            # Border indicator for venue camera view
            bordered = img_rgb.copy()
            cv2.rectangle(bordered, (0, 0), (bordered.shape[1]-1, bordered.shape[0]-1), border_color, 12)
            
            records.append({
                "Camera / File": name,
                "Predicted Count": round(pred_count, 1),
                "Ground Truth": round(gt_cnt, 1) if gt_cnt else "N/A",
                "Error": round(err, 1) if err else "N/A",
                "Accuracy (%)": f"{acc:.1f}%" if acc else "N/A",
                "Safety Level": status_badge,
                "Active Engine": engine_info,
                "Latency (ms)": round(latency_ms, 1),
                "_acc_val": acc,
                "_err_val": err,
                "_cnt": pred_count
            })
            gallery.append((name, bordered, heatmap_rgb, blend_overlay, pred_count, gt_cnt, status_badge, acc))
            progress_bar.progress((idx + 1) / len(items_to_process))
            
        status_text.success(f"✅ Batch evaluation completed for {len(items_to_process)} images!")
        df = pd.DataFrame(records)
        
        st.divider()
        
        # ==========================================================
        # 1. ACCURACY & PERFORMANCE METRICS ON THE TOP
        # ==========================================================
        st.subheader("🎯 Overall Batch Performance & Accuracy Telemetry")
        
        valid_accs = [x for x in df["_acc_val"] if x is not None]
        valid_errs = [x for x in df["_err_val"] if x is not None]
        mean_acc = np.mean(valid_accs) if valid_accs else None
        mean_mae = np.mean(valid_errs) if valid_errs else None
        total_crowd = sum(df["_cnt"])
        safe_zones = sum(1 for r in records if "SAFE" in r["Safety Level"])
        warn_zones = sum(1 for r in records if "WARNING" in r["Safety Level"])
        crit_zones = sum(1 for r in records if "CRITICAL" in r["Safety Level"])
        
        c1, c2, c3, c4, c5 = st.columns(5)
        if mean_acc is not None:
            c1.metric("🎯 Mean Counting Accuracy", f"{mean_acc:.1f}%", help="Average counting accuracy relative to ground truth across all images.")
        else:
            c1.metric("Counting Accuracy", "N/A (No GT)")
            
        if mean_mae is not None:
            c2.metric("📉 Mean Absolute Error (MAE)", f"{mean_mae:.1f} Heads", help="Average headcount deviation.")
        else:
            c2.metric("Monitored Cameras", f"{len(df)} Feeds")
            
        c3.metric("👥 Total Venue Crowd", f"{total_crowd:,.0f} People")
        c4.metric("🟢 Safe Zones", f"{safe_zones} Locations")
        c5.metric("🔴 Hotspots / Warnings", f"{crit_zones} Crit | {warn_zones} Warn", delta_color="inverse")

        # ==========================================================
        # 2. IMAGE GALLERY DISPLAY
        # ==========================================================
        st.divider()
        st.subheader("🖼️ Processed Images & Spatial Density Heatmaps")
        
        num_to_show = min(len(gallery), 12)
        if len(gallery) > 4:
            num_to_show = st.slider("Select how many images to display:", min_value=2, max_value=len(gallery), value=min(len(gallery), 12), key="batch_gallery_slider")
            
        for idx_g in range(num_to_show):
            name, bordered, heatmap_rgb, blend_overlay, pred_count, gt_cnt, status_badge, acc = gallery[idx_g]
            acc_label = f" | Accuracy: {acc:.1f}%" if acc is not None else ""
            gt_label = f" | Ground Truth: {gt_cnt:.0f}" if gt_cnt is not None else ""
            with st.expander(f"📷 [{idx_g+1}/{num_to_show}] {name} — Estimated: {pred_count:.0f} People ({status_badge}){acc_label}{gt_label}", expanded=(idx_g < 4)):
                gc1, gc2, gc3 = st.columns(3)
                gc1.image(bordered, caption=f"Camera Feed: {name} (Actual: {gt_cnt if gt_cnt is not None else 'N/A'})", use_container_width=True)
                gc2.image(heatmap_rgb, caption=f"CSRNet v2 Density Heatmap ({pred_count:.1f} People{acc_label})", use_container_width=True)
                gc3.image(blend_overlay, caption=f"Augmented Safety Overlay ({status_badge})", use_container_width=True)

        # ==========================================================
        # 3. DETAILED TELEMETRY TABLE & EXPORT
        # ==========================================================
        st.divider()
        st.subheader("📋 Detailed Multi-Camera Telemetry Report")
        
        cols_to_drop = ["_acc_val", "_err_val", "_cnt"]
        clean_df = df.drop(columns=cols_to_drop, errors="ignore")
        st.dataframe(clean_df, use_container_width=True)
        
        csv_data = clean_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Batch CSV Report", data=csv_data, file_name="crowd_safety_batch_report.csv", mime="text/csv")


# ================= MODE 4: WEBCAM LIVE MONITORING =================
elif mode == "Webcam / Live Stream":
    st.info("Using system camera for real-time inference with `CSRNet v2 (Pretrained A+B Best)`.")
    run_cam = st.checkbox("Start Live Monitoring Stream")
    cam_display = st.image([])
    
    if run_cam:
        cap = cv2.VideoCapture(0)
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not access webcam.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_count, heatmap_rgb, blend_overlay, latency_ms, engine_info, _ = predict_density(model, device, frame_rgb, yolo_model, enable_hybrid_yolo)
            
            # HUD overlay
            cv2.putText(blend_overlay, f"CSRNet v2 Count: {pred_count:.0f}", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if pred_count < thresh_warning else (0, 0, 255), 2)
            cam_display.image(blend_overlay, use_container_width=True)
        cap.release()


# ================= MODE 5: HARDWARE & IOT =================
elif mode == "Hardware / IoT Monitor":
    st.subheader("Hardware & IoT Stampede Warning System Architecture")
    st.markdown("""
    ### Edge Deployment & Public Safety Integration
    This module coordinates real-time AI density inference with physical hardware alarms:
    
    1. **Edge Gateway**: Laptop / RTX 2050 executes CSRNet v2 on incoming RTSP IP Camera stream.
    2. **Serial / GPIO Link**: Communicates via `PySerial` (`COM3` or `/dev/ttyUSB0`) to an **Arduino UNO** or **ESP32**.
    3. **Physical Actuators**:
       - **Green LED**: Normal flow, venue capacity safe.
       - **Buzzer / Siren**: High-decibel audible alarm triggered when crowd density exceeds safe limits ($> 1.5 \text{ persons/m}^2$).
       - **Automated Turnstile**: Stops entry gate turnstiles to prevent further crowd influx into crowded platforms.
    4. **Cloud / Telegram Webhook**: Dispatches automated SMS/Telegram alert with snapshot to security officers.
    """)
    
    st.code("""
    // Arduino C++ Receiver Code (Hardware Actuator):
    void setup() {
        Serial.begin(9600);
        pinMode(13, OUTPUT); // Buzzer Pin
        pinMode(12, OUTPUT); // Red LED Pin
    }
    void loop() {
        if (Serial.available() > 0) {
            char signal = Serial.read();
            if (signal == 'R') {       // 'R' = RED CRITICAL
                digitalWrite(13, HIGH); // Sound Alarm
                digitalWrite(12, HIGH); // Turn on Red Strobe
            } else {
                digitalWrite(13, LOW);
                digitalWrite(12, LOW);
            }
        }
    }
    """, language="cpp")
