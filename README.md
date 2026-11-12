# Smart Crowd Density Monitoring & Public Safety Early Warning System using CSRNet

**Course**: CSBB 311 - Machine Learning (End-Semester Project)  
**Department**: Computer Science and Engineering, National Institute of Technology Durgapur (NITD)  
**Deep Learning Architecture**: CSRNet (Congested Scene Recognition Network, CVPR 2018)  
**Target Hardware**: NVIDIA GeForce RTX 2050 (Edge Computing Server) + IoT Safety Actuators  

---

## 📌 1. Project Overview & Motivation

Overcrowding at public venues (railway terminals, religious pilgrimages like Kumbh Mela or Rath Yatra, sports stadiums, and concerts) poses severe public safety hazards and fatal stampede risks. 

Traditional computer vision detection frameworks (such as YOLO or Faster R-CNN) rely on individual bounding boxes and fail drastically under severe occlusions and high crowd density. **CSRNet** overcomes this by reformulating detection as a continuous **Density Map Estimation problem**, enabling accurate counting and spatial density mapping without segregating individual pedestrians.

### Key Objectives
1. **Accurate Density Estimation**: Accurately predict crowd headcounts under both sparse and extreme congestion.
2. **Spatial Risk Localization**: Pinpoint exact bottlenecks and high-density clusters using continuous density heatmaps.
3. **Real-Time Stampede Early Warning**: Dynamically classify crowd risk levels (Safe, Warning, Critical Hazard) based on headcounts and rate of influx ($\frac{\Delta N}{\Delta t}$).
4. **Hardware & IoT Actuation**: Trigger physical alarms (sirens, strobe lights, and automated turnstile control) to physically regulate crowd flow.

---

## 🧠 2. Deep Learning Architecture (CSRNet)

CSRNet comprises two core components:
1. **Frontend (Feature Representation)**:
   - First 10 convolutional layers of **VGG-16** pretrained on ImageNet (downsampling factor = 8x).
   - Extracts rich multi-scale visual features.
2. **Backend (Dilated Convolutional Neural Network)**:
   - 6 dilated convolutional layers with dilation rate $r = 2$ and $3 \times 3$ kernels.
   - **Why Dilated Convolutions?** Standard pooling layers reduce spatial resolution. Dilated convolutions enlarge the receptive field exponentially without losing spatial resolution or introducing additional parameters.
3. **Output Layer**:
   - $1 \times 1$ convolution generating the single-channel density map.

### Mathematical Formulation
The predicted crowd count $N$ across an image $I$ is obtained by integrating the continuous density map:
$$\text{Total Crowd Count } N = \iint_{I} D(x, y) \, dx \, dy$$

**Training Loss Function**: Pixel-wise Euclidean Loss (Mean Squared Error):
$$L(\Theta) = \frac{1}{2N} \sum_{i=1}^N \|Z(X_i; \Theta) - Z^{GT}_i\|_2^2$$
where $Z(X_i; \Theta)$ is the predicted density map and $Z^{GT}_i$ is the ground-truth density map.

---

## 📊 3. Dataset Collection & Ground Truth Generation

The project utilizes the benchmark **ShanghaiTech Dataset**:
- **Part A (Highly Congested)**: 482 images scraped from the web with extreme crowd densities (average 501 people/image).
- **Part B (Surveillance / Streets)**: 716 images captured from Shanghai street surveillance cameras (average 123 people/image).

### Geometry-Adaptive Gaussian Kernels
Raw ground truth annotations are head center coordinates $(x_i, y_i)$. These are converted into smooth Gaussian density maps:
$$D(p) = \sum_{i=1}^N \delta(p - \bar{x}_i) * G_{\sigma_i}(p)$$

- **For Part A**: Geometry-adaptive spread $\sigma_i = \beta \cdot \bar{d}_i$ ($\beta=0.3$), where $\bar{d}_i$ is the mean distance to the $k=3$ nearest neighbors computed via KD-Tree.
- **For Part B**: Fixed Gaussian kernel $\sigma = 15.0$.

Density maps are stored in fast-loading **HDF5 (`.h5`)** files alongside the images.

---

## 📂 4. Project Directory Structure

```
Project-ENDSEM/
├── .venv/                         # Isolated Python Virtual Environment
├── ShanghaiTech/                  # Benchmark Dataset
│   ├── part_A/                    # Dense crowd subset (train_data & test_data)
│   └── part_B/                    # Surveillance subset (train_data & test_data)
├── checkpoints/                   # Saved PyTorch Model Weights (.pth)
├── eval_results/                  # Qualitative comparison figures (RGB vs Heatmap)
├── wheels_cache/                  # PyTorch & Torchvision CUDA wheels
├── model.py                       # CSRNet PyTorch architecture
├── dataset.py                     # Custom PyTorch DataLoader with 8x density scaling
├── make_density_maps.py           # Geometry-adaptive Gaussian density generator
├── train.py                       # Training pipeline with checkpointing & validation
├── evaluate.py                    # Evaluation script for MAE, MSE, and heatmap plots
├── app.py                         # Streamlit interactive public safety dashboard
├── hardware_simulator.py          # Arduino/ESP32 IoT serial alert actuator
├── requirements.txt               # Environment dependencies
└── README.md                      # Comprehensive academic report & documentation
```

---

## 🚀 5. How to Run the Project

### Step 1: Activate Virtual Environment
```powershell
.venv\Scripts\activate
```

### Step 2: Generate Ground Truth Density Maps (.h5)
*(Already executed: generated all 1,198 `.h5` files for Part A & Part B)*
```powershell
python make_density_maps.py
```

### Step 3: Train the CSRNet Model
Train on the surveillance dataset (`part_B`) or dense dataset (`part_A`):
```powershell
# Train on Part B (Surveillance CCTV) on RTX 2050 GPU
python train.py --dataset part_B --epochs 30 --lr 1e-5 --save_dir checkpoints
```

### Step 4: Evaluate Model Performance (MAE & MSE)
```powershell
python evaluate.py --checkpoint checkpoints/best_model_part_B.pth --dataset part_B --output_dir eval_results
```
This generates quantitative metrics (MAE, MSE) and saves side-by-side visual comparisons into `eval_results/`.

### Step 5: Launch the Interactive Safety Dashboard
```powershell
streamlit run app.py
```
**Features of the Dashboard**:
- Upload crowd images or select from ShanghaiTech test split.
- Live density heatmap overlay with Jet color-map.
- Automatic risk tier classification:
  - 🟢 **SAFE**: Crowd density within normal capacity.
  - 🟡 **WARNING**: Bottleneck congestion detected; ingress throttling advised.
  - 🔴 **CRITICAL**: Stampede risk; emergency sirens activated and automated alerts dispatched.
- Real-time FPS and latency benchmarks.

### Step 6: Test Hardware / IoT Alarm Actuation
```powershell
python hardware_simulator.py --mock
```

---

## 📈 6. Evaluation Metrics

| Metric | Formula | Objective |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | $\text{MAE} = \frac{1}{N}\sum \|C_{pred} - C_{gt}\|$ | Evaluates counting accuracy across diverse crowd scales. |
| **Mean Squared Error (MSE)** | $\text{MSE} = \sqrt{\frac{1}{N}\sum (C_{pred} - C_{gt})^2}$ | Measures robustness against extreme density spikes and outliers. |
| **Inference Latency & FPS** | Throughput (FPS) on RTX 2050 GPU | Validates real-time edge CCTV feasibility ($>15$ FPS). |

---

## 🔌 7. Hardware & Public Safety Integration

```
  [CCTV Camera] ──(RTSP/USB)──> [Edge Gateway: Laptop RTX 2050]
                                             │ (CSRNet Density Estimation)
                                             ▼
                                  [Capacity Threshold Logic]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [Serial / USB COM3]                       [Network / Webhook]
                       │                                           │
                       ▼                                           ▼
             [Arduino / ESP32]                             [Security Dispatch]
             ├── Pin 13: Piezo Siren                       ├── SMS / Telegram Alert
             ├── Pin 12: Strobe Warning Light              └── Incident Snapshot
             └── Pin 11: Turnstile Gate Lock
```

---

## 📚 References
1. **CSRNet**: Y. Li, X. Zhang, D. Chen, *"CSRNet: Dilated Convolutional Neural Networks for Understanding Highly Congested Scenes"*, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
2. **ShanghaiTech Dataset**: Y. Zhang, D. Zhou, S. Chen, S. Gao, Y. Ma, *"Single-Image Crowd Counting via Multi-Column Convolutional Neural Network"*, CVPR, 2016.

