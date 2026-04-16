# Sentinel AI — Real-Time Video Intelligence Pipeline

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-brightgreen.svg)
![React 18](https://img.shields.io/badge/Frontend-React%2018-cyan.svg)
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Sentinel AI** is a high-performance computer vision pipeline designed for industrial safety and autonomous monitoring. By combining real-time person tracking, 17-point skeletal pose estimation, and temporal behavioral classification, Sentinel AI identifies critical anomalies—such as falls, unauthorized zone intrusions, and loitering—with sub-100ms latency. The system features a cinematic, JARVIS-inspired HUD for real-time situational awareness and automated forensic recording.

## 🚀 Features

- **Fall Detection**: A robust 4-signal state machine monitoring velocity, aspect ratio, head drop, and post-fall stillness.
- **Dynamic Zone Monitoring**: User-definable polygon zones (Restricted, Time-Limited) with millisecond-accurate intrusion auditing.
- **Behavioral Intelligence**: ActionLSTM classifier trained on temporal pose sequences for nuanced action recognition (e.g., fighting, loitering).
- **PPE Compliance**: (In Development) Integrated compliance checking for safety helmets and vests.
- **Cinematic HUD**: Real-time React dashboard with WebSocket streaming, CRT-scanline overlays, and multi-track telemetry.
- **Forensic Storage**: Rolling 10-second MP4 buffers that persist automatically upon alert detection.

## 🏗️ Architecture

Sentinel AI operates on a strictly decoupled 6-layer intelligence stack:

```text
[ LAYER 6: HUD & INTERFACE ] <---------- [ REAL-TIME WEB HUD ]
      ^                                  (React, WebSockets)
      |
[ LAYER 5: INTELLIGENCE ENGINE ] <------ [ BEHAVIORAL ANALYSIS ]
      ^                                  (State Machines, ActionLSTM)
      |
[ LAYER 4: TEMPORAL ANALYTICS ] <------- [ FEATURE ACCUMULATION ]
      ^                                  (Sliding Windows, Normalized Sequences)
      |
[ LAYER 3: POSE ESTIMATION ] <---------- [ SKELETAL EXTRACTION ]
      ^                                  (YOLOv8-Pose, 17-Point Flux)
      |
[ LAYER 2: PERCEPTION CORE ] <---------- [ DETECTION & TRACKING ]
      ^                                  (YOLOv8, ByteTrack IDs)
      |
[ LAYER 1: VIDEO INGESTION ] <---------- [ THREADED CAPTURE ]
                                         (RTSP, USB-Cam, SQLite Forensic)
```

For a detailed deep-dive into each layer, see [ARCHITECTURE.md](ARCHITECTURE.md).

## ⚡ Quick Start

### Local Setup
1. **Clone & Install**:
   ```bash
   git clone https://github.com/yourusername/sentinel-ai.git
   cd sentinel-ai
   pip install -r backend/requirements.txt
   ```
2. **Launch Pipeline**:
   ```bash
   python backend/main.py --source 0
   ```
3. **Open Dashboard**:
   Navigate to `http://localhost:5173` (ensure you have run `npm install && npm run dev` in the `frontend` folder).

### 🐳 Docker Deployment
```bash
docker-compose up --build
```

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Detection** | YOLOv8 (Ultralytics) |
| **Tracking** | ByteTrack |
| **Pose Estimation** | YOLOv8-Pose |
| **Anomaly Logic** | State Machines + PyTorch LSTM |
| **Backend** | FastAPI / WebSockets |
| **Frontend** | React 18, Tailwind CSS, Recharts |
| **Storage** | SQLite + Local MP4 Rolling Buffer |

## 📦 Training your own Action Models
Sentinel AI includes a dedicated training pipeline to adapt to your environment:
1. **Prep**: `python training/data_prep.py --input_dir data/raw_videos`
2. **Train**: `python training/train_fall_detector.py --data_dir data/poses`
3. **Evaluate**: `python training/evaluate.py --model_path models/action_lstm.pt`

## 🔮 Future Roadmap
- [ ] **Cross-Camera ReID**: Maintain track IDs across multiple overlapping camera feeds.
- [ ] **Edge Deployment**: Optimization for NVIDIA Orin and Coral Edge TPU via TensorRT/OpenVINO.
- [ ] **Attention-Based Temporal Model**: Implement Vision Transformers (ViT) for long-duration behavioral patterns.
- [ ] **Cloud Dashboard**: Centralized management portal for multi-site deployments.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
