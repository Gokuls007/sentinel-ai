# Sentinel AI - System Architecture

Sentinel AI is built on a high-throughput, 6-layer perception-to-intelligence pipeline designed for extreme low-latency video analytics and behavioral understanding.

## 🏗️ The 6-Layer Intelligence Stack

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

## 🛠️ Layer Responsibilities

1.  **Layer 1: Video Ingestion**: Managed by a persistent daemon thread that decouples video frame polling from processing logic. Includes a rolling 10-second forensic buffer for event recording.
2.  **Layer 2: Perception Core**: Uses YOLOv8 (Detector) and ByteTrack (Object Tracking) to assign and maintain persistent identities for every individual in the frame.
3.  **Layer 3: Pose Estimation**: Parallelized YOLOv8-Pose model that extracts 17 keypoint coordinates per person for dense skeletal understanding.
4.  **Layer 4: Temporal Analytics**: Accumulates multi-frame features (velocity, displacement, aspect ratio) into sliding window buffers.
5.  **Layer 5: Intelligence Engine**: Combined rule-based state machines (Fall detection, Zone Violations) and Deep Learning classifiers (ActionLSTM) for behavioral anomaly detection.
6.  **Layer 6: HUD & Interface**: A FastAPI-based WebSocket server that streams Base64 annotated frames and real-time telemetry to a high-fidelity React dashboard.
