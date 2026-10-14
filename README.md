# Smart Mask & Expression Detector

A production-ready, client-side web application for real-time Face Mask Compliance Verification and Facial Expression Recognition. Built with Vanilla JavaScript, **face-api.js**, and **TensorFlow.js / Google Teachable Machine**.

---

## 🌟 Key Features

1. **Real-Time Webcam Face Tracking**: Fast, real-time multi-face detection using `TinyFaceDetector`.
2. **Dual-Model Pipeline**:
   - **Mask Detection**: Powered by a custom TensorFlow.js model trained on Google Teachable Machine (`mask_model/`).
   - **Expression Recognition**: Powered by `FaceExpressionNet` & `FaceLandmark68Net` (detects *Happy, Sad, Angry, Surprised, Neutral, Fearful, Disgusted*).
3. **Smart Conditional Bounding Boxes**:
   - **Mask Detected**: Emerald Green bounding box + `😷 Mask (confidence%)`.
   - **No Mask Detected**: Crimson Red bounding box + `😊 [Emotion] (confidence%)`.
4. **Mirrored Camera Feed with Perfectly Aligned Overlay**:
   - Live video is mirrored horizontally like a real mirror.
   - Canvas coordinate transformation ensures labels remain upright, left-to-right readable, and non-inverted.
5. **Interactive Top Status Banner & Pill Badge**: Large real-time state banner and bottom pill badge with dynamic glowing color states.
6. **Analytics Counters & FPS Meter**: Real-time tracking of active faces in frame, total mask vs no-mask detections, and inference FPS.
7. **Photo Capture**: Instant screenshot button compositing the mirrored video feed, detected bounding boxes, and timestamp into a downloadable PNG.
8. **Synthetic Web Audio Alerts**: Pleasant melodic chime on mask compliance and double alert beep on no-mask warning (zero external audio file dependencies).

---

## 📁 Project File Structure

```text
MASK DETECTION/
├── index.html              # Main HTML5 structure & UI layout
├── style.css               # Modern dark theme (#1a1a2e to #16213e) & neon glowing borders
├── script.js               # Application logic (camera, inference loop, audio, canvas)
├── lib/
│   ├── face-api.min.js     # face-api.js library
│   ├── tf.min.js           # TensorFlow.js library
│   └── teachablemachine-image.min.js # Teachable Machine image wrapper
├── models/                 # face-api.js pre-trained weights
│   ├── tiny_face_detector_model-weights_manifest.json
│   ├── tiny_face_detector_model-shard1
│   ├── face_expression_model-weights_manifest.json
│   ├── face_expression_model-shard1
│   ├── face_landmark_68_model-weights_manifest.json
│   └── face_landmark_68_model-shard1
└── mask_model/             # Custom Teachable Machine model
    ├── metadata.json       # Class labels & model metadata
    ├── model.json          # Neural network topology
    └── weights.bin         # Trained model weights
```

---

## 🚀 How to Run Locally

Because the application loads neural network weights (`.json` and binary shard files), modern browsers enforce CORS security policies that block `file:///` URLs. You **must serve the project via a local HTTP server**.

### Option 1: Python (Pre-installed on most systems)
Open your terminal in the project directory and run:
```bash
python -m http.server 8080
```
Then open your browser and navigate to:
```text
http://localhost:8080
```

### Option 2: Node.js `npx serve` or `http-server`
```bash
npx serve .
# or
npx http-server -p 8080
```

### Option 3: VS Code Live Server Extension
1. Open this folder in VS Code.
2. Install the **Live Server** extension.
3. Right-click `index.html` and select **"Open with Live Server"**.

---

## 🧠 How to Train & Export a Custom Mask Model with Google Teachable Machine

If you wish to train your own custom mask detection neural network:

1. **Open Google Teachable Machine**:
   - Go to [https://teachablemachine.withgoogle.com/](https://teachablemachine.withgoogle.com/)
   - Click **Get Started** -> **Image Project** -> **Standard image model** (224x224 px color).
2. **Define Classes**:
   - Rename **Class 1** to `Mask` (or `withmask`).
   - Rename **Class 2** to `No Mask` (or `nomask`).
3. **Collect Samples**:
   - Click **Webcam** (or upload images from a dataset like Kaggle).
   - Record 200–400 images of yourself wearing a mask (surgical, cloth, N95) from different angles.
   - Record 200–400 images without a mask, making various expressions and head poses.
4. **Train Model**:
   - Click **Train Model** (do not switch browser tabs while training).
5. **Export the Model**:
   - Click **Export Model**.
   - Under the **TensorFlow.js** tab, select **Download** (do NOT choose upload/cloud link).
   - Click **Download my model**.
   - Extract the downloaded `.zip` file. You will get:
     - `model.json`
     - `metadata.json`
     - `weights.bin`
6. **Deploy to Project**:
   - Copy these 3 files into the `mask_model/` folder of this project, replacing the existing files.
   - Refresh your browser! The app will immediately use your custom trained neural network.

---

## 🌐 Free Deployment Instructions

### Method 1: Netlify (Easiest & Fastest)
1. Sign up for free at [netlify.com](https://www.netlify.com/).
2. Drag and drop the entire project folder directly into the Netlify **"Deploy manually"** drop zone.
3. Netlify will deploy it instantly with free HTTPS (webcam access strictly requires HTTPS in production!).

### Method 2: Vercel
1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```
2. In the project directory, run:
   ```bash
   vercel
   ```
3. Follow the CLI prompts to deploy.

### Method 3: GitHub Pages
1. Create a new GitHub repository (e.g. `smart-mask-detector`).
2. Push your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Smart Mask & Expression Detector"
   git branch -M main
   git remote add origin https://github.com/<your-username>/smart-mask-detector.git
   git push -u origin main
   ```
3. In GitHub, go to **Settings** -> **Pages**.
4. Under **Branch**, select `main` / `root` and click **Save**.
5. Your app will be live at `https://<your-username>.github.io/smart-mask-detector/`.

---

## 🛡️ License & Credits
- Face detection, landmarks, and facial expression classification powered by [face-api.js](https://github.com/justadudewhohacks/face-api.js).
- Machine learning runtime powered by [TensorFlow.js](https://www.tensorflow.org/js) and [Google Teachable Machine](https://teachablemachine.withgoogle.com/).
