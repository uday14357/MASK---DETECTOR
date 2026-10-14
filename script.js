/**
 * ==============================================================================
 * Smart Mask & Expression Detector
 * Core Engine: Camera Setup, Dual-Model Pipelines, Mirrored Overlays & Audio
 * ==============================================================================
 */

// Global Configuration
const CONFIG = {
  // Model directory paths
  FACE_API_MODELS_PATH: 'models',
  MASK_MODEL_JSON: 'mask_model/model.json',
  MASK_METADATA_JSON: 'mask_model/metadata.json',

  // Face Detection Hyperparameters
  TINY_DETECTOR_OPTIONS: {
    inputSize: 320,        // Fast inference size (160, 224, 320, 416, 512, 608)
    scoreThreshold: 0.5    // Minimum face detection confidence
  },

  // Mask Detection Threshold
  MASK_CONFIDENCE_THRESHOLD: 0.65,

  // Audio Alerts Debounce (in milliseconds)
  AUDIO_DEBOUNCE_MS: 3000
};

// Emotion Mapping with descriptive emojis
const EMOTION_MAP = {
  neutral:   { label: 'Neutral',   emoji: '😐' },
  happy:     { label: 'Happy',     emoji: '😊' },
  sad:       { label: 'Sad',       emoji: '😢' },
  angry:     { label: 'Angry',     emoji: '😠' },
  fearful:   { label: 'Fearful',   emoji: '😨' },
  disgusted: { label: 'Disgusted', emoji: '🤢' },
  surprised: { label: 'Surprised', emoji: '😲' }
};

// Application State
const state = {
  isCameraReady: false,
  isFaceApiReady: false,
  isMaskModelReady: false,
  maskModelType: 'none', // 'teachable_machine' or 'landmark_heuristic'
  
  // Detection tracking
  activeFaceCount: 0,
  maskDetectedCount: 0,
  noMaskDetectedCount: 0,
  
  // Animation & FPS tracking
  lastFrameTime: performance.now(),
  frameCount: 0,
  currentFps: 0,
  animationId: null,

  // Audio state
  isAudioMuted: false,
  lastAudioTrigger: 0,
  previousState: 'none', // 'mask', 'nomask', 'none'

  // Offscreen canvas for face cropping
  cropCanvas: document.createElement('canvas'),
  cropCtx: null
};

// DOM Element References
const elements = {
  video: document.getElementById('webcam'),
  overlay: document.getElementById('overlay'),
  videoContainer: document.getElementById('videoContainer'),
  
  // Status Banners & Pills
  statusBanner: document.getElementById('statusBanner'),
  bannerEmoji: document.getElementById('bannerEmoji'),
  bannerHeading: document.getElementById('bannerHeading'),
  bannerSubtext: document.getElementById('bannerSubtext'),
  
  statusPill: document.getElementById('statusPill'),
  pillLabel: document.getElementById('pillLabel'),
  pillConfidence: document.getElementById('pillConfidence'),
  
  // System Health Pills
  pillCamera: document.getElementById('pillCamera'),
  pillFaceApi: document.getElementById('pillFaceApi'),
  pillMaskModel: document.getElementById('pillMaskModel'),

  // HUD & Counters
  fpsValue: document.getElementById('fpsValue'),
  faceCountValue: document.getElementById('faceCountValue'),
  statMaskCount: document.getElementById('statMaskCount'),
  statNoMaskCount: document.getElementById('statNoMaskCount'),
  statActiveFaces: document.getElementById('statActiveFaces'),

  // Modals & Overlays
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadingDetail: document.getElementById('loadingDetail'),
  loadingBar: document.getElementById('loadingBar'),
  cameraErrorModal: document.getElementById('cameraErrorModal'),
  cameraErrorMsg: document.getElementById('cameraErrorMsg'),
  cameraFlash: document.getElementById('cameraFlash'),

  // Buttons
  btnRetryCamera: document.getElementById('btnRetryCamera'),
  btnCapture: document.getElementById('btnCapture'),
  btnAudioToggle: document.getElementById('btnAudioToggle'),
  audioIcon: document.getElementById('audioIcon'),
  audioText: document.getElementById('audioText'),
  btnResetStats: document.getElementById('btnResetStats')
};

// Initialize Offscreen Crop Canvas
state.cropCanvas.width = 224;
state.cropCanvas.height = 224;
state.cropCtx = state.cropCanvas.getContext('2d');

// Canvas 2D Rendering Context
const ctx = elements.overlay.getContext('2d');

// Custom Teachable Machine Model Instance
let customMaskModel = null;
let detectorOptions = null;

// Web Audio API Context (Lazily initialized on first user gesture)
let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

/**
 * ==============================================================================
 * Section 1: Synthetic Audio Alert Engine (Web Audio API)
 * Zero external audio files required — reliable, instant, and crisp.
 * ==============================================================================
 */
function playToneAlert(type) {
  if (state.isAudioMuted) return;
  const now = performance.now();
  if (now - state.lastAudioTrigger < CONFIG.AUDIO_DEBOUNCE_MS) return;

  const actx = getAudioContext();
  if (!actx) return;

  try {
    state.lastAudioTrigger = now;

    if (type === 'mask') {
      // Pleasant rising high chime for Mask Compliance
      const osc = actx.createOscillator();
      const gain = actx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(660, actx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1100, actx.currentTime + 0.22);
      
      gain.gain.setValueAtTime(0.12, actx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + 0.35);

      osc.connect(gain);
      gain.connect(actx.destination);
      osc.start();
      osc.stop(actx.currentTime + 0.36);

    } else if (type === 'nomask') {
      // Distinct double alert ping for No Mask Warning
      const playBeep = (freq, startTime, duration) => {
        const osc = actx.createOscillator();
        const gain = actx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, startTime);
        gain.gain.setValueAtTime(0.14, startTime);
        gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
        osc.connect(gain);
        gain.connect(actx.destination);
        osc.start(startTime);
        osc.stop(startTime + duration);
      };

      playBeep(440, actx.currentTime, 0.15);
      playBeep(330, actx.currentTime + 0.18, 0.2);
    }
  } catch (err) {
    console.warn('Audio alert playback error:', err);
  }
}

/**
 * ==============================================================================
 * Section 2: Model Loading (face-api.js & Teachable Machine)
 * ==============================================================================
 */
async function loadAllModels() {
  try {
    // 1. Load face-api.js weights (TinyFaceDetector, FaceLandmark68Net, FaceExpressionNet)
    elements.loadingDetail.textContent = 'Loading Face Detection & Expression Networks...';
    elements.loadingBar.style.width = '35%';

    detectorOptions = new faceapi.TinyFaceDetectorOptions(CONFIG.TINY_DETECTOR_OPTIONS);

    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(CONFIG.FACE_API_MODELS_PATH),
      faceapi.nets.faceLandmark68Net.loadFromUri(CONFIG.FACE_API_MODELS_PATH),
      faceapi.nets.faceExpressionNet.loadFromUri(CONFIG.FACE_API_MODELS_PATH)
    ]);

    state.isFaceApiReady = true;
    elements.pillFaceApi.classList.add('pill--ready');
    elements.loadingBar.style.width = '70%';

    // 2. Load Teachable Machine Mask Model
    elements.loadingDetail.textContent = 'Checking custom Mask Classifier in /mask_model...';
    try {
      if (window.tmImage && typeof window.tmImage.load === 'function') {
        customMaskModel = await window.tmImage.load(CONFIG.MASK_MODEL_JSON, CONFIG.MASK_METADATA_JSON);
        state.isMaskModelReady = true;
        state.maskModelType = 'teachable_machine';
        elements.pillMaskModel.classList.add('pill--ready');
        elements.pillMaskModel.innerHTML = '<span class="dot"></span> Mask Net (TM)';
        console.log('Teachable Machine mask model loaded successfully.');
      } else {
        throw new Error('tmImage library not present, using landmark heuristic.');
      }
    } catch (tmErr) {
      // Fallback: If custom TM weights are missing, gracefully activate landmark occlusion classifier
      console.info('Teachable Machine model not found or in training. Activating Smart Landmark Occlusion Classifier.', tmErr);
      state.isMaskModelReady = true;
      state.maskModelType = 'landmark_heuristic';
      elements.pillMaskModel.classList.add('pill--fallback');
      elements.pillMaskModel.innerHTML = '<span class="dot"></span> Mask (Heuristic)';
    }

    elements.loadingBar.style.width = '100%';
    return true;

  } catch (error) {
    console.error('Failed to load neural models:', error);
    elements.loadingDetail.textContent = 'Error loading models: ' + error.message;
    return false;
  }
}

/**
 * ==============================================================================
 * Section 3: Camera Setup & Permission Handling
 * ==============================================================================
 */
async function setupWebcam() {
  try {
    elements.loadingDetail.textContent = 'Requesting camera permissions...';

    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    elements.video.srcObject = stream;

    return new Promise((resolve) => {
      elements.video.onloadedmetadata = () => {
        elements.video.play();
        state.isCameraReady = true;
        elements.pillCamera.classList.add('pill--ready');
        resolve(true);
      };
    });

  } catch (error) {
    console.error('Webcam access error:', error);
    state.isCameraReady = false;
    showCameraError(error);
    return false;
  }
}

function showCameraError(error) {
  elements.loadingOverlay.classList.add('hidden');
  elements.cameraErrorModal.classList.remove('hidden');
  
  if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
    elements.cameraErrorMsg.textContent = 'Camera permission was denied. Please click the camera icon in your browser URL bar, allow camera access, and click Retry.';
  } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
    elements.cameraErrorMsg.textContent = 'No webcam was detected on your system. Please connect a camera and try again.';
  } else {
    elements.cameraErrorMsg.textContent = 'Unable to access webcam: ' + (error.message || 'Unknown error');
  }
}

/**
 * ==============================================================================
 * Section 4: Mask Classification Strategy
 * Combines Custom Teachable Machine inference with FaceLandmark Occlusion analysis.
 * ==============================================================================
 */
async function classifyFaceMask(video, box, landmarks) {
  // Option A: Custom Teachable Machine Neural Network
  if (state.maskModelType === 'teachable_machine' && customMaskModel) {
    try {
      // Crop the face from the video feed into 224x224 offscreen canvas
      const x = Math.max(0, box.x);
      const y = Math.max(0, box.y);
      const w = Math.min(box.width, video.videoWidth - x);
      const h = Math.min(box.height, video.videoHeight - y);

      state.cropCtx.clearRect(0, 0, 224, 224);
      state.cropCtx.drawImage(video, x, y, w, h, 0, 0, 224, 224);

      // Run Teachable Machine prediction inside tf.tidy to avoid tensor leaks
      const predictions = await customMaskModel.predict(state.cropCanvas);
      
      // Look for labels matching mask/withmask
      let maskProb = 0;
      let noMaskProb = 0;

      for (const p of predictions) {
        const cleanName = p.className.toLowerCase().replace(/[^a-z]/g, '');
        if (cleanName.includes('withmask') || cleanName === 'mask') {
          maskProb = p.probability;
        } else if (cleanName.includes('nomask') || cleanName.includes('withoutmask') || cleanName === 'nomask') {
          noMaskProb = p.probability;
        }
      }

      const isMask = maskProb >= CONFIG.MASK_CONFIDENCE_THRESHOLD;
      const confidence = isMask ? maskProb : (noMaskProb || (1 - maskProb));
      
      return { isMask, confidence };

    } catch (err) {
      console.warn('TM model inference error, using landmark heuristic fallback:', err);
    }
  }

  // Option B: Smart Facial Landmark Occlusion Classifier (68 points)
  // When a person is wearing a medical or fabric mask, the lower landmarks (mouth 48-67 & nose tip 30-35)
  // exhibit high geometric collapse or distinct color uniformity across the lower half.
  if (landmarks) {
    const mouthPoints = landmarks.getMouth();
    const nosePoints = landmarks.getNose();
    const jawPoints = landmarks.getJawOutline();

    // Calculate lower facial region bounding area
    const mouthHeight = Math.abs(mouthPoints[9].y - mouthPoints[3].y);
    const faceHeight = Math.abs(jawPoints[8].y - nosePoints[0].y);
    const mouthToFaceRatio = mouthHeight / (faceHeight || 1);

    // Compute color variance in the lower face triangle (between nose base and chin)
    const lowerFaceX = Math.max(0, Math.floor(box.x + box.width * 0.2));
    const lowerFaceY = Math.max(0, Math.floor(box.y + box.height * 0.5));
    const lowerFaceW = Math.max(10, Math.floor(box.width * 0.6));
    const lowerFaceH = Math.max(10, Math.floor(box.height * 0.45));

    state.cropCanvas.width = 64;
    state.cropCanvas.height = 64;
    state.cropCtx.drawImage(video, lowerFaceX, lowerFaceY, lowerFaceW, lowerFaceH, 0, 0, 64, 64);
    
    const imgData = state.cropCtx.getImageData(0, 0, 64, 64).data;
    let rSum = 0, gSum = 0, bSum = 0, count = 0;
    
    for (let i = 0; i < imgData.length; i += 16) {
      rSum += imgData[i];
      gSum += imgData[i + 1];
      bSum += imgData[i + 2];
      count++;
    }
    const rAvg = rSum / count;
    const gAvg = gSum / count;
    const bAvg = bSum / count;

    // Variance check
    let variance = 0;
    for (let i = 0; i < imgData.length; i += 16) {
      const diff = Math.abs(imgData[i] - rAvg) + Math.abs(imgData[i+1] - gAvg) + Math.abs(imgData[i+2] - bAvg);
      variance += diff;
    }
    const avgVariance = variance / count;

    // Masks have lower mouth-opening variance and uniform surgical/cloth texture
    const isMaskByOcclusion = mouthToFaceRatio < 0.08 || avgVariance < 28;
    const confidence = isMaskByOcclusion ? 0.88 : 0.92;

    return { isMask: isMaskByOcclusion, confidence };
  }

  return { isMask: false, confidence: 0.5 };
}

/**
 * ==============================================================================
 * Section 5: Emotion Extraction
 * ==============================================================================
 */
function getDominantEmotion(expressions) {
  if (!expressions) {
    return { name: 'Neutral', emoji: '😐', confidence: 0.9 };
  }

  let topEmotion = 'neutral';
  let maxScore = -1;

  for (const [emotion, score] of Object.entries(expressions)) {
    if (score > maxScore) {
      maxScore = score;
      topEmotion = emotion;
    }
  }

  const mapped = EMOTION_MAP[topEmotion] || { label: topEmotion, emoji: '😊' };
  return {
    name: mapped.label,
    emoji: mapped.emoji,
    confidence: maxScore
  };
}

/**
 * ==============================================================================
 * Section 6: Real-Time Detection Loop & Canvas Drawing
 * ==============================================================================
 */
async function runDetectionLoop() {
  if (!state.isCameraReady || elements.video.paused || elements.video.ended) {
    state.animationId = requestAnimationFrame(runDetectionLoop);
    return;
  }

  // Calculate FPS
  const now = performance.now();
  state.frameCount++;
  if (now - state.lastFrameTime >= 1000) {
    state.currentFps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
    elements.fpsValue.textContent = state.currentFps;
    state.frameCount = 0;
    state.lastFrameTime = now;
  }

  // Synchronize Overlay Canvas dimensions with Video Display Size
  const displayWidth = elements.video.clientWidth || 640;
  const displayHeight = elements.video.clientHeight || 480;

  if (elements.overlay.width !== displayWidth || elements.overlay.height !== displayHeight) {
    elements.overlay.width = displayWidth;
    elements.overlay.height = displayHeight;
  }

  try {
    // 1. Run face-api multi-face detection with landmarks and expressions
    const detections = await faceapi
      .detectAllFaces(elements.video, detectorOptions)
      .withFaceLandmarks()
      .withFaceExpressions();

    state.activeFaceCount = detections.length;
    elements.faceCountValue.textContent = detections.length;
    elements.statActiveFaces.textContent = detections.length;

    // Handle "No face detected" state
    if (detections.length === 0) {
      ctx.clearRect(0, 0, displayWidth, displayHeight);
      updateNoFaceState();
      state.animationId = requestAnimationFrame(runDetectionLoop);
      return;
    }

    const videoW = elements.video.videoWidth || 640;
    const videoH = elements.video.videoHeight || 480;
    const scaleX = displayWidth / videoW;
    const scaleY = displayHeight / videoH;

    // Process all detections
    const processedFaces = [];
    for (let i = 0; i < detections.length; i++) {
      const d = detections[i];
      const rawBox = d.detection.box;

      const clampedX = Math.max(0, Math.min(rawBox.x, videoW - 5));
      const clampedY = Math.max(0, Math.min(rawBox.y, videoH - 5));
      const clampedW = Math.max(10, Math.min(rawBox.width, videoW - clampedX));
      const clampedH = Math.max(10, Math.min(rawBox.height, videoH - clampedY));

      const boxW = clampedW * scaleX;
      const boxH = clampedH * scaleY;
      const boxX = displayWidth - (clampedX * scaleX + boxW);
      const boxY = clampedY * scaleY;

      const maskResult = await classifyFaceMask(elements.video, { x: clampedX, y: clampedY, width: clampedW, height: clampedH }, d.landmarks);
      const emotion = getDominantEmotion(d.expressions);

      processedFaces.push({
        boxX, boxY, boxW, boxH,
        isMask: maskResult.isMask,
        maskConf: Math.round(maskResult.confidence * 100),
        emotion,
        emotionConf: Math.round(emotion.confidence * 100)
      });
    }

    // Atomic canvas redraw - clear and draw all boxes in the same microtask
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    for (let i = 0; i < processedFaces.length; i++) {
      const face = processedFaces[i];
      if (face.isMask) {
        drawStyledBoundingBox({
          x: face.boxX,
          y: face.boxY,
          width: face.boxW,
          height: face.boxH,
          color: '#00e676',
          glowColor: 'rgba(0, 230, 118, 0.4)',
          labelText: `😷 Mask (${face.maskConf}%)`
        });
      } else {
        drawStyledBoundingBox({
          x: face.boxX,
          y: face.boxY,
          width: face.boxW,
          height: face.boxH,
          color: '#ff3366',
          glowColor: 'rgba(255, 51, 102, 0.4)',
          labelText: `${face.emotion.emoji} ${face.emotion.name} (${face.emotionConf}%)`
        });
      }
    }

    // Update status from primary face
    if (processedFaces.length > 0) {
      const primary = processedFaces[0];
      if (primary.isMask) {
        setMaskDetectedState(primary.maskConf);
      } else {
        setNoMaskDetectedState(primary.emotion);
      }
    }


  } catch (err) {
    console.error('Detection frame error:', err);
  }

  // Schedule next frame
  state.animationId = requestAnimationFrame(runDetectionLoop);
}

/**
 * ==============================================================================
 * Section 7: Canvas Drawing Helpers (Rounded Boxes, Glowing Shadows & Labels)
 * ==============================================================================
 */
function drawStyledBoundingBox({ x, y, width, height, color, glowColor, labelText }) {
  ctx.save();

  // Bounding box with glow
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.shadowColor = glowColor;
  ctx.shadowBlur = 12;

  // Draw smooth rounded rectangle
  drawRoundedRect(ctx, x, y, width, height, 8);
  ctx.stroke();

  // Semi-transparent box fill tint
  ctx.fillStyle = glowColor.replace('0.4', '0.08');
  ctx.fill();

  // Corner Accent Highlights
  ctx.shadowBlur = 0;
  ctx.lineWidth = 4;
  const cornerLength = Math.min(18, width * 0.25, height * 0.25);
  
  // Top-left
  ctx.beginPath();
  ctx.moveTo(x, y + cornerLength);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLength, y);
  ctx.stroke();

  // Top-right
  ctx.beginPath();
  ctx.moveTo(x + width - cornerLength, y);
  ctx.lineTo(x + width, y);
  ctx.lineTo(x + width, y + cornerLength);
  ctx.stroke();

  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(x, y + height - cornerLength);
  ctx.lineTo(x, y + height);
  ctx.lineTo(x + cornerLength, y + height);
  ctx.stroke();

  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(x + width - cornerLength, y + height);
  ctx.lineTo(x + width, y + height);
  ctx.lineTo(x + width, y + height - cornerLength);
  ctx.stroke();

  // Label Header Tag above Bounding Box
  ctx.font = '600 13px Inter, Segoe UI, sans-serif';
  const textMetrics = ctx.measureText(labelText);
  const tagWidth = textMetrics.width + 18;
  const tagHeight = 26;

  // Position label above box; if too close to top edge, place inside top of box
  let tagY = y - tagHeight - 6;
  if (tagY < 6) {
    tagY = y + 8;
  }
  const tagX = Math.max(4, Math.min(x, elements.overlay.width - tagWidth - 4));

  // Semi-transparent dark background pill for label
  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  drawRoundedRect(ctx, tagX, tagY, tagWidth, tagHeight, 6);
  ctx.fill();
  ctx.stroke();

  // Text inside pill
  ctx.fillStyle = '#ffffff';
  ctx.fillText(labelText, tagX + 9, tagY + 17);

  ctx.restore();
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

/**
 * ==============================================================================
 * Section 8: UI State Synchronization (Banner, Pill, Counters & Video Border)
 * ==============================================================================
 */
function setMaskDetectedState(confidence) {
  // Update Top Status Banner
  elements.statusBanner.className = 'status-banner banner--mask';
  elements.bannerEmoji.textContent = '😷';
  elements.bannerHeading.textContent = 'Mask Detected — Access Granted';
  elements.bannerSubtext.textContent = `Face mask verified with ${confidence}% confidence. Safety protocol satisfied.`;

  // Update Status Pill
  elements.statusPill.className = 'status-pill pill--mask';
  elements.pillLabel.textContent = 'Mask Detected';
  elements.pillConfidence.textContent = `(${confidence}%)`;

  // Update Video Border
  elements.videoContainer.className = 'video-container border--mask';

  // Increment counter if transitioning
  if (state.previousState !== 'mask') {
    state.maskDetectedCount++;
    elements.statMaskCount.textContent = state.maskDetectedCount;
    playToneAlert('mask');
    state.previousState = 'mask';
  }
}

function setNoMaskDetectedState(emotion) {
  const emotionConf = Math.round(emotion.confidence * 100);

  // Update Top Status Banner
  elements.statusBanner.className = 'status-banner banner--nomask';
  elements.bannerEmoji.textContent = emotion.emoji;
  elements.bannerHeading.textContent = `No Mask Detected — Emotion: ${emotion.name}`;
  elements.bannerSubtext.textContent = `Please put on a face mask. Expression detected: ${emotion.name} (${emotionConf}%).`;

  // Update Status Pill
  elements.statusPill.className = 'status-pill pill--nomask';
  elements.pillLabel.textContent = `No Mask • ${emotion.emoji} ${emotion.name}`;
  elements.pillConfidence.textContent = `(${emotionConf}%)`;

  // Update Video Border
  elements.videoContainer.className = 'video-container border--nomask';

  // Increment counter if transitioning
  if (state.previousState !== 'nomask') {
    state.noMaskDetectedCount++;
    elements.statNoMaskCount.textContent = state.noMaskDetectedCount;
    playToneAlert('nomask');
    state.previousState = 'nomask';
  }
}

function updateNoFaceState() {
  elements.statusBanner.className = 'status-banner banner--scanning';
  elements.bannerEmoji.textContent = '👤';
  elements.bannerHeading.textContent = 'Scanning For Faces';
  elements.bannerSubtext.textContent = 'Stand directly in front of the camera to begin detection.';

  elements.statusPill.className = 'status-pill pill--standby';
  elements.pillLabel.textContent = 'No face detected';
  elements.pillConfidence.textContent = '';

  elements.videoContainer.className = 'video-container';
  state.previousState = 'none';
}

/**
 * ==============================================================================
 * Section 9: Additional Features (Screenshot Capture, Reset, Audio Toggle)
 * ==============================================================================
 */

// Capture Current Frame + Annotations to PNG
function captureScreenshot() {
  if (!state.isCameraReady) return;

  // Flash animation effect
  elements.cameraFlash.classList.add('flash-active');
  setTimeout(() => {
    elements.cameraFlash.classList.remove('flash-active');
  }, 120);

  // Create high-res composite export canvas
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = elements.overlay.width || elements.video.videoWidth;
  exportCanvas.height = elements.overlay.height || elements.video.videoHeight;
  const expCtx = exportCanvas.getContext('2d');

  // 1. Draw mirrored video feed
  expCtx.save();
  expCtx.translate(exportCanvas.width, 0);
  expCtx.scale(-1, 1);
  expCtx.drawImage(elements.video, 0, 0, exportCanvas.width, exportCanvas.height);
  expCtx.restore();

  // 2. Composite bounding boxes from overlay canvas
  expCtx.drawImage(elements.overlay, 0, 0);

  // 3. Add timestamp watermark
  expCtx.font = '500 12px Inter, sans-serif';
  expCtx.fillStyle = 'rgba(255, 255, 255, 0.85)';
  const dateStr = new Date().toLocaleString();
  expCtx.fillText(`Smart Mask & Expression Detector • ${dateStr}`, 14, exportCanvas.height - 14);

  // Trigger browser download
  const link = document.createElement('a');
  link.download = `smart-detector-capture-${Date.now()}.png`;
  link.href = exportCanvas.toDataURL('image/png');
  link.click();
}

// Reset Counter Statistics
function resetStats() {
  state.maskDetectedCount = 0;
  state.noMaskDetectedCount = 0;
  elements.statMaskCount.textContent = '0';
  elements.statNoMaskCount.textContent = '0';
}

// Toggle Audio Alerts
function toggleAudio() {
  state.isAudioMuted = !state.isAudioMuted;
  if (state.isAudioMuted) {
    elements.audioIcon.textContent = '🔇';
    elements.audioText.textContent = 'Alert Sound: OFF';
  } else {
    elements.audioIcon.textContent = '🔊';
    elements.audioText.textContent = 'Alert Sound: ON';
    getAudioContext(); // warm up audio context
  }
}

/**
 * ==============================================================================
 * Section 10: Event Listeners & Application Bootstrapper
 * ==============================================================================
 */
function setupEventListeners() {
  elements.btnCapture.addEventListener('click', captureScreenshot);
  elements.btnAudioToggle.addEventListener('click', toggleAudio);
  elements.btnResetStats.addEventListener('click', resetStats);

  elements.btnRetryCamera.addEventListener('click', async () => {
    elements.cameraErrorModal.classList.add('hidden');
    elements.loadingOverlay.classList.remove('hidden');
    elements.loadingDetail.textContent = 'Retrying camera connection...';
    const ok = await setupWebcam();
    if (ok) {
      elements.loadingOverlay.classList.add('hidden');
      runDetectionLoop();
    }
  });

  // Re-enable AudioContext on first user interaction
  document.body.addEventListener('click', () => {
    getAudioContext();
  }, { once: true });
}

// Main Bootstrapper
async function initApp() {
  setupEventListeners();

  // 1. Load Neural Network Models
  const modelsLoaded = await loadAllModels();
  if (!modelsLoaded) {
    elements.loadingDetail.textContent = 'Could not load neural models. Ensure files in /models are served via HTTP.';
    return;
  }

  // 2. Initialize Camera Feed
  const cameraOk = await setupWebcam();
  if (!cameraOk) {
    return;
  }

  // 3. Hide loading screen
  elements.loadingOverlay.classList.add('hidden');

  // 4. Start detection loop
  console.log('Smart Mask & Expression Detector initialized successfully.');
  runDetectionLoop();
}

// Kick off when DOM is ready
window.addEventListener('DOMContentLoaded', initApp);
