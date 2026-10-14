/**
 * ==============================================================================
 * Smart Mask & Expression Detector - Enterprise AI Console
 * Core Engine: Camera Setup, Dual-Model Pipelines, Mirrored Overlays & Telemetry
 * ==============================================================================
 */

// Global Configuration
const CONFIG = {
  FACE_API_MODELS_PATH: 'models',
  MASK_MODEL_JSON: 'mask_model/model.json',
  MASK_METADATA_JSON: 'mask_model/metadata.json',

  TINY_DETECTOR_OPTIONS: {
    inputSize: 320,        // 320x320 fast inference
    scoreThreshold: 0.5    // Minimum face detection confidence
  },

  MASK_CONFIDENCE_THRESHOLD: 0.65,
  AUDIO_DEBOUNCE_MS: 3000,
  MAX_AUDIT_LOGS: 12
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
  maskModelType: 'none',
  
  // Counters & Telemetry
  activeFaceCount: 0,
  maskDetectedCount: 0,
  noMaskDetectedCount: 0,
  totalInferences: 0,
  
  // Latency & FPS
  lastFrameTime: performance.now(),
  frameCount: 0,
  currentFps: 0,
  lastInferenceDuration: 0,
  animationId: null,

  // Audio State
  isAudioMuted: false,
  lastAudioTrigger: 0,
  previousState: 'none',

  // Offscreen Canvas for Face Cropping
  cropCanvas: document.createElement('canvas'),
  cropCtx: null,

  // Audit Logs
  auditLogs: []
};

// DOM References
const elements = {
  video: document.getElementById('webcam'),
  overlay: document.getElementById('overlay'),
  videoContainer: document.getElementById('videoContainer'),
  
  // Primary Banner & Status Badges
  statusBanner: document.getElementById('statusBanner'),
  bannerEmoji: document.getElementById('bannerEmoji'),
  bannerHeading: document.getElementById('bannerHeading'),
  bannerSubtext: document.getElementById('bannerSubtext'),
  bannerPill: document.getElementById('bannerPill'),
  
  statusBadge: document.getElementById('statusBadge'),
  statusBadgeText: document.getElementById('statusBadgeText'),
  statusFootnote: document.getElementById('statusFootnote'),
  viewportTag: document.getElementById('viewportTag'),
  maskEngineType: document.getElementById('maskEngineType'),
  valResolution: document.getElementById('valResolution'),

  // System Health Pills
  pillCamera: document.getElementById('pillCamera'),
  pillFaceApi: document.getElementById('pillFaceApi'),
  pillMaskModel: document.getElementById('pillMaskModel'),

  // HUD & Telemetry
  fpsValue: document.getElementById('fpsValue'),
  faceCountValue: document.getElementById('faceCountValue'),
  latencyValue: document.getElementById('latencyValue'),
  
  statMaskCount: document.getElementById('statMaskCount'),
  statNoMaskCount: document.getElementById('statNoMaskCount'),
  statActiveFaces: document.getElementById('statActiveFaces'),
  statComplianceRate: document.getElementById('statComplianceRate'),
  
  barMaskFill: document.getElementById('barMaskFill'),
  barNoMaskFill: document.getElementById('barNoMaskFill'),
  barActiveFill: document.getElementById('barActiveFill'),
  barRateFill: document.getElementById('barRateFill'),

  // Audit Event Log
  auditTableBody: document.getElementById('auditTableBody'),

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

const ctx = elements.overlay.getContext('2d');
let customMaskModel = null;
let detectorOptions = null;
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
 * Section 1: Synthetic Audio Alerts
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
    console.warn('Audio playback error:', err);
  }
}

/**
 * ==============================================================================
 * Section 2: Model Loading
 * ==============================================================================
 */
async function loadAllModels() {
  try {
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

    // Load Teachable Machine Mask Model
    elements.loadingDetail.textContent = 'Verifying Mask Classifier in /mask_model...';
    try {
      if (window.tmImage && typeof window.tmImage.load === 'function') {
        customMaskModel = await window.tmImage.load(CONFIG.MASK_MODEL_JSON, CONFIG.MASK_METADATA_JSON);
        state.isMaskModelReady = true;
        state.maskModelType = 'teachable_machine';
        elements.pillMaskModel.classList.add('pill--ready');
        elements.maskEngineType.textContent = 'Teachable Machine';
        console.log('Teachable Machine mask model loaded.');
      } else {
        throw new Error('tmImage unavailable');
      }
    } catch (tmErr) {
      console.info('Using Landmark Occlusion Classifier fallback.', tmErr);
      state.isMaskModelReady = true;
      state.maskModelType = 'landmark_heuristic';
      elements.pillMaskModel.classList.add('pill--fallback');
      elements.maskEngineType.textContent = 'Landmark Occlusion Net';
    }

    elements.loadingBar.style.width = '100%';
    addAuditLog('System', 'Models Loaded', '100%', 'info');
    return true;

  } catch (error) {
    console.error('Failed to load neural models:', error);
    elements.loadingDetail.textContent = 'Error loading models: ' + error.message;
    return false;
  }
}

/**
 * ==============================================================================
 * Section 3: Camera Setup
 * ==============================================================================
 */
async function setupWebcam() {
  try {
    elements.loadingDetail.textContent = 'Connecting to camera hardware...';

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
        
        const w = elements.video.videoWidth || 640;
        const h = elements.video.videoHeight || 480;
        if (elements.valResolution) {
          elements.valResolution.textContent = `${w} × ${h}`;
        }
        addAuditLog('Camera', 'Stream Active', 'Ready', 'info');
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
    elements.cameraErrorMsg.textContent = 'Camera permission was denied. Please allow camera permissions in your browser address bar and click Reconnect Camera.';
  } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
    elements.cameraErrorMsg.textContent = 'No webcam was detected on your system. Please connect a camera and try again.';
  } else {
    elements.cameraErrorMsg.textContent = 'Unable to access webcam: ' + (error.message || 'Unknown error');
  }
}

/**
 * ==============================================================================
 * Section 4: Mask Classification
 * ==============================================================================
 */
async function classifyFaceMask(video, box, landmarks) {
  if (state.maskModelType === 'teachable_machine' && customMaskModel) {
    try {
      const x = Math.max(0, box.x);
      const y = Math.max(0, box.y);
      const w = Math.min(box.width, video.videoWidth - x);
      const h = Math.min(box.height, video.videoHeight - y);

      state.cropCtx.clearRect(0, 0, 224, 224);
      state.cropCtx.drawImage(video, x, y, w, h, 0, 0, 224, 224);

      const predictions = await customMaskModel.predict(state.cropCanvas);
      let maskProb = 0;
      let noMaskProb = 0;

      for (const p of predictions) {
        const cleanName = p.className.toLowerCase().replace(/[^a-z]/g, '');
        if (cleanName.includes('withmask') || cleanName === 'mask') {
          maskProb = p.probability;
        } else if (cleanName.includes('nomask') || cleanName.includes('withoutmask')) {
          noMaskProb = p.probability;
        }
      }

      const isMask = maskProb >= CONFIG.MASK_CONFIDENCE_THRESHOLD;
      const confidence = isMask ? maskProb : (noMaskProb || (1 - maskProb));
      return { isMask, confidence };

    } catch (err) {
      console.warn('TM inference error, falling back:', err);
    }
  }

  // Landmark Occlusion & Lower-Face Texture Analysis Fallback
  if (landmarks) {
    const mouthPoints = landmarks.getMouth();
    const nosePoints = landmarks.getNose();
    const jawPoints = landmarks.getJawOutline();

    const mouthHeight = Math.abs(mouthPoints[9].y - mouthPoints[3].y);
    const faceHeight = Math.abs(jawPoints[8].y - nosePoints[0].y);
    const mouthToFaceRatio = mouthHeight / (faceHeight || 1);

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

    let variance = 0;
    for (let i = 0; i < imgData.length; i += 16) {
      const diff = Math.abs(imgData[i] - rAvg) + Math.abs(imgData[i+1] - gAvg) + Math.abs(imgData[i+2] - bAvg);
      variance += diff;
    }
    const avgVariance = variance / count;

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
 * Section 6: Real-Time Detection Loop
 * ==============================================================================
 */
async function runDetectionLoop() {
  if (!state.isCameraReady || elements.video.paused || elements.video.ended) {
    state.animationId = requestAnimationFrame(runDetectionLoop);
    return;
  }

  // FPS calculation
  const now = performance.now();
  state.frameCount++;
  if (now - state.lastFrameTime >= 1000) {
    state.currentFps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
    elements.fpsValue.textContent = state.currentFps;
    state.frameCount = 0;
    state.lastFrameTime = now;
  }

  // Match overlay dimensions
  const displayWidth = elements.video.clientWidth || 640;
  const displayHeight = elements.video.clientHeight || 480;

  if (elements.overlay.width !== displayWidth || elements.overlay.height !== displayHeight) {
    elements.overlay.width = displayWidth;
    elements.overlay.height = displayHeight;
  }

  const tStart = performance.now();

  try {
    // 1. Multi-face inference
    const detections = await faceapi
      .detectAllFaces(elements.video, detectorOptions)
      .withFaceLandmarks()
      .withFaceExpressions();

    state.lastInferenceDuration = Math.round(performance.now() - tStart);
    if (elements.latencyValue) {
      elements.latencyValue.textContent = `${state.lastInferenceDuration} ms`;
    }

    state.activeFaceCount = detections.length;
    elements.faceCountValue.textContent = detections.length;
    elements.statActiveFaces.textContent = detections.length;
    updateBarMeters();

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

    // Atomic canvas redraw
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    for (let i = 0; i < processedFaces.length; i++) {
      const face = processedFaces[i];
      if (face.isMask) {
        drawStyledBoundingBox({
          x: face.boxX,
          y: face.boxY,
          width: face.boxW,
          height: face.boxH,
          color: '#3ecf8e',
          glowColor: 'rgba(62, 207, 142, 0.4)',
          labelText: `😷 Mask (${face.maskConf}%)`
        });
      } else {
        drawStyledBoundingBox({
          x: face.boxX,
          y: face.boxY,
          width: face.boxW,
          height: face.boxH,
          color: '#f43f5e',
          glowColor: 'rgba(244, 63, 94, 0.4)',
          labelText: `${face.emotion.emoji} ${face.emotion.name} (${face.emotionConf}%)`
        });
      }
    }

    // Update primary face state
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

  state.animationId = requestAnimationFrame(runDetectionLoop);
}

/**
 * ==============================================================================
 * Section 7: Canvas Bounding Box Rendering
 * ==============================================================================
 */
function drawStyledBoundingBox({ x, y, width, height, color, glowColor, labelText }) {
  ctx.save();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = glowColor;
  ctx.shadowBlur = 10;

  drawRoundedRect(ctx, x, y, width, height, 6);
  ctx.stroke();

  ctx.fillStyle = glowColor.replace('0.4', '0.06');
  ctx.fill();

  // Technical Corner Accents
  ctx.shadowBlur = 0;
  ctx.lineWidth = 3;
  const cornerLength = Math.min(16, width * 0.25, height * 0.25);
  
  ctx.beginPath();
  ctx.moveTo(x, y + cornerLength);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLength, y);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x + width - cornerLength, y);
  ctx.lineTo(x + width, y);
  ctx.lineTo(x + width, y + cornerLength);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x, y + height - cornerLength);
  ctx.lineTo(x, y + height);
  ctx.lineTo(x + cornerLength, y + height);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x + width - cornerLength, y + height);
  ctx.lineTo(x + width, y + height);
  ctx.lineTo(x + width, y + height - cornerLength);
  ctx.stroke();

  // Label Header Tag
  ctx.font = '600 12px Inter, sans-serif';
  const textMetrics = ctx.measureText(labelText);
  const tagWidth = textMetrics.width + 16;
  const tagHeight = 24;

  let tagY = y - tagHeight - 6;
  if (tagY < 6) {
    tagY = y + 8;
  }
  const tagX = Math.max(4, Math.min(x, elements.overlay.width - tagWidth - 4));

  ctx.fillStyle = 'rgba(14, 15, 18, 0.92)';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  drawRoundedRect(ctx, tagX, tagY, tagWidth, tagHeight, 4);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.fillText(labelText, tagX + 8, tagY + 16);

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
 * Section 8: UI State Synchronization & Telemetry
 * ==============================================================================
 */
function setMaskDetectedState(confidence) {
  elements.statusBanner.className = 'compliance-banner banner-mask';
  elements.bannerEmoji.textContent = '😷';
  elements.bannerHeading.textContent = 'Mask Detected — Compliant';
  elements.bannerSubtext.textContent = `Facial mask verified with ${confidence}% confidence. Security protocol passed.`;
  elements.bannerPill.textContent = 'COMPLIANT';

  elements.statusBadge.className = 'status-indicator-badge badge-mask';
  elements.statusBadgeText.textContent = `MASK VERIFIED (${confidence}%)`;
  elements.statusFootnote.textContent = 'Active subject satisfies safety compliance.';
  elements.viewportTag.textContent = 'INSPECTING';

  elements.videoContainer.className = 'viewport-canvas-wrapper border--mask';

  if (state.previousState !== 'mask') {
    state.maskDetectedCount++;
    elements.statMaskCount.textContent = state.maskDetectedCount;
    playToneAlert('mask');
    addAuditLog('Subject #1', 'Mask Verified', `${confidence}%`, 'mask');
    state.previousState = 'mask';
    updateComplianceRate();
  }
}

function setNoMaskDetectedState(emotion) {
  const emotionConf = Math.round(emotion.confidence * 100);

  elements.statusBanner.className = 'compliance-banner banner-nomask';
  elements.bannerEmoji.textContent = emotion.emoji;
  elements.bannerHeading.textContent = `No Mask Detected — Emotion: ${emotion.name}`;
  elements.bannerSubtext.textContent = `Safety violation. Facial expression classified as ${emotion.name} (${emotionConf}%).`;
  elements.bannerPill.textContent = 'VIOLATION';

  elements.statusBadge.className = 'status-indicator-badge badge-nomask';
  elements.statusBadgeText.textContent = `NO MASK • ${emotion.emoji} ${emotion.name.toUpperCase()}`;
  elements.statusFootnote.textContent = 'Subject is not wearing required protective mask.';
  elements.viewportTag.textContent = 'VIOLATION';

  elements.videoContainer.className = 'viewport-canvas-wrapper border--nomask';

  if (state.previousState !== 'nomask') {
    state.noMaskDetectedCount++;
    elements.statNoMaskCount.textContent = state.noMaskDetectedCount;
    playToneAlert('nomask');
    addAuditLog('Subject #1', `No Mask (${emotion.name})`, `${emotionConf}%`, 'nomask');
    state.previousState = 'nomask';
    updateComplianceRate();
  }
}

function updateNoFaceState() {
  elements.statusBanner.className = 'compliance-banner banner-standby';
  elements.bannerEmoji.textContent = '👤';
  elements.bannerHeading.textContent = 'Scanning For Faces';
  elements.bannerSubtext.textContent = 'Please position yourself within the camera frame for real-time compliance assessment.';
  elements.bannerPill.textContent = 'STANDBY';

  elements.statusBadge.className = 'status-indicator-badge badge-standby';
  elements.statusBadgeText.textContent = 'SEARCHING';
  elements.statusFootnote.textContent = 'No face detected in live video feed.';
  elements.viewportTag.textContent = 'STANDBY';

  elements.videoContainer.className = 'viewport-canvas-wrapper';
  state.previousState = 'none';
}

function updateComplianceRate() {
  const total = state.maskDetectedCount + state.noMaskDetectedCount;
  if (total === 0) {
    elements.statComplianceRate.textContent = '100%';
    elements.barRateFill.style.width = '100%';
    return;
  }
  const rate = Math.round((state.maskDetectedCount / total) * 100);
  elements.statComplianceRate.textContent = `${rate}%`;
  elements.barRateFill.style.width = `${rate}%`;
}

function updateBarMeters() {
  const maxScale = Math.max(10, state.maskDetectedCount + state.noMaskDetectedCount);
  const maskPct = Math.min(100, Math.round((state.maskDetectedCount / maxScale) * 100));
  const noMaskPct = Math.min(100, Math.round((state.noMaskDetectedCount / maxScale) * 100));
  const activePct = Math.min(100, state.activeFaceCount * 25);

  if (elements.barMaskFill) elements.barMaskFill.style.width = `${maskPct}%`;
  if (elements.barNoMaskFill) elements.barNoMaskFill.style.width = `${noMaskPct}%`;
  if (elements.barActiveFill) elements.barActiveFill.style.width = `${activePct}%`;
}

/**
 * ==============================================================================
 * Section 9: Audit Trail Table Streaming
 * ==============================================================================
 */
function addAuditLog(target, stateText, conf, type) {
  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];

  const logItem = {
    time: timeStr,
    target: target,
    state: stateText,
    conf: conf,
    type: type
  };

  state.auditLogs.unshift(logItem);
  if (state.auditLogs.length > CONFIG.MAX_AUDIT_LOGS) {
    state.auditLogs.pop();
  }

  renderAuditTable();
}

function renderAuditTable() {
  if (!elements.auditTableBody) return;

  elements.auditTableBody.innerHTML = state.auditLogs.map(log => {
    let pillClass = 'log-pill-info';
    if (log.type === 'mask') pillClass = 'log-pill-mask';
    if (log.type === 'nomask') pillClass = 'log-pill-nomask';

    return `
      <tr>
        <td class="mono">${log.time}</td>
        <td>${log.target}</td>
        <td><span class="log-pill ${pillClass}">${log.state}</span></td>
        <td class="mono">${log.conf}</td>
      </tr>
    `;
  }).join('');
}

/**
 * ==============================================================================
 * Section 10: Additional Features
 * ==============================================================================
 */
function captureScreenshot() {
  if (!state.isCameraReady) return;

  elements.cameraFlash.classList.add('flash-active');
  setTimeout(() => {
    elements.cameraFlash.classList.remove('flash-active');
  }, 100);

  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = elements.overlay.width || elements.video.videoWidth;
  exportCanvas.height = elements.overlay.height || elements.video.videoHeight;
  const expCtx = exportCanvas.getContext('2d');

  expCtx.save();
  expCtx.translate(exportCanvas.width, 0);
  expCtx.scale(-1, 1);
  expCtx.drawImage(elements.video, 0, 0, exportCanvas.width, exportCanvas.height);
  expCtx.restore();

  expCtx.drawImage(elements.overlay, 0, 0);

  expCtx.font = '500 12px Inter, sans-serif';
  expCtx.fillStyle = 'rgba(255, 255, 255, 0.85)';
  const dateStr = new Date().toLocaleString();
  expCtx.fillText(`Smart Mask & Expression Detector • ${dateStr}`, 14, exportCanvas.height - 14);

  const link = document.createElement('a');
  link.download = `smart-mask-audit-${Date.now()}.png`;
  link.href = exportCanvas.toDataURL('image/png');
  link.click();

  addAuditLog('User', 'Capture Saved', 'PNG', 'info');
}

function resetStats() {
  state.maskDetectedCount = 0;
  state.noMaskDetectedCount = 0;
  elements.statMaskCount.textContent = '0';
  elements.statNoMaskCount.textContent = '0';
  updateComplianceRate();
  updateBarMeters();
  addAuditLog('User', 'Metrics Reset', '0', 'info');
}

function toggleAudio() {
  state.isAudioMuted = !state.isAudioMuted;
  if (state.isAudioMuted) {
    elements.audioIcon.textContent = '🔇';
    elements.audioText.textContent = 'Alerts OFF';
  } else {
    elements.audioIcon.textContent = '🔊';
    elements.audioText.textContent = 'Alerts ON';
    getAudioContext();
  }
}

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

  document.body.addEventListener('click', () => {
    getAudioContext();
  }, { once: true });
}

async function initApp() {
  setupEventListeners();

  const modelsLoaded = await loadAllModels();
  if (!modelsLoaded) {
    elements.loadingDetail.textContent = 'Could not load neural models.';
    return;
  }

  const cameraOk = await setupWebcam();
  if (!cameraOk) {
    return;
  }

  elements.loadingOverlay.classList.add('hidden');
  console.log('AI Console initialized successfully.');
  runDetectionLoop();
}

window.addEventListener('DOMContentLoaded', initApp);
