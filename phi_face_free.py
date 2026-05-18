import os
import io
import base64
import math
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
PHI = 1.6180339887

# Load cascades once at startup
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phi Face Analyzer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=Josefin+Sans:wght@100;300;400&display=swap');
  :root {
    --gold: #C9A84C; --gold-light: #E8C97A; --gold-dim: #8A6E2F;
    --dark: #0A0A0A; --dark2: #111111; --dark3: #1A1A1A;
    --text: #E8E0D0; --text-dim: #8A8070;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--dark); color:var(--text); font-family:'Josefin Sans',sans-serif; font-weight:300; min-height:100vh; }
  body::before {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(201,168,76,0.08) 0%, transparent 60%);
  }
  .container { position:relative; z-index:1; max-width:880px; margin:0 auto; padding:60px 24px 80px; }

  /* Header */
  .header { text-align:center; margin-bottom:56px; }
  .phi-symbol {
    font-family:'Cormorant Garamond',serif; font-size:88px; font-weight:300;
    color:var(--gold); line-height:1; display:block;
    animation: glow 4s ease-in-out infinite;
  }
  @keyframes glow {
    0%,100% { text-shadow:0 0 30px rgba(201,168,76,0.3); }
    50%      { text-shadow:0 0 70px rgba(201,168,76,0.6),0 0 120px rgba(201,168,76,0.2); }
  }
  .title { font-size:10px; letter-spacing:8px; text-transform:uppercase; color:var(--text-dim); margin-top:12px; }
  .subtitle { font-family:'Cormorant Garamond',serif; font-size:22px; font-weight:300; color:var(--text); margin-top:14px; opacity:.85; }
  .divider { display:flex; align-items:center; gap:16px; margin:28px auto; max-width:280px; }
  .divider-line { flex:1; height:1px; background:linear-gradient(to right,transparent,var(--gold-dim)); }
  .divider-line:last-child { background:linear-gradient(to left,transparent,var(--gold-dim)); }
  .divider-dot { width:6px; height:6px; background:var(--gold); transform:rotate(45deg); box-shadow:0 0 10px rgba(201,168,76,.5); }
  .desc { font-family:'Cormorant Garamond',serif; font-size:15px; color:var(--text-dim); line-height:1.75; max-width:500px; margin:0 auto; }

  /* Badge */
  .free-badge {
    display:inline-block; margin-top:18px;
    border:1px solid var(--gold-dim); padding:5px 18px;
    font-size:9px; letter-spacing:4px; text-transform:uppercase; color:var(--gold-dim);
  }

  /* Upload */
  .upload-zone {
    border:1px solid var(--gold-dim); border-radius:2px; padding:56px 40px;
    text-align:center; cursor:pointer; background:linear-gradient(135deg,var(--dark2),var(--dark3));
    transition:all .4s; position:relative; overflow:hidden;
  }
  .upload-zone::before {
    content:''; position:absolute; inset:0;
    background:linear-gradient(135deg,rgba(201,168,76,.06) 0%,transparent 60%);
    opacity:0; transition:opacity .4s;
  }
  .upload-zone:hover::before { opacity:1; }
  .upload-zone:hover, .upload-zone.drag-over { border-color:var(--gold); }
  .upload-icon { font-size:44px; display:block; margin-bottom:18px; opacity:.55; }
  .upload-text { font-size:10px; letter-spacing:4px; text-transform:uppercase; color:var(--gold); display:block; margin-bottom:8px; }
  .upload-hint { font-family:'Cormorant Garamond',serif; font-size:16px; color:var(--text-dim); }
  #fileInput { display:none; }

  /* Preview */
  .preview-wrap { display:none; border:1px solid var(--gold-dim); border-radius:2px; overflow:hidden; background:var(--dark2); position:relative; }
  .preview-wrap img { width:100%; max-height:440px; object-fit:contain; display:block; }
  .preview-overlay { position:absolute; top:12px; right:12px; }
  .btn-remove {
    background:rgba(10,10,10,.85); border:1px solid var(--gold-dim); color:var(--text-dim);
    padding:6px 14px; font-family:'Josefin Sans',sans-serif; font-size:9px;
    letter-spacing:3px; text-transform:uppercase; cursor:pointer; transition:all .3s;
  }
  .btn-remove:hover { border-color:var(--gold); color:var(--gold); }

  /* Analyze btn */
  .analyze-btn {
    width:100%; margin-top:20px; padding:20px; background:transparent;
    border:1px solid var(--gold); color:var(--gold);
    font-family:'Josefin Sans',sans-serif; font-size:11px; font-weight:400;
    letter-spacing:6px; text-transform:uppercase; cursor:pointer;
    position:relative; overflow:hidden; transition:all .4s; display:none;
  }
  .analyze-btn::before {
    content:''; position:absolute; inset:0;
    background:linear-gradient(135deg,var(--gold),var(--gold-light));
    transform:translateX(-100%); transition:transform .4s; z-index:0;
  }
  .analyze-btn:hover::before { transform:translateX(0); }
  .analyze-btn:hover { color:var(--dark); }
  .analyze-btn span { position:relative; z-index:1; }
  .analyze-btn:disabled { opacity:.45; cursor:not-allowed; }
  .analyze-btn:disabled::before { display:none; }

  /* Loading */
  .loading { display:none; text-align:center; padding:48px; }
  .loading-ring {
    width:56px; height:56px; border:1px solid var(--dark3);
    border-top-color:var(--gold); border-radius:50%;
    animation:spin 1.2s linear infinite; margin:0 auto 22px;
  }
  @keyframes spin { to { transform:rotate(360deg); } }
  .loading-text { font-size:9px; letter-spacing:5px; text-transform:uppercase; color:var(--text-dim); animation:pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:1} }

  /* Results */
  .results { display:none; animation:fadeUp .8s ease forwards; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

  /* Score */
  .score-section {
    text-align:center; padding:48px 24px;
    background:linear-gradient(135deg,var(--dark2),var(--dark3));
    border:1px solid var(--gold-dim); margin-bottom:28px; position:relative; overflow:hidden;
  }
  .score-section::before {
    content:'φ'; position:absolute; font-family:'Cormorant Garamond',serif;
    font-size:220px; color:rgba(201,168,76,.03);
    top:50%; left:50%; transform:translate(-50%,-50%); pointer-events:none;
  }
  .score-label { font-size:9px; letter-spacing:5px; text-transform:uppercase; color:var(--text-dim); display:block; margin-bottom:14px; }
  .score-number { font-family:'Cormorant Garamond',serif; font-size:100px; font-weight:300; line-height:1; color:var(--gold); display:inline; }
  .score-unit { font-family:'Cormorant Garamond',serif; font-size:36px; font-weight:300; color:var(--gold-dim); }
  .score-verdict { font-family:'Cormorant Garamond',serif; font-size:20px; font-weight:300; color:var(--text); margin-top:14px; font-style:italic; }
  .score-bar-wrap { margin-top:26px; }
  .score-bar-track { height:2px; background:var(--dark3); border-radius:2px; }
  .score-bar-fill {
    height:100%; background:linear-gradient(to right,var(--gold-dim),var(--gold),var(--gold-light));
    border-radius:2px; width:0%; transition:width 1.6s cubic-bezier(.4,0,.2,1);
  }

  /* Annotated image */
  .annotated-wrap {
    border:1px solid var(--gold-dim); margin-bottom:28px; background:var(--dark2);
    position:relative; overflow:hidden;
  }
  .annotated-wrap img { width:100%; display:block; }
  .annotated-label {
    font-size:9px; letter-spacing:4px; text-transform:uppercase; color:var(--gold-dim);
    padding:10px 16px; border-top:1px solid var(--dark3); text-align:center;
  }

  /* Ratios */
  .section-title {
    font-size:9px; letter-spacing:5px; text-transform:uppercase; color:var(--text-dim);
    margin-bottom:18px; display:flex; align-items:center; gap:12px;
  }
  .section-title::after { content:''; flex:1; height:1px; background:var(--dark3); }

  .ratios-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(255px,1fr)); gap:12px; margin-bottom:28px; }
  .ratio-card {
    background:var(--dark2); border:1px solid var(--dark3); padding:18px; transition:border-color .3s;
  }
  .ratio-card:hover { border-color:var(--gold-dim); }
  .ratio-name { font-size:9px; letter-spacing:3px; text-transform:uppercase; color:var(--text-dim); margin-bottom:10px; }
  .ratio-row { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; }
  .ratio-score { font-family:'Cormorant Garamond',serif; font-size:30px; color:var(--gold); }
  .ratio-info { font-size:10px; color:var(--text-dim); text-align:right; line-height:1.5; }
  .ratio-bar { height:1px; background:var(--dark3); }
  .ratio-bar-fill { height:100%; background:var(--gold); transition:width 1s ease; }

  /* Analysis */
  .analysis-box {
    background:var(--dark2); border:1px solid var(--dark3); border-left:2px solid var(--gold-dim);
    padding:26px; margin-bottom:28px;
  }
  .analysis-title { font-size:9px; letter-spacing:5px; text-transform:uppercase; color:var(--gold-dim); display:block; margin-bottom:14px; }
  .analysis-text { font-family:'Cormorant Garamond',serif; font-size:17px; line-height:1.85; color:var(--text); }

  /* Error */
  .error-box { display:none; border:1px solid #8B3A3A; background:rgba(139,58,58,.1); padding:24px; margin-top:20px; text-align:center; }
  .error-box p { font-family:'Cormorant Garamond',serif; font-size:16px; color:#E87070; }

  .reset-btn {
    width:100%; background:transparent; border:1px solid var(--dark3); color:var(--text-dim);
    padding:14px; font-family:'Josefin Sans',sans-serif; font-size:9px;
    letter-spacing:4px; text-transform:uppercase; cursor:pointer; transition:all .3s;
  }
  .reset-btn:hover { border-color:var(--gold-dim); color:var(--gold); }
  .footer { margin-top:60px; text-align:center; font-size:9px; letter-spacing:3px; text-transform:uppercase; color:var(--text-dim); opacity:.4; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <span class="phi-symbol">φ</span>
    <p class="title">Golden Ratio · Facial Harmony Analysis</p>
    <p class="subtitle">The mathematics of aesthetic proportion</p>
    <div class="divider">
      <div class="divider-line"></div><div class="divider-dot"></div><div class="divider-line"></div>
    </div>
    <p class="desc">Since antiquity, the golden ratio φ = 1.618… has been considered the universal standard of beauty. Upload a frontal selfie to discover how closely your facial proportions align with this timeless ideal.</p>
    <span class="free-badge">◈ &nbsp; No API Key Required &nbsp; ◈</span>
  </div>

  <!-- Upload -->
  <div id="dropZone" class="upload-zone" onclick="document.getElementById('fileInput').click()">
    <span class="upload-icon">◈</span>
    <span class="upload-text">Upload Frontal Selfie</span>
    <span class="upload-hint">Drag & drop or click · JPG PNG WEBP</span>
    <input type="file" id="fileInput" accept="image/*">
  </div>

  <div class="preview-wrap" id="previewWrap">
    <img id="previewImg" src="" alt="">
    <div class="preview-overlay">
      <button class="btn-remove" onclick="resetAll()">✕ Remove</button>
    </div>
  </div>

  <button class="analyze-btn" id="analyzeBtn" onclick="analyze()">
    <span>◈ &nbsp; Analyze Phi Harmony</span>
  </button>

  <div class="loading" id="loading">
    <div class="loading-ring"></div>
    <p class="loading-text">Measuring golden ratios</p>
  </div>

  <div class="error-box" id="errorBox"><p id="errorText"></p></div>

  <!-- Results -->
  <div class="results" id="results">
    <div class="score-section">
      <span class="score-label">Overall Phi Harmony Score</span><br>
      <span class="score-number" id="scoreNum">0</span><span class="score-unit">%</span>
      <p class="score-verdict" id="scoreVerdict"></p>
      <div class="score-bar-wrap">
        <div class="score-bar-track"><div class="score-bar-fill" id="scoreBar"></div></div>
      </div>
    </div>

    <div class="annotated-wrap" id="annotatedWrap" style="display:none">
      <img id="annotatedImg" src="" alt="Annotated face">
      <p class="annotated-label">◈ &nbsp; Detected Facial Landmarks &nbsp; ◈</p>
    </div>

    <p class="section-title">Individual Ratio Analysis</p>
    <div class="ratios-grid" id="ratiosGrid"></div>

    <div class="analysis-box">
      <span class="analysis-title">φ · Aesthetic Interpretation</span>
      <p class="analysis-text" id="analysisText"></p>
    </div>

    <button class="reset-btn" onclick="resetAll()">↺ &nbsp; Analyze Another Photo</button>
  </div>

  <div class="footer">φ = 1.6180339887… · The Divine Proportion · Powered by OpenCV</div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
let currentFile = null;

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) loadFile(e.target.files[0]); });

function loadFile(file) {
  currentFile = file;
  const r = new FileReader();
  r.onload = ev => {
    document.getElementById('previewImg').src = ev.target.result;
    dropZone.style.display = 'none';
    document.getElementById('previewWrap').style.display = 'block';
    document.getElementById('analyzeBtn').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
  };
  r.readAsDataURL(file);
}

function resetAll() {
  currentFile = null; fileInput.value = '';
  document.getElementById('previewImg').src = '';
  dropZone.style.display = 'block';
  document.getElementById('previewWrap').style.display = 'none';
  document.getElementById('analyzeBtn').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('loading').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';
}

async function analyze() {
  if (!currentFile) return;
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';

  const fd = new FormData();
  fd.append('image', currentFile);

  try {
    const res = await fetch('/analyze', { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok || data.error) { showError(data.error || 'Analysis failed.'); return; }
    displayResults(data);
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}

function showError(msg) {
  document.getElementById('errorText').textContent = msg;
  document.getElementById('errorBox').style.display = 'block';
}

function getVerdict(s) {
  if (s >= 90) return 'Exceptional — Near-perfect divine proportion';
  if (s >= 80) return 'Remarkable — Highly harmonious features';
  if (s >= 70) return 'Harmonious — Strong phi alignment';
  if (s >= 60) return 'Balanced — Pleasant natural proportion';
  if (s >= 50) return 'Moderate — Characteristic individuality';
  return 'Distinctive — Unique beyond classical proportion';
}

function displayResults(data) {
  const score = data.overall_score;
  document.getElementById('scoreVerdict').textContent = getVerdict(score);
  setTimeout(() => { document.getElementById('scoreBar').style.width = score + '%'; }, 100);
  animateNum('scoreNum', 0, score, 1300);

  // Annotated image
  if (data.annotated_image) {
    document.getElementById('annotatedImg').src = 'data:image/jpeg;base64,' + data.annotated_image;
    document.getElementById('annotatedWrap').style.display = 'block';
  }

  // Ratio cards
  const grid = document.getElementById('ratiosGrid');
  grid.innerHTML = '';
  data.ratios.forEach(r => {
    const card = document.createElement('div');
    card.className = 'ratio-card';
    card.innerHTML = `
      <div class="ratio-name">${r.name}</div>
      <div class="ratio-row">
        <span class="ratio-score">${r.score.toFixed(1)}%</span>
        <span class="ratio-info">Measured: ${r.measured.toFixed(3)}<br>φ deviation: ${r.deviation.toFixed(3)}</span>
      </div>
      <div class="ratio-bar"><div class="ratio-bar-fill" style="width:0%" data-target="${r.score}"></div></div>`;
    grid.appendChild(card);
    setTimeout(() => card.querySelector('.ratio-bar-fill').style.width = r.score + '%', 200);
  });

  document.getElementById('analysisText').textContent = data.analysis;
  document.getElementById('results').style.display = 'block';
}

function animateNum(id, from, to, dur) {
  const el = document.getElementById(id);
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / dur, 1);
    const e = t < .5 ? 2*t*t : -1+(4-2*t)*t;
    el.textContent = (from + (to - from) * e).toFixed(1);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
</script>
</body>
</html>
"""


def ratio_score(measured):
    """Convert a measured ratio to a 0-100 score based on proximity to phi."""
    deviation = abs(measured - PHI)
    score = max(0.0, 100.0 * (1.0 - deviation / PHI))
    return round(score, 2), round(deviation, 4)


def generate_analysis(overall, ratios):
    """Generate a written interpretation based on scores."""
    best = max(ratios, key=lambda r: r['score'])
    worst = min(ratios, key=lambda r: r['score'])

    if overall >= 85:
        opening = "Your facial proportions demonstrate a remarkable alignment with the golden ratio, placing you among the most harmoniously proportioned faces by classical aesthetic standards."
    elif overall >= 72:
        opening = "Your face exhibits strong phi harmony across multiple dimensions, reflecting the kind of balanced, naturally pleasing proportion that artists and architects have sought to capture for millennia."
    elif overall >= 60:
        opening = "Your facial structure shows moderate golden ratio alignment, with several features echoing the divine proportion in ways that contribute to an overall sense of natural balance."
    else:
        opening = "Your face carries a distinctive character that departs from classical phi proportions — a quality shared by many celebrated faces throughout art history, where individuality often surpasses formula."

    middle = f"Your strongest alignment is in {best['name'].lower()} ({best['score']:.1f}%), while {worst['name'].lower()} shows the most deviation from φ."
    closing = f"With an overall harmony score of {overall:.1f}%, your features reflect the beautiful complexity of human proportion that no single formula can fully contain."

    return f"{opening} {middle} {closing}"


def annotate_image(img, face, eyes, nose_pt, mouth_pt, landmarks):
    """Draw golden ratio landmarks on the image."""
    out = img.copy()
    x, y, w, h = face

    # Face rectangle
    cv2.rectangle(out, (x, y), (x+w, y+h), (201, 168, 76), 1)

    # Eyes
    for (ex, ey, ew, eh) in eyes:
        cx = x + ex + ew // 2
        cy = y + ey + eh // 2
        cv2.circle(out, (cx, cy), 3, (201, 168, 76), -1)
        cv2.ellipse(out, (cx, cy), (ew//2, eh//2), 0, 0, 360, (201, 168, 76), 1)

    # Nose estimate
    if nose_pt:
        cv2.circle(out, nose_pt, 4, (232, 201, 122), -1)
        cv2.circle(out, nose_pt, 10, (232, 201, 122), 1)

    # Mouth estimate
    if mouth_pt:
        cv2.circle(out, mouth_pt, 4, (138, 110, 47), -1)

    # Vertical thirds lines
    top3    = y
    third1  = y + h // 3
    third2  = y + 2 * h // 3
    bottom3 = y + h
    for yy in [top3, third1, third2, bottom3]:
        cv2.line(out, (x, yy), (x+w, yy), (201, 168, 76), 1)

    # Horizontal center
    cx_face = x + w // 2
    cv2.line(out, (cx_face, y), (cx_face, y+h), (201, 168, 76), 1)

    # Phi label
    cv2.putText(out, f"phi={PHI:.4f}", (x, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (201, 168, 76), 1, cv2.LINE_AA)

    return out


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    img_bytes = file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image. Please upload a valid JPG or PNG.'}), 400

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # --- Face detection ---
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
    if len(faces) == 0:
        return jsonify({'error': 'No face detected. Please use a clear frontal selfie with good lighting and no heavy shadows.'}), 400

    # Use largest face
    face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = face

    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
    face_roi_color = img[fy:fy+fh, fx:fx+fw]

    # --- Eye detection within face ROI ---
    eyes = EYE_CASCADE.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # Filter eyes to upper half of face
    eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey < fh * 0.55]

    # Sort left to right
    eyes = sorted(eyes, key=lambda e: e[0])

    # --- Derived landmark estimates ---
    # Eye centers in face-local coords
    eye_centers = [(ex + ew//2, ey + eh//2) for (ex, ey, ew, eh) in eyes]

    # Estimate nose tip: center-x, ~60% down face
    nose_local = (fw // 2, int(fh * 0.60))
    nose_global = (fx + nose_local[0], fy + nose_local[1])

    # Estimate mouth center: center-x, ~78% down face
    mouth_local = (fw // 2, int(fh * 0.78))
    mouth_global = (fx + mouth_local[0], fy + mouth_local[1])

    # --- Phi Ratio calculations ---
    ratios = []

    # 1. Face Width to Height
    face_wh = fw / fh if fh > 0 else PHI
    s, d = ratio_score(face_wh)
    ratios.append({'name': 'Face Width to Height', 'measured': round(face_wh, 4), 'score': s, 'deviation': d})

    # 2. Eye spacing to eye width (if 2 eyes detected)
    if len(eyes) >= 2:
        left_eye = eyes[0]
        right_eye = eyes[-1]
        eye_spacing = abs((right_eye[0] + right_eye[2]//2) - (left_eye[0] + left_eye[2]//2))
        avg_eye_w = (left_eye[2] + right_eye[2]) / 2
        ratio_es = eye_spacing / avg_eye_w if avg_eye_w > 0 else PHI
        s, d = ratio_score(ratio_es)
        ratios.append({'name': 'Eye Spacing to Eye Width', 'measured': round(ratio_es, 4), 'score': s, 'deviation': d})

        # 3. Face width to eye spacing
        ratio_fe = fw / eye_spacing if eye_spacing > 0 else PHI
        s, d = ratio_score(ratio_fe)
        ratios.append({'name': 'Face Width to Eye Span', 'measured': round(ratio_fe, 4), 'score': s, 'deviation': d})

        # 4. Eye width ratio (left/right symmetry)
        eye_sym = max(left_eye[2], right_eye[2]) / min(left_eye[2], right_eye[2]) if min(left_eye[2], right_eye[2]) > 0 else 1.0
        ratio_sym = eye_sym
        s, d = ratio_score(ratio_sym)
        ratios.append({'name': 'Eye Width Symmetry', 'measured': round(ratio_sym, 4), 'score': s, 'deviation': d})
    else:
        # Single eye or no eyes — use face geometry estimates
        est_eye_w = fw * 0.22
        est_spacing = fw * 0.35
        ratio_es = est_spacing / est_eye_w
        s, d = ratio_score(ratio_es)
        ratios.append({'name': 'Eye Spacing to Eye Width (est.)', 'measured': round(ratio_es, 4), 'score': s, 'deviation': d})

    # 5. Upper face (hairline→eyes) to mid face (eyes→nose)
    if len(eye_centers) >= 1:
        avg_eye_y = sum(ec[1] for ec in eye_centers) / len(eye_centers)
        upper_face = avg_eye_y          # hairline to eyes (in face ROI)
        mid_face   = nose_local[1] - avg_eye_y
        if mid_face > 0 and upper_face > 0:
            ratio_um = upper_face / mid_face
            s, d = ratio_score(ratio_um)
            ratios.append({'name': 'Hairline to Eye / Eye to Nose', 'measured': round(ratio_um, 4), 'score': s, 'deviation': d})

        # 6. Mid face (eyes→nose) to lower face (nose→chin)
        lower_face = fh - nose_local[1]
        if mid_face > 0 and lower_face > 0:
            ratio_ml = mid_face / lower_face
            s, d = ratio_score(ratio_ml)
            ratios.append({'name': 'Mid Face to Lower Face', 'measured': round(ratio_ml, 4), 'score': s, 'deviation': d})

    # 7. Nose width estimate (~28% face width) to mouth width (~46% face width)
    nose_w_est  = fw * 0.28
    mouth_w_est = fw * 0.46
    ratio_nm = mouth_w_est / nose_w_est
    s, d = ratio_score(ratio_nm)
    ratios.append({'name': 'Mouth Width to Nose Width', 'measured': round(ratio_nm, 4), 'score': s, 'deviation': d})

    # 8. Face height to nose-to-chin distance
    nose_chin = fh - nose_local[1]
    ratio_hn = fh / nose_chin if nose_chin > 0 else PHI
    s, d = ratio_score(ratio_hn)
    ratios.append({'name': 'Face Height to Chin Segment', 'measured': round(ratio_hn, 4), 'score': s, 'deviation': d})

    # --- Overall score: weighted average ---
    # Eye-based ratios weighted higher (more accurate)
    weights = [1.0] * len(ratios)
    if len(eyes) >= 2:
        weights[1] = 1.5  # eye spacing
        weights[2] = 1.3  # face width to eye span
    total_w = sum(weights)
    overall = sum(r['score'] * w for r, w in zip(ratios, weights)) / total_w

    # --- Annotate image ---
    annotated = annotate_image(img, face, eyes, nose_global, mouth_global, eye_centers)
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    annotated_b64 = base64.standard_b64encode(buf).decode('utf-8')

    # --- Analysis text ---
    analysis = generate_analysis(overall, ratios)

    return jsonify({
        'overall_score': round(overall, 2),
        'ratios': ratios,
        'analysis': analysis,
        'annotated_image': annotated_b64,
        'landmarks_detected': len(eyes)
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5050))
    print("\n  φ  Phi Face Analyzer  (No API Key Required)")
    print("  ──────────────────────────────────────────")
    print(f"  Running on port {port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
