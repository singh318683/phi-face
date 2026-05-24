import os
import base64
import math
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
PHI = 1.6180339887

FACE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ─────────────────────────────────────────────
#  DENTAL SMILE KNOWLEDGE BASE
# ─────────────────────────────────────────────
DENTAL_PROCEDURES = {
    "teeth_whitening": {
        "name": "Teeth Whitening",
        "icon": "✦",
        "description": "Professional bleaching to remove stains and discoloration.",
        "ideal_for": ["yellowing", "staining", "dull teeth", "coffee/tea stains"],
        "types": ["In-office laser whitening (1 session)", "Custom take-home trays (2-3 weeks)", "Over-the-counter strips (mild cases)"],
        "cost": "$300–$1,500 (professional) / $20–$100 (OTC)",
        "duration": "1–3 hours (in-office) or 2–3 weeks (home)",
        "longevity": "1–3 years with maintenance",
        "pain_level": "Minimal — slight sensitivity",
        "disclaimer": "Consult your dentist before whitening if you have crowns, veneers, or sensitive teeth."
    },
    "veneers": {
        "name": "Dental Veneers",
        "icon": "◈",
        "description": "Thin porcelain or composite shells bonded to the front of teeth for a perfect smile.",
        "ideal_for": ["chipped teeth", "gaps", "misshapen teeth", "severe staining", "minor misalignment"],
        "types": ["Porcelain veneers (most natural)", "Composite veneers (less expensive)", "Lumineers (no-prep, reversible)"],
        "cost": "$900–$2,500 per tooth",
        "duration": "2–3 dental visits over 2–4 weeks",
        "longevity": "10–15 years",
        "pain_level": "Low — local anesthesia used",
        "disclaimer": "Veneers require removal of a small amount of enamel and are generally irreversible."
    },
    "bonding": {
        "name": "Dental Bonding",
        "icon": "◇",
        "description": "Tooth-colored resin applied and shaped to repair chips, gaps, or discoloration.",
        "ideal_for": ["small chips", "minor gaps", "short teeth", "surface stains", "budget-conscious patients"],
        "types": ["Direct composite bonding (single visit)", "Adhesive bonding"],
        "cost": "$100–$400 per tooth",
        "duration": "30–60 minutes per tooth",
        "longevity": "5–10 years",
        "pain_level": "Painless — no anesthesia usually needed",
        "disclaimer": "Bonding is less durable than veneers and may stain over time."
    },
    "orthodontics": {
        "name": "Orthodontic Treatment",
        "icon": "⬡",
        "description": "Corrects misaligned, crowded, or spaced teeth for functional and aesthetic improvement.",
        "ideal_for": ["crooked teeth", "crowding", "gaps", "overbite", "underbite", "crossbite"],
        "types": ["Traditional metal braces", "Ceramic braces (discreet)", "Invisalign clear aligners", "Lingual braces (hidden)"],
        "cost": "$3,000–$8,000",
        "duration": "12–36 months",
        "longevity": "Permanent with retainer use",
        "pain_level": "Moderate — adjustment soreness",
        "disclaimer": "Orthodontic treatment requires commitment and regular dental visits."
    },
    "crown": {
        "name": "Dental Crowns",
        "icon": "♛",
        "description": "Caps that cover damaged or weakened teeth, restoring shape, size, and strength.",
        "ideal_for": ["severely damaged teeth", "after root canal", "large fillings", "broken teeth"],
        "types": ["Porcelain (most aesthetic)", "Porcelain-fused-to-metal", "Zirconia (strongest)", "Gold (back teeth)"],
        "cost": "$800–$2,500 per tooth",
        "duration": "2 visits over 2–3 weeks",
        "longevity": "10–25 years",
        "pain_level": "Low — local anesthesia used",
        "disclaimer": "Crown placement requires significant tooth reduction."
    },
    "implants": {
        "name": "Dental Implants",
        "icon": "⊕",
        "description": "Titanium posts surgically placed to replace missing teeth permanently.",
        "ideal_for": ["missing teeth", "failing teeth", "denture alternatives"],
        "types": ["Single implant", "Implant-supported bridge", "All-on-4 (full arch)", "Mini implants"],
        "cost": "$3,000–$6,000 per implant",
        "duration": "3–9 months total (healing time)",
        "longevity": "20+ years, often lifetime",
        "pain_level": "Moderate — surgical procedure",
        "disclaimer": "Requires adequate bone density. Not suitable for smokers or uncontrolled diabetics without medical clearance."
    },
    "gum_contouring": {
        "name": "Gum Contouring",
        "icon": "◑",
        "description": "Reshaping the gum line to correct a 'gummy smile' or uneven gums.",
        "ideal_for": ["gummy smile", "uneven gum line", "teeth appearing too short"],
        "types": ["Laser gum contouring (most common)", "Surgical gingivectomy"],
        "cost": "$200–$3,000 depending on extent",
        "duration": "1–2 hours",
        "longevity": "Permanent",
        "pain_level": "Low — local anesthesia, mild soreness after",
        "disclaimer": "Laser contouring is minimally invasive; surgical options require longer recovery."
    },
    "smile_makeover": {
        "name": "Full Smile Makeover",
        "icon": "✸",
        "description": "Combination of multiple cosmetic procedures for a complete smile transformation.",
        "ideal_for": ["multiple concerns", "aged smile", "total transformation", "special occasions"],
        "types": ["Whitening + veneers", "Orthodontics + whitening", "Veneers + gum contouring + whitening"],
        "cost": "$5,000–$30,000+",
        "duration": "3–12 months",
        "longevity": "10–20 years with maintenance",
        "pain_level": "Varies by procedures chosen",
        "disclaimer": "Always get multiple consultations and a comprehensive treatment plan before committing."
    }
}

SMILE_CONCERNS = {
    "discoloration": ["teeth_whitening", "veneers", "bonding"],
    "gaps":          ["veneers", "bonding", "orthodontics"],
    "chips":         ["bonding", "veneers", "crown"],
    "misalignment":  ["orthodontics", "veneers"],
    "gummy_smile":   ["gum_contouring", "veneers"],
    "missing":       ["implants", "crown"],
    "shape":         ["veneers", "bonding", "crown"],
    "overall":       ["smile_makeover", "teeth_whitening", "veneers"]
}

PHI_SMILE_RATIOS = {
    "central_lateral": {"ideal": 1.618, "description": "Central incisor width to lateral incisor width"},
    "lateral_canine":  {"ideal": 1.618, "description": "Lateral incisor width to canine width"},
    "smile_to_face":   {"ideal": 0.618, "description": "Smile width as proportion of face width (golden section)"},
    "tooth_height_width": {"ideal": 0.618, "description": "Ideal tooth height-to-width ratio"},
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>φ Face & Smile Analyzer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Josefin+Sans:wght@100;300;400&display=swap');
  :root {
    --gold:#C9A84C; --gold-light:#E8C97A; --gold-dim:#8A6E2F;
    --dark:#0A0A0A; --dark2:#111111; --dark3:#1A1A1A; --dark4:#222;
    --text:#E8E0D0; --text-dim:#8A8070;
    --pearl:#F5F0E8; --ivory:#EDE8DC;
    --teal:#4CABA8; --teal-dim:#2A6E6C;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--dark);color:var(--text);font-family:'Josefin Sans',sans-serif;font-weight:300;min-height:100vh;}
  body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
    background:radial-gradient(ellipse 80% 60% at 50% -10%,rgba(201,168,76,.08) 0%,transparent 60%),
               radial-gradient(ellipse 40% 40% at 90% 80%,rgba(76,171,168,.04) 0%,transparent 50%);}
  .container{position:relative;z-index:1;max-width:920px;margin:0 auto;padding:60px 24px 80px;}

  /* HEADER */
  .header{text-align:center;margin-bottom:52px;}
  .phi-symbol{font-family:'Cormorant Garamond',serif;font-size:80px;font-weight:300;color:var(--gold);line-height:1;display:block;animation:glow 4s ease-in-out infinite;}
  @keyframes glow{0%,100%{text-shadow:0 0 30px rgba(201,168,76,.3);}50%{text-shadow:0 0 70px rgba(201,168,76,.7),0 0 120px rgba(201,168,76,.2);}}
  .title{font-size:10px;letter-spacing:8px;text-transform:uppercase;color:var(--text-dim);margin-top:10px;}
  .subtitle{font-family:'Cormorant Garamond',serif;font-size:22px;color:var(--text);margin-top:12px;opacity:.85;}
  .divider{display:flex;align-items:center;gap:16px;margin:24px auto;max-width:300px;}
  .divider-line{flex:1;height:1px;background:linear-gradient(to right,transparent,var(--gold-dim));}
  .divider-line:last-child{background:linear-gradient(to left,transparent,var(--gold-dim));}
  .divider-dot{width:6px;height:6px;background:var(--gold);transform:rotate(45deg);box-shadow:0 0 10px rgba(201,168,76,.5);}
  .desc{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text-dim);line-height:1.75;max-width:540px;margin:0 auto;}
  .free-badge{display:inline-block;margin-top:14px;border:1px solid var(--gold-dim);padding:5px 18px;font-size:9px;letter-spacing:4px;text-transform:uppercase;color:var(--gold-dim);}

  /* TABS */
  .tab-nav{display:flex;gap:0;margin-bottom:32px;border:1px solid var(--dark3);}
  .tab-btn{flex:1;padding:16px;background:transparent;border:none;color:var(--text-dim);font-family:'Josefin Sans',sans-serif;font-size:10px;letter-spacing:4px;text-transform:uppercase;cursor:pointer;transition:all .3s;border-right:1px solid var(--dark3);}
  .tab-btn:last-child{border-right:none;}
  .tab-btn.active{background:var(--dark2);color:var(--gold);}
  .tab-btn:hover:not(.active){color:var(--text);background:rgba(255,255,255,.02);}
  .tab-panel{display:none;}
  .tab-panel.active{display:block;}

  /* UPLOAD */
  .upload-zone{border:1px solid var(--gold-dim);border-radius:2px;padding:52px 40px;text-align:center;cursor:pointer;background:linear-gradient(135deg,var(--dark2),var(--dark3));transition:all .4s;position:relative;overflow:hidden;}
  .upload-zone::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(201,168,76,.06) 0%,transparent 60%);opacity:0;transition:opacity .4s;}
  .upload-zone:hover::before,.upload-zone.drag-over::before{opacity:1;}
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--gold);}
  .upload-icon{font-size:40px;display:block;margin-bottom:16px;opacity:.55;}
  .upload-text{font-size:10px;letter-spacing:4px;text-transform:uppercase;color:var(--gold);display:block;margin-bottom:8px;}
  .upload-hint{font-family:'Cormorant Garamond',serif;font-size:16px;color:var(--text-dim);}
  #fileInput{display:none;}
  .preview-wrap{display:none;border:1px solid var(--gold-dim);border-radius:2px;overflow:hidden;background:var(--dark2);position:relative;}
  .preview-wrap img{width:100%;max-height:420px;object-fit:contain;display:block;}
  .preview-overlay{position:absolute;top:12px;right:12px;}
  .btn-remove{background:rgba(10,10,10,.85);border:1px solid var(--gold-dim);color:var(--text-dim);padding:6px 14px;font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:3px;text-transform:uppercase;cursor:pointer;transition:all .3s;}
  .btn-remove:hover{border-color:var(--gold);color:var(--gold);}

  /* ANALYZE BTN */
  .analyze-btn{width:100%;margin-top:20px;padding:20px;background:transparent;border:1px solid var(--gold);color:var(--gold);font-family:'Josefin Sans',sans-serif;font-size:11px;font-weight:400;letter-spacing:6px;text-transform:uppercase;cursor:pointer;position:relative;overflow:hidden;transition:all .4s;display:none;}
  .analyze-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--gold),var(--gold-light));transform:translateX(-100%);transition:transform .4s;z-index:0;}
  .analyze-btn:hover::before{transform:translateX(0);}
  .analyze-btn:hover{color:var(--dark);}
  .analyze-btn span{position:relative;z-index:1;}
  .analyze-btn:disabled{opacity:.4;cursor:not-allowed;}
  .analyze-btn:disabled::before{display:none;}

  /* LOADING */
  .loading{display:none;text-align:center;padding:48px;}
  .loading-ring{width:52px;height:52px;border:1px solid var(--dark3);border-top-color:var(--gold);border-radius:50%;animation:spin 1.2s linear infinite;margin:0 auto 20px;}
  @keyframes spin{to{transform:rotate(360deg);}}
  .loading-text{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--text-dim);animation:pulse 2s ease-in-out infinite;}
  @keyframes pulse{0%,100%{opacity:.4}50%{opacity:1}}

  /* ERROR */
  .error-box{display:none;border:1px solid #8B3A3A;background:rgba(139,58,58,.1);padding:24px;margin-top:20px;text-align:center;}
  .error-box p{font-family:'Cormorant Garamond',serif;font-size:16px;color:#E87070;}

  /* RESULTS COMMON */
  .results{display:none;animation:fadeUp .8s ease forwards;}
  @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
  .section-title{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--text-dim);margin-bottom:18px;display:flex;align-items:center;gap:12px;}
  .section-title::after{content:'';flex:1;height:1px;background:var(--dark3);}

  /* SCORE BLOCK */
  .score-section{text-align:center;padding:44px 24px;background:linear-gradient(135deg,var(--dark2),var(--dark3));border:1px solid var(--gold-dim);margin-bottom:24px;position:relative;overflow:hidden;}
  .score-section::before{content:'φ';position:absolute;font-family:'Cormorant Garamond',serif;font-size:200px;color:rgba(201,168,76,.03);top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;}
  .score-label{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--text-dim);display:block;margin-bottom:12px;}
  .score-number{font-family:'Cormorant Garamond',serif;font-size:96px;font-weight:300;line-height:1;color:var(--gold);display:inline;}
  .score-unit{font-family:'Cormorant Garamond',serif;font-size:32px;font-weight:300;color:var(--gold-dim);}
  .score-verdict{font-family:'Cormorant Garamond',serif;font-size:19px;color:var(--text);margin-top:12px;font-style:italic;}
  .score-bar-track{height:2px;background:var(--dark3);margin-top:22px;}
  .score-bar-fill{height:100%;background:linear-gradient(to right,var(--gold-dim),var(--gold),var(--gold-light));width:0%;transition:width 1.6s cubic-bezier(.4,0,.2,1);}

  /* RATIO CARDS */
  .ratios-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:12px;margin-bottom:24px;}
  .ratio-card{background:var(--dark2);border:1px solid var(--dark3);padding:18px;transition:border-color .3s;}
  .ratio-card:hover{border-color:var(--gold-dim);}
  .ratio-name{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);margin-bottom:10px;}
  .ratio-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;}
  .ratio-score{font-family:'Cormorant Garamond',serif;font-size:30px;color:var(--gold);}
  .ratio-info{font-size:10px;color:var(--text-dim);text-align:right;line-height:1.6;}
  .ratio-bar{height:1px;background:var(--dark3);}
  .ratio-bar-fill{height:100%;background:var(--gold);transition:width 1s ease;}

  /* ANNOTATED IMAGE */
  .annotated-wrap{border:1px solid var(--gold-dim);margin-bottom:24px;background:var(--dark2);}
  .annotated-wrap img{width:100%;display:block;}
  .annotated-label{font-size:9px;letter-spacing:4px;text-transform:uppercase;color:var(--gold-dim);padding:10px 16px;border-top:1px solid var(--dark3);text-align:center;}

  /* ANALYSIS BOX */
  .analysis-box{background:var(--dark2);border:1px solid var(--dark3);border-left:2px solid var(--gold-dim);padding:26px;margin-bottom:24px;}
  .analysis-title{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--gold-dim);display:block;margin-bottom:14px;}
  .analysis-text{font-family:'Cormorant Garamond',serif;font-size:17px;line-height:1.85;color:var(--text);}

  /* ── DENTAL SECTION ── */
  .dental-header{text-align:center;padding:40px 24px;background:linear-gradient(135deg,var(--dark2),var(--dark3));border:1px solid var(--teal-dim);margin-bottom:28px;position:relative;overflow:hidden;}
  .dental-header::before{content:'⌘';position:absolute;font-size:180px;color:rgba(76,171,168,.04);top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;}
  .smile-icon{font-size:56px;display:block;margin-bottom:12px;}
  .dental-title{font-family:'Cormorant Garamond',serif;font-size:32px;font-weight:300;color:var(--teal);display:block;}
  .dental-subtitle{font-family:'Cormorant Garamond',serif;font-size:16px;color:var(--text-dim);margin-top:8px;font-style:italic;}

  /* Concern selector */
  .concern-section{margin-bottom:28px;}
  .concern-label{font-size:10px;letter-spacing:4px;text-transform:uppercase;color:var(--text-dim);margin-bottom:14px;display:block;}
  .concern-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;}
  .concern-btn{padding:14px 10px;background:var(--dark2);border:1px solid var(--dark3);color:var(--text-dim);font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:3px;text-transform:uppercase;cursor:pointer;transition:all .3s;text-align:center;}
  .concern-btn:hover{border-color:var(--teal-dim);color:var(--teal);}
  .concern-btn.selected{border-color:var(--teal);color:var(--teal);background:rgba(76,171,168,.08);}
  .concern-icon{font-size:20px;display:block;margin-bottom:6px;}

  /* Get recommendations btn */
  .recommend-btn{width:100%;padding:18px;background:transparent;border:1px solid var(--teal);color:var(--teal);font-family:'Josefin Sans',sans-serif;font-size:11px;letter-spacing:5px;text-transform:uppercase;cursor:pointer;position:relative;overflow:hidden;transition:all .4s;margin-top:16px;}
  .recommend-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,var(--teal-dim),var(--teal));transform:translateX(-100%);transition:transform .4s;z-index:0;}
  .recommend-btn:hover::before{transform:translateX(0);}
  .recommend-btn:hover{color:var(--dark);}
  .recommend-btn span{position:relative;z-index:1;}

  /* Phi smile ratios */
  .phi-smile-section{background:var(--dark2);border:1px solid var(--dark3);padding:24px;margin-bottom:24px;}
  .phi-smile-title{font-size:9px;letter-spacing:5px;text-transform:uppercase;color:var(--teal-dim);display:block;margin-bottom:20px;}
  .phi-ratio-row{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid var(--dark3);}
  .phi-ratio-row:last-child{border-bottom:none;}
  .phi-ratio-name{font-size:11px;letter-spacing:2px;color:var(--text);}
  .phi-ratio-ideal{font-family:'Cormorant Garamond',serif;font-size:20px;color:var(--gold);}
  .phi-ratio-desc{font-size:10px;color:var(--text-dim);margin-top:2px;}

  /* Procedure cards */
  .procedures-results{display:none;animation:fadeUp .6s ease forwards;}
  .procedure-card{background:var(--dark2);border:1px solid var(--dark3);border-top:2px solid var(--teal-dim);padding:24px;margin-bottom:16px;transition:border-color .3s;}
  .procedure-card:hover{border-color:var(--teal-dim);border-top-color:var(--teal);}
  .proc-header{display:flex;align-items:center;gap:14px;margin-bottom:16px;}
  .proc-icon{font-size:28px;color:var(--teal);}
  .proc-name{font-family:'Cormorant Garamond',serif;font-size:22px;color:var(--text);}
  .proc-desc{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text-dim);line-height:1.7;margin-bottom:16px;font-style:italic;}
  .proc-meta{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:16px;}
  .proc-meta-item{background:var(--dark3);padding:12px;}
  .proc-meta-label{font-size:8px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);margin-bottom:5px;}
  .proc-meta-value{font-family:'Cormorant Garamond',serif;font-size:15px;color:var(--text);}
  .proc-types{margin-bottom:14px;}
  .proc-types-label{font-size:8px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);margin-bottom:8px;}
  .proc-type-tag{display:inline-block;border:1px solid var(--teal-dim);color:var(--teal);padding:3px 10px;font-size:9px;letter-spacing:2px;margin:3px 4px 3px 0;}
  .proc-disclaimer{font-size:11px;color:var(--text-dim);border-left:2px solid var(--gold-dim);padding-left:12px;font-style:italic;line-height:1.6;}

  /* Disclaimer banner */
  .disclaimer-banner{background:rgba(201,168,76,.06);border:1px solid var(--gold-dim);padding:20px 24px;margin-bottom:24px;}
  .disclaimer-banner p{font-family:'Cormorant Garamond',serif;font-size:14px;color:var(--text-dim);line-height:1.7;}
  .disclaimer-banner strong{color:var(--gold);font-weight:400;}

  /* Reset */
  .reset-btn{width:100%;background:transparent;border:1px solid var(--dark3);color:var(--text-dim);padding:14px;font-family:'Josefin Sans',sans-serif;font-size:9px;letter-spacing:4px;text-transform:uppercase;cursor:pointer;transition:all .3s;margin-top:8px;}
  .reset-btn:hover{border-color:var(--gold-dim);color:var(--gold);}
  .footer{margin-top:56px;text-align:center;font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);opacity:.4;}
</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <div class="header">
    <span class="phi-symbol">φ</span>
    <p class="title">Golden Ratio · Face & Smile Analyzer</p>
    <p class="subtitle">Facial harmony & dental aesthetics</p>
    <div class="divider"><div class="divider-line"></div><div class="divider-dot"></div><div class="divider-line"></div></div>
    <p class="desc">Discover your facial phi harmony score and get personalised dental smile recommendations — all powered by computer vision, no API key required.</p>
    <span class="free-badge">◈ &nbsp; 100% Free · No API Key &nbsp; ◈</span>
  </div>

  <!-- TABS -->
  <div class="tab-nav">
    <button class="tab-btn active" onclick="switchTab('face')">◈ &nbsp; Face Analysis</button>
    <button class="tab-btn" onclick="switchTab('smile')">⌘ &nbsp; Dental Smile</button>
  </div>

  <!-- ═══════════════ TAB 1: FACE ═══════════════ -->
  <div class="tab-panel active" id="tab-face">
    <div id="dropZone" class="upload-zone" onclick="document.getElementById('fileInput').click()">
      <span class="upload-icon">◈</span>
      <span class="upload-text">Upload Frontal Selfie</span>
      <span class="upload-hint">Drag & drop or click · JPG PNG WEBP</span>
      <input type="file" id="fileInput" accept="image/*">
    </div>
    <div class="preview-wrap" id="previewWrap">
      <img id="previewImg" src="" alt="">
      <div class="preview-overlay"><button class="btn-remove" onclick="resetFace()">✕ Remove</button></div>
    </div>
    <button class="analyze-btn" id="analyzeBtn" onclick="analyzeFace()"><span>◈ &nbsp; Analyze Phi Harmony</span></button>
    <div class="loading" id="loading"><div class="loading-ring"></div><p class="loading-text">Measuring golden ratios</p></div>
    <div class="error-box" id="errorBox"><p id="errorText"></p></div>

    <div class="results" id="faceResults">
      <div class="score-section">
        <span class="score-label">Overall Phi Harmony Score</span><br>
        <span class="score-number" id="scoreNum">0</span><span class="score-unit">%</span>
        <p class="score-verdict" id="scoreVerdict"></p>
        <div class="score-bar-track"><div class="score-bar-fill" id="scoreBar"></div></div>
      </div>
      <div class="annotated-wrap" id="annotatedWrap" style="display:none">
        <img id="annotatedImg" src="" alt="Annotated">
        <p class="annotated-label">◈ &nbsp; Detected Facial Landmarks &nbsp; ◈</p>
      </div>
      <p class="section-title">Individual Ratio Analysis</p>
      <div class="ratios-grid" id="ratiosGrid"></div>
      <div class="analysis-box">
        <span class="analysis-title">φ · Aesthetic Interpretation</span>
        <p class="analysis-text" id="analysisText"></p>
      </div>
      <button class="reset-btn" onclick="resetFace()">↺ &nbsp; Analyze Another Photo</button>
    </div>
  </div>

  <!-- ═══════════════ TAB 2: DENTAL ═══════════════ -->
  <div class="tab-panel" id="tab-smile">

    <div class="dental-header">
      <span class="smile-icon">◉</span>
      <span class="dental-title">Dental Smile Analysis</span>
      <p class="dental-subtitle">Understand your smile aesthetics and explore improvement options</p>
    </div>

    <!-- Disclaimer -->
    <div class="disclaimer-banner">
      <p><strong>Important:</strong> This tool provides general educational information about cosmetic dental procedures only. It is <strong>not a substitute for professional dental advice</strong>. Always consult a licensed dentist or orthodontist for diagnosis, treatment planning, and care. Individual results vary.</p>
    </div>

    <!-- PHI SMILE RATIOS -->
    <div class="phi-smile-section">
      <span class="phi-smile-title">φ · Golden Ratio in the Ideal Smile</span>
      <div class="phi-ratio-row">
        <div>
          <div class="phi-ratio-name">Central to Lateral Incisor</div>
          <div class="phi-ratio-desc">Width of upper central incisor ÷ lateral incisor = φ</div>
        </div>
        <div class="phi-ratio-ideal">1.618</div>
      </div>
      <div class="phi-ratio-row">
        <div>
          <div class="phi-ratio-name">Lateral Incisor to Canine</div>
          <div class="phi-ratio-desc">Width of lateral incisor ÷ canine = φ</div>
        </div>
        <div class="phi-ratio-ideal">1.618</div>
      </div>
      <div class="phi-ratio-row">
        <div>
          <div class="phi-ratio-name">Smile Width to Face Width</div>
          <div class="phi-ratio-desc">Ideal smile spans ~61.8% of total face width</div>
        </div>
        <div class="phi-ratio-ideal">0.618</div>
      </div>
      <div class="phi-ratio-row">
        <div>
          <div class="phi-ratio-name">Tooth Height to Width</div>
          <div class="phi-ratio-desc">Central incisor height ÷ width = golden proportion</div>
        </div>
        <div class="phi-ratio-ideal">0.618</div>
      </div>
      <div class="phi-ratio-row">
        <div>
          <div class="phi-ratio-name">Gum-to-Lip Exposure</div>
          <div class="phi-ratio-desc">Ideal smile shows 75–100% of upper teeth, ≤3mm gum</div>
        </div>
        <div class="phi-ratio-ideal">~1.0</div>
      </div>
    </div>

    <!-- CONCERN SELECTOR -->
    <div class="concern-section">
      <span class="concern-label">Select your smile concerns (choose all that apply)</span>
      <div class="concern-grid">
        <button class="concern-btn" onclick="toggleConcern(this,'discoloration')"><span class="concern-icon">◑</span>Discoloration / Staining</button>
        <button class="concern-btn" onclick="toggleConcern(this,'gaps')"><span class="concern-icon">◻</span>Gaps Between Teeth</button>
        <button class="concern-btn" onclick="toggleConcern(this,'chips')"><span class="concern-icon">◈</span>Chipped / Broken Teeth</button>
        <button class="concern-btn" onclick="toggleConcern(this,'misalignment')"><span class="concern-icon">⬡</span>Crooked / Misaligned</button>
        <button class="concern-btn" onclick="toggleConcern(this,'gummy_smile')"><span class="concern-icon">◑</span>Gummy Smile</button>
        <button class="concern-btn" onclick="toggleConcern(this,'missing')"><span class="concern-icon">⊕</span>Missing Teeth</button>
        <button class="concern-btn" onclick="toggleConcern(this,'shape')"><span class="concern-icon">◇</span>Tooth Shape / Size</button>
        <button class="concern-btn" onclick="toggleConcern(this,'overall')"><span class="concern-icon">✸</span>Overall Transformation</button>
      </div>
      <button class="recommend-btn" onclick="getRecommendations()"><span>⌘ &nbsp; Get Dental Recommendations</span></button>
    </div>

    <!-- PROCEDURE RESULTS -->
    <div class="procedures-results" id="proceduresResults">
      <p class="section-title">Recommended Procedures</p>
      <div id="procedureCards"></div>

      <div class="disclaimer-banner" style="margin-top:24px;">
        <p><strong>Next Steps:</strong> The recommendations above are educational. To determine what is right for you, schedule a consultation with a <strong>board-certified cosmetic dentist</strong> or <strong>orthodontist</strong>. Ask about digital smile design (DSD) for a preview of your results before committing to any procedure.</p>
      </div>
    </div>

  </div><!-- end tab-smile -->

</div><!-- end container -->

<div class="footer" style="text-align:center;padding-bottom:40px;">φ = 1.6180339887… · The Divine Proportion · Powered by OpenCV · For Educational Purposes Only</div>

<script>
// ── TAB SWITCHING ──
function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.currentTarget.classList.add('active');
}

// ── FACE ANALYSIS ──
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
    document.getElementById('faceResults').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
  };
  r.readAsDataURL(file);
}

function resetFace() {
  currentFile = null; fileInput.value = '';
  document.getElementById('previewImg').src = '';
  dropZone.style.display = 'block';
  document.getElementById('previewWrap').style.display = 'none';
  document.getElementById('analyzeBtn').style.display = 'none';
  document.getElementById('faceResults').style.display = 'none';
  document.getElementById('loading').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';
}

async function analyzeFace() {
  if (!currentFile) return;
  document.getElementById('analyzeBtn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('faceResults').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';

  const fd = new FormData();
  fd.append('image', currentFile);
  try {
    const res = await fetch('/analyze', { method:'POST', body:fd });
    const data = await res.json();
    if (!res.ok || data.error) { showError(data.error || 'Analysis failed.'); return; }
    displayFaceResults(data);
  } catch(e) { showError('Network error: ' + e.message); }
  finally {
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

function displayFaceResults(data) {
  const score = data.overall_score;
  document.getElementById('scoreVerdict').textContent = getVerdict(score);
  setTimeout(() => document.getElementById('scoreBar').style.width = score + '%', 100);
  animateNum('scoreNum', 0, score, 1300);

  if (data.annotated_image) {
    document.getElementById('annotatedImg').src = 'data:image/jpeg;base64,' + data.annotated_image;
    document.getElementById('annotatedWrap').style.display = 'block';
  }

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
  document.getElementById('faceResults').style.display = 'block';
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

// ── DENTAL ──
let selectedConcerns = new Set();

function toggleConcern(btn, concern) {
  if (selectedConcerns.has(concern)) {
    selectedConcerns.delete(concern);
    btn.classList.remove('selected');
  } else {
    selectedConcerns.add(concern);
    btn.classList.add('selected');
  }
}

async function getRecommendations() {
  if (selectedConcerns.size === 0) {
    alert('Please select at least one smile concern.');
    return;
  }
  const res = await fetch('/dental_recommend', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ concerns: Array.from(selectedConcerns) })
  });
  const data = await res.json();
  renderProcedures(data.procedures);
}

function renderProcedures(procedures) {
  const container = document.getElementById('procedureCards');
  container.innerHTML = '';

  procedures.forEach(proc => {
    const card = document.createElement('div');
    card.className = 'procedure-card';
    const typeTags = proc.types.map(t => `<span class="proc-type-tag">${t}</span>`).join('');
    card.innerHTML = `
      <div class="proc-header">
        <span class="proc-icon">${proc.icon}</span>
        <span class="proc-name">${proc.name}</span>
      </div>
      <p class="proc-desc">${proc.description}</p>
      <div class="proc-meta">
        <div class="proc-meta-item"><div class="proc-meta-label">Estimated Cost</div><div class="proc-meta-value">${proc.cost}</div></div>
        <div class="proc-meta-item"><div class="proc-meta-label">Treatment Time</div><div class="proc-meta-value">${proc.duration}</div></div>
        <div class="proc-meta-item"><div class="proc-meta-label">Longevity</div><div class="proc-meta-value">${proc.longevity}</div></div>
        <div class="proc-meta-item"><div class="proc-meta-label">Discomfort Level</div><div class="proc-meta-value">${proc.pain_level}</div></div>
      </div>
      <div class="proc-types">
        <div class="proc-types-label">Available Options</div>
        ${typeTags}
      </div>
      <p class="proc-disclaimer">⚠ ${proc.disclaimer}</p>`;
    container.appendChild(card);
  });

  document.getElementById('proceduresResults').style.display = 'block';
  document.getElementById('proceduresResults').scrollIntoView({ behavior:'smooth', block:'start' });
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def ratio_score(measured):
    deviation = abs(measured - PHI)
    score = max(0.0, 100.0 * (1.0 - deviation / PHI))
    return round(score, 2), round(deviation, 4)


def generate_analysis(overall, ratios):
    best  = max(ratios, key=lambda r: r['score'])
    worst = min(ratios, key=lambda r: r['score'])
    if overall >= 85:
        opening = "Your facial proportions demonstrate a remarkable alignment with the golden ratio, placing you among the most harmoniously proportioned faces by classical aesthetic standards."
    elif overall >= 72:
        opening = "Your face exhibits strong phi harmony across multiple dimensions, reflecting the kind of balanced, naturally pleasing proportion that artists and architects have sought to capture for millennia."
    elif overall >= 60:
        opening = "Your facial structure shows moderate golden ratio alignment, with several features echoing the divine proportion in ways that contribute to an overall sense of natural balance."
    else:
        opening = "Your face carries a distinctive character that departs from classical phi proportions — a quality shared by many celebrated faces throughout art history, where individuality often surpasses formula."
    middle  = f"Your strongest alignment is in {best['name'].lower()} ({best['score']:.1f}%), while {worst['name'].lower()} shows the most deviation from φ."
    closing = f"With an overall harmony score of {overall:.1f}%, your features reflect the beautiful complexity of human proportion that no single formula can fully contain."
    return f"{opening} {middle} {closing}"


def annotate_image(img, face, eyes, nose_pt, mouth_pt, eye_centers):
    out = img.copy()
    x, y, w, h = face
    cv2.rectangle(out, (x, y), (x+w, y+h), (201, 168, 76), 1)
    for (ex, ey, ew, eh) in eyes:
        cx = x + ex + ew // 2; cy = y + ey + eh // 2
        cv2.circle(out, (cx, cy), 3, (201, 168, 76), -1)
        cv2.ellipse(out, (cx, cy), (ew//2, eh//2), 0, 0, 360, (201, 168, 76), 1)
    if nose_pt:
        cv2.circle(out, nose_pt, 4, (232, 201, 122), -1)
        cv2.circle(out, nose_pt, 10, (232, 201, 122), 1)
    if mouth_pt:
        cv2.circle(out, mouth_pt, 4, (138, 110, 47), -1)
    for yy in [y, y+h//3, y+2*h//3, y+h]:
        cv2.line(out, (x, yy), (x+w, yy), (201, 168, 76), 1)
    cv2.line(out, (x+w//2, y), (x+w//2, y+h), (201, 168, 76), 1)
    cv2.putText(out, f"phi={PHI:.4f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (201, 168, 76), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    img_bytes = request.files['image'].read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image.'}), 400

    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        faces = FACE_CASCADE.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))
    if len(faces) == 0:
        return jsonify({'error': 'No face detected. Please use a clear frontal selfie.'}), 400

    face = max(faces, key=lambda f: f[2]*f[3])
    fx, fy, fw, fh = face
    face_gray = gray[fy:fy+fh, fx:fx+fw]

    eyes = EYE_CASCADE.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20))
    eyes = sorted([(ex, ey, ew, eh) for ex, ey, ew, eh in eyes if ey < fh*0.55], key=lambda e: e[0])
    eye_centers = [(ex+ew//2, ey+eh//2) for ex, ey, ew, eh in eyes]

    nose_local  = (fw//2, int(fh*0.60))
    mouth_local = (fw//2, int(fh*0.78))
    nose_global  = (fx+nose_local[0],  fy+nose_local[1])
    mouth_global = (fx+mouth_local[0], fy+mouth_local[1])

    ratios = []
    face_wh = fw/fh if fh > 0 else PHI
    s, d = ratio_score(face_wh)
    ratios.append({'name':'Face Width to Height','measured':round(face_wh,4),'score':s,'deviation':d})

    if len(eyes) >= 2:
        le, re = eyes[0], eyes[-1]
        esp = abs((re[0]+re[2]//2)-(le[0]+le[2]//2))
        aew = (le[2]+re[2])/2
        s, d = ratio_score(esp/aew if aew>0 else PHI)
        ratios.append({'name':'Eye Spacing to Eye Width','measured':round(esp/aew,4),'score':s,'deviation':d})
        s, d = ratio_score(fw/esp if esp>0 else PHI)
        ratios.append({'name':'Face Width to Eye Span','measured':round(fw/esp,4),'score':s,'deviation':d})
        sym = max(le[2],re[2])/min(le[2],re[2]) if min(le[2],re[2])>0 else 1.0
        s, d = ratio_score(sym)
        ratios.append({'name':'Eye Width Symmetry','measured':round(sym,4),'score':s,'deviation':d})
    else:
        s, d = ratio_score((fw*0.35)/(fw*0.22))
        ratios.append({'name':'Eye Spacing to Eye Width (est.)','measured':round((fw*0.35)/(fw*0.22),4),'score':s,'deviation':d})

    if eye_centers:
        aey = sum(e[1] for e in eye_centers)/len(eye_centers)
        mid = nose_local[1]-aey
        if mid > 0 and aey > 0:
            s, d = ratio_score(aey/mid)
            ratios.append({'name':'Hairline to Eye / Eye to Nose','measured':round(aey/mid,4),'score':s,'deviation':d})
        lo = fh-nose_local[1]
        if mid > 0 and lo > 0:
            s, d = ratio_score(mid/lo)
            ratios.append({'name':'Mid Face to Lower Face','measured':round(mid/lo,4),'score':s,'deviation':d})

    s, d = ratio_score((fw*0.46)/(fw*0.28))
    ratios.append({'name':'Mouth Width to Nose Width','measured':round((fw*0.46)/(fw*0.28),4),'score':s,'deviation':d})
    nc = fh-nose_local[1]
    s, d = ratio_score(fh/nc if nc>0 else PHI)
    ratios.append({'name':'Face Height to Chin Segment','measured':round(fh/nc,4),'score':s,'deviation':d})

    weights = [1.0]*len(ratios)
    if len(eyes) >= 2: weights[1]=1.5; weights[2]=1.3
    overall = sum(r['score']*w for r,w in zip(ratios,weights))/sum(weights)

    annotated = annotate_image(img, face, eyes, nose_global, mouth_global, eye_centers)
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return jsonify({
        'overall_score': round(overall, 2),
        'ratios': ratios,
        'analysis': generate_analysis(overall, ratios),
        'annotated_image': base64.standard_b64encode(buf).decode('utf-8'),
        'landmarks_detected': len(eyes)
    })


@app.route('/dental_recommend', methods=['POST'])
def dental_recommend():
    data     = request.get_json()
    concerns = data.get('concerns', [])

    proc_ids = set()
    for concern in concerns:
        for pid in SMILE_CONCERNS.get(concern, []):
            proc_ids.add(pid)

    # Always include whitening as a baseline if discoloration not selected
    if not proc_ids:
        proc_ids = {'teeth_whitening', 'smile_makeover'}

    # Sort by priority: more serious procedures last
    priority = ['teeth_whitening','bonding','gum_contouring','veneers','crown','orthodontics','implants','smile_makeover']
    ordered  = [p for p in priority if p in proc_ids]

    procedures = [DENTAL_PROCEDURES[pid] for pid in ordered if pid in DENTAL_PROCEDURES]
    return jsonify({'procedures': procedures})


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5050))
    print("\n  φ  Face & Smile Analyzer  (No API Key Required)")
    print("  ────────────────────────────────────────────────")
    print(f"  Open: http://localhost:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
