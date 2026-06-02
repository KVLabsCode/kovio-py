"""The single-page HTML served by :class:`BrowserScreenAdapter`.

Kept in its own module so ``screen.py`` stays focused on adapter logic. The
page polls ``/api/state`` and renders one of two layers — a breathing idle
wordmark when no creative is up, or the live creative (with a save QR and an
engage hint) when one is. Clicking anywhere over a live creative POSTs to
``/api/engage``.

Brand tokens mirror the Kovio web app (cream ``#f2ecdc`` page, rust ``#c57a3f``
accent, italic Instrument Serif wordmark with a Georgia fallback).
"""
from __future__ import annotations

PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>kovio — robot screen</title>
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
  <style>
    :root {
      --page: #f2ecdc; --ink: #1c1a18; --ink-3: #8a7f6f;
      --rust: #c57a3f; --rust-dark: #9f5e2a; --card: #f5efe3;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; overflow: hidden; background: var(--page); }
    body {
      font-family: "Instrument Serif", Georgia, serif;
      -webkit-font-smoothing: antialiased; cursor: default;
      user-select: none; -webkit-user-select: none;
    }
    .layer { position: fixed; inset: 0; }
    .hidden { display: none !important; }

    /* ---- idle layer ---- */
    #idle {
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      background: var(--page); color: var(--ink);
    }
    .word {
      font-style: italic; font-weight: 400;
      font-size: clamp(96px, 22vw, 320px); line-height: 1;
      letter-spacing: -0.01em;
      animation: breathe 4.2s ease-in-out infinite;
    }
    @keyframes breathe {
      0%, 100% { opacity: .5;  transform: scale(1); }
      50%      { opacity: 1;   transform: scale(1.035); }
    }
    .status {
      margin-top: 4vh; font-family: ui-monospace, "JetBrains Mono", monospace;
      font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
      color: var(--ink-3);
    }
    .status .dot {
      display: inline-block; width: 7px; height: 7px; border-radius: 50%;
      background: var(--rust); margin-right: 8px; vertical-align: middle;
      animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

    /* ---- creative layer ---- */
    #creative { background: #000; }
    #frame {
      position: absolute; inset: 0; width: 100%; height: 100%;
      border: 0; pointer-events: none;   /* clicks fall through to #catcher */
    }
    #catcher { position: absolute; inset: 0; z-index: 2; cursor: pointer; }
    .qr {
      position: fixed; top: 26px; right: 26px; z-index: 3;
      background: #fff; padding: 12px; border-radius: 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
      line-height: 0;
    }
    .qr svg, .qr img { width: 124px; height: 124px; display: block; }
    .hint {
      position: fixed; left: 50%; bottom: 30px; transform: translateX(-50%);
      z-index: 3; display: flex; align-items: center; gap: 10px;
      padding: 11px 20px; border-radius: 999px;
      background: rgba(242,236,220,.92); color: var(--ink);
      font-family: ui-monospace, "JetBrains Mono", monospace;
      font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase;
      box-shadow: 0 6px 20px rgba(0,0,0,.3); backdrop-filter: blur(4px);
    }
    .hint .dot {
      width: 7px; height: 7px; border-radius: 50%; background: var(--rust);
    }

    /* ---- engagement checkmark ---- */
    #check {
      position: fixed; inset: 0; z-index: 4; display: none;
      align-items: center; justify-content: center; pointer-events: none;
    }
    #check.show { display: flex; }
    #check .badge {
      width: 168px; height: 168px; border-radius: 50%;
      background: var(--rust); color: #fff;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 18px 50px rgba(159,94,42,.55);
      animation: pop 900ms cubic-bezier(.2,.9,.25,1) forwards;
    }
    #check svg { width: 92px; height: 92px; }
    #check path {
      stroke: #fff; stroke-width: 9; fill: none;
      stroke-linecap: round; stroke-linejoin: round;
      stroke-dasharray: 120; stroke-dashoffset: 120;
      animation: draw 460ms ease-out 80ms forwards;
    }
    @keyframes pop {
      0%   { opacity: 0; transform: scale(.55); }
      18%  { opacity: 1; transform: scale(1); }
      72%  { opacity: 1; transform: scale(1); }
      100% { opacity: 0; transform: scale(1.04); }
    }
    @keyframes draw { to { stroke-dashoffset: 0; } }
  </style>
</head>
<body>
  <div id="idle" class="layer">
    <div class="word">kovio</div>
    <div class="status"><span class="dot"></span><span id="statusText">idle · waiting for attention</span></div>
  </div>

  <div id="creative" class="layer hidden">
    <iframe id="frame" referrerpolicy="no-referrer"></iframe>
    <div id="catcher"></div>
    <div class="qr" id="qr"></div>
    <div class="hint"><span class="dot"></span><span>Tap to engage · scan to save</span></div>
  </div>

  <div id="check"><div class="badge">
    <svg viewBox="0 0 100 100"><path d="M26 52 L44 70 L76 32"></path></svg>
  </div></div>

  <script>
    const idle = document.getElementById('idle');
    const creative = document.getElementById('creative');
    const frame = document.getElementById('frame');
    const qr = document.getElementById('qr');
    const check = document.getElementById('check');
    const catcher = document.getElementById('catcher');

    let showing = false;
    let currentSrc = null;
    let qrLoaded = false;

    function render(s) {
      if (s.qr_svg && !qrLoaded) { qr.innerHTML = s.qr_svg; qrLoaded = true; }
      if (s.showing) {
        if (s.creative_src && s.creative_src !== currentSrc) {
          currentSrc = s.creative_src;
          frame.src = s.creative_src;   // only reload when the creative changes
        }
        if (!showing) { idle.classList.add('hidden'); creative.classList.remove('hidden'); }
        showing = true;
      } else {
        if (showing) { creative.classList.add('hidden'); idle.classList.remove('hidden'); }
        showing = false;
      }
    }

    async function poll() {
      try {
        const r = await fetch('/api/state', { cache: 'no-store' });
        render(await r.json());
      } catch (e) { /* server momentarily unavailable — keep last frame */ }
    }

    let busy = false;
    async function engage() {
      if (!showing || busy) return;
      busy = true;
      check.classList.remove('show');
      void check.offsetWidth;          // restart the CSS animation
      check.classList.add('show');
      setTimeout(() => { check.classList.remove('show'); busy = false; }, 950);
      try { await fetch('/api/engage', { method: 'POST' }); } catch (e) {}
    }

    catcher.addEventListener('click', engage);
    setInterval(poll, 700);
    poll();
  </script>
</body>
</html>
"""
