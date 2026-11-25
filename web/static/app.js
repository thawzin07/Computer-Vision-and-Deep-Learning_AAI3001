(() => {
  "use strict";

  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
  ];

  const UPPER_POSE_EDGES = [
    [7, 8], [7, 9], [9, 11], [8, 10], [10, 12],
    [7, 13], [8, 14], [13, 14], [5, 6],
  ];

  const state = {
    activeView: "words", // 'words' | 'letters' | 'demo'
    demoSubTab: "words", // 'words' | 'letters'
    mode: "words",
    autoWordMode: true,
    manualRecording: false,
    streaming: false,
    autoAppend: true,
    polishEnglish: false,
    selectedDeviceId: "",
    status: null,
    demoSigns: null,
    demoLetters: null,
    calCounts: null,
    calBusy: false,
    confirmedFlashTimer: null,
    tuning: {
      word_conf: 0.68,
      word_motion: 0.014,
      letter_conf: 0.72,
      letter_hold: 0.65,
    },
    ws: null,
    mediaStream: null,
    inFlight: false,
    lastSendAt: 0,
    fpsTimes: [],
    rafId: null,
    captureCanvas: document.createElement("canvas"),
    tokens: [], // { text, kind: 'word' | 'letter' }
    joinLetters: false,
    pendingSpace: false,
  };

  const els = {
    tabWords: document.getElementById("tabWords"),
    tabLetters: document.getElementById("tabLetters"),
    tabDemo: document.getElementById("tabDemo"),
    badgeDeploy: document.getElementById("badgeDeploy"),
    packSelect: document.getElementById("packSelect"),
    badgeWords: document.getElementById("badgeWords"),
    badgeLetters: document.getElementById("badgeLetters"),
    btnReload: document.getElementById("btnReload"),

    onboardingStrip: document.getElementById("onboardingStrip"),
    btnOnboardingDemo: document.getElementById("btnOnboardingDemo"),
    btnDismissTip: document.getElementById("btnDismissTip"),

    liveView: document.getElementById("liveView"),
    demoView: document.getElementById("demoView"),
    demoSubWords: document.getElementById("demoSubWords"),
    demoSubLetters: document.getElementById("demoSubLetters"),
    demoFilter: document.getElementById("demoFilter"),
    btnTryWords: document.getElementById("btnTryWords"),
    demoLetterChartCard: document.getElementById("demoLetterChartCard"),
    demoLetterChartImg: document.getElementById("demoLetterChartImg"),
    demoGrid: document.getElementById("demoGrid"),

    stageBox: document.getElementById("stageBox"),
    liveVideo: document.getElementById("liveVideo"),
    overlayCanvas: document.getElementById("overlayCanvas"),
    previewImg: document.getElementById("previewImg"),
    stageEmpty: document.getElementById("stageEmpty"),
    stagePill: document.getElementById("stagePill"),

    btnCamera: document.getElementById("btnCamera"),
    cameraSelect: document.getElementById("cameraSelect"),
    btnUpload: document.getElementById("btnUpload"),
    fileInput: document.getElementById("fileInput"),
    wordSegControls: document.getElementById("wordSegControls"),
    btnAutoMode: document.getElementById("btnAutoMode"),
    btnManualCapture: document.getElementById("btnManualCapture"),
    btnTune: document.getElementById("btnTune"),
    btnCalibrate: document.getElementById("btnCalibrate"),
    chkMirror: document.getElementById("chkMirror"),
    chkSkeleton: document.getElementById("chkSkeleton"),

    tunePanel: document.getElementById("tunePanel"),
    btnTuneReset: document.getElementById("btnTuneReset"),
    rngWordConf: document.getElementById("rngWordConf"),
    valWordConf: document.getElementById("valWordConf"),
    rngWordMotion: document.getElementById("rngWordMotion"),
    valWordMotion: document.getElementById("valWordMotion"),
    rngLetterConf: document.getElementById("rngLetterConf"),
    valLetterConf: document.getElementById("valLetterConf"),
    rngLetterHold: document.getElementById("rngLetterHold"),
    valLetterHold: document.getElementById("valLetterHold"),

    calibratePanel: document.getElementById("calibratePanel"),
    calLabelSelect: document.getElementById("calLabelSelect"),
    calCustomLabel: document.getElementById("calCustomLabel"),
    calSignerId: document.getElementById("calSignerId"),
    calLighting: document.getElementById("calLighting"),
    calBackground: document.getElementById("calBackground"),
    calHandedness: document.getElementById("calHandedness"),
    calTrimRow: document.getElementById("calTrimRow"),
    rngTrimStart: document.getElementById("rngTrimStart"),
    valTrimStart: document.getElementById("valTrimStart"),
    rngTrimEnd: document.getElementById("rngTrimEnd"),
    valTrimEnd: document.getElementById("valTrimEnd"),
    calCountBadge: document.getElementById("calCountBadge"),
    btnCalRecord: document.getElementById("btnCalRecord"),
    calStatusMsg: document.getElementById("calStatusMsg"),
    calRecentList: document.getElementById("calRecentList"),
    btnCalExport: document.getElementById("btnCalExport"),

    missingBanner: document.getElementById("missingBanner"),
    missingBannerText: document.getElementById("missingBannerText"),
    lblLatency: document.getElementById("lblLatency"),
    lblPred: document.getElementById("lblPred"),
    lblConf: document.getElementById("lblConf"),
    lblStatus: document.getElementById("lblStatus"),
    topkList: document.getElementById("topkList"),
    lblMeterTitle: document.getElementById("lblMeterTitle"),
    lblMeterHint: document.getElementById("lblMeterHint"),
    meterFill: document.getElementById("meterFill"),
    meterMarker: document.getElementById("meterMarker"),

    telemetryCol: document.getElementById("telemetryCol"),
    liveLetterChartCard: document.getElementById("liveLetterChartCard"),
    vocabCard: document.getElementById("vocabCard"),
    bottomRow: document.getElementById("bottomRow"),
    lblVocabTitle: document.getElementById("lblVocabTitle"),
    lblVocabMeta: document.getElementById("lblVocabMeta"),
    vocabGrid: document.getElementById("vocabGrid"),

    chkAutoAppend: document.getElementById("chkAutoAppend"),
    chkPolish: document.getElementById("chkPolish"),
    transcriptBox: document.getElementById("transcriptBox"),
    transcriptRaw: document.getElementById("transcriptRaw"),
    btnSpace: document.getElementById("btnSpace"),
    btnUndo: document.getElementById("btnUndo"),
    btnClear: document.getElementById("btnClear"),
    btnCopy: document.getElementById("btnCopy"),
    btnSpeak: document.getElementById("btnSpeak"),
  };

  function setPill(el, ready, text) {
    el.classList.toggle("ready", ready);
    el.classList.toggle("pending", !ready);
    el.querySelector(".pill-text").textContent = text;
  }

  function applyStatusPayload(data) {
    state.status = data;
    const w = state.status.words;
    const l = state.status.letters;
    const dep = state.status.deployment || {};

    const wAcc = typeof w.test_top1_acc === "number" ? ` · ${(w.test_top1_acc * 100).toFixed(1)}%` : "";
    const lAcc = typeof l.test_top1_acc === "number" ? ` · ${(l.test_top1_acc * 100).toFixed(1)}%` : "";

    setPill(els.badgeWords, w.available, w.available ? `Words Ready${wAcc}` : "Words Missing");
    setPill(els.badgeLetters, l.available, l.available ? `Letters Ready${lAcc}` : "Letters Pending");

    const envLabel = (dep.env || "local").toUpperCase();
    const activePrec = (state.mode === "words" ? w.precision : l.precision) || (dep.prefer_int8 ? "int8" : "fp32");
    els.badgeDeploy.classList.toggle("int8", String(activePrec).toUpperCase() === "INT8");
    els.badgeDeploy.querySelector(".pill-text").textContent = `${envLabel} · ${String(activePrec).toUpperCase()}`;

    const wTabCount = els.tabWords.querySelector(".seg-count");
    if (wTabCount && Array.isArray(w.labels)) {
      wTabCount.textContent = String(w.labels.length);
    }

    if (Array.isArray(dep.packs) && dep.packs.length) {
      els.packSelect.innerHTML = "";
      for (const pk of dep.packs) {
        const opt = document.createElement("option");
        opt.value = pk.id;
        opt.textContent = pk.name;
        els.packSelect.appendChild(opt);
      }
      els.packSelect.value = dep.active_pack || "default";
    }

    renderModeShell();
  }

  async function fetchStatus(endpoint = "/api/status", method = "GET") {
    const resp = await fetch(endpoint, { method });
    if (!resp.ok) throw new Error("Failed to reach server status");
    applyStatusPayload(await resp.json());
  }

  function renderModeShell() {
    const isWords = state.mode === "words";
    const isDemo = state.activeView === "demo";
    els.tabWords.classList.toggle("active", !isDemo && isWords);
    els.tabLetters.classList.toggle("active", !isDemo && !isWords);
    els.tabDemo.classList.toggle("active", isDemo);
    els.tabWords.setAttribute("aria-selected", String(!isDemo && isWords));
    els.tabLetters.setAttribute("aria-selected", String(!isDemo && !isWords));
    els.tabDemo.setAttribute("aria-selected", String(isDemo));
    els.liveView.classList.toggle("hidden", isDemo);
    els.demoView.classList.toggle("hidden", !isDemo);

    els.btnUpload.textContent = isWords ? "Upload Video" : "Upload Image";
    els.fileInput.accept = isWords ? "video/*" : "image/*";
    els.wordSegControls.classList.toggle("hidden", !isWords);

    els.liveLetterChartCard.classList.toggle("hidden", isWords);
    els.bottomRow.classList.toggle("split-bottom", !isWords);
    (isWords ? els.telemetryCol : els.bottomRow).appendChild(els.vocabCard);

    els.lblMeterTitle.textContent = isWords ? "HAND MOTION ENERGY" : "HOLD-TO-CONFIRM";
    els.lblMeterHint.textContent = isWords ? (state.autoWordMode ? "Auto gate" : "Manual") : "Dynamic J & Z active";
    els.meterMarker.classList.toggle("hidden", !isWords);

    if (state.status) {
      const info = isWords ? state.status.words : state.status.letters;
      if (!info.available) {
        els.missingBanner.classList.remove("hidden");
        els.missingBannerText.innerHTML = isWords
          ? "Place <code>asl_words_conformer.onnx</code> &amp; <code>word_model_config.json</code> into <code>models/</code>, then click <strong>Reload Models</strong>."
          : "Notebook 2 model isn't added yet. Drop <code>asl_letters_hybrid.onnx</code> &amp; <code>letter_model_config.json</code> into <code>models/</code>, then click <strong>Reload Models</strong>. Hand tracking preview still works!";
      } else {
        els.missingBanner.classList.add("hidden");
      }

      const acc = typeof info.test_top1_acc === "number" ? `Top-1 ${(info.test_top1_acc * 100).toFixed(1)}%` : "";
      const f1 = typeof info.test_macro_f1 === "number" ? `F1 ${(info.test_macro_f1 * 100).toFixed(1)}%` : "";
      els.lblVocabTitle.textContent = isWords ? `${info.labels.length} TRAINED WORDS` : `${info.labels.length} FINGERSPELLING LETTERS`;
      els.lblVocabMeta.textContent = [acc, f1].filter(Boolean).join(" · ");

      els.vocabGrid.className = `vocab-grid ${isWords ? "words" : "letters"}`;
      els.vocabGrid.innerHTML = "";
      for (const lbl of info.labels) {
        const chip = document.createElement("div");
        chip.className = "vocab-chip";
        chip.dataset.label = lbl.toLowerCase();
        chip.textContent = lbl;
        els.vocabGrid.appendChild(chip);
      }

      syncCalibrationControls();
    }

    updatePredictionUI({
      label: "",
      confidence: 0,
      topk: [],
      latency_ms: 0,
      motion_energy: 0,
      hold_progress: 0,
      status: state.streaming ? "Listening…" : "Ready · Start camera or upload a file",
      confirmed_token: null,
    });
  }

  function updatePredictionUI(pred) {
    const hasLabel = pred.label && pred.label !== "---";
    els.lblPred.textContent = hasLabel ? pred.label.toUpperCase() : "—";
    els.lblPred.className = "pred-value" + (pred.confirmed_token ? " confirmed" : hasLabel ? " preview" : "");
    els.lblConf.textContent = pred.confidence > 0 ? `${(pred.confidence * 100).toFixed(1)}%` : "—";
    els.lblStatus.textContent = pred.status || "Ready";
    els.lblLatency.textContent = pred.latency_ms > 0 ? `${pred.latency_ms.toFixed(1)} ms` : "— ms";

    // Top-3 bars
    els.topkList.innerHTML = "";
    const topk = pred.topk || [];
    for (let i = 0; i < 3; i++) {
      const item = topk[i];
      const lbl = item ? item[0].toUpperCase() : "—";
      const pct = item ? Math.max(0, Math.min(100, item[1] * 100)) : 0;

      const row = document.createElement("div");
      row.className = `topk-row${i === 0 && item ? " top1" : ""}`;
      row.innerHTML = `
        <div class="topk-name">${lbl}</div>
        <div class="topk-track"><div class="topk-fill" style="width:${pct.toFixed(1)}%"></div></div>
        <div class="topk-pct">${item ? pct.toFixed(0) + "%" : "—"}</div>
      `;
      els.topkList.appendChild(row);
    }

    // Meter bar
    if (state.mode === "words") {
      const thresh = state.tuning.word_motion || state.status?.words?.motion_threshold || 0.014;
      const ratio = Math.min(100, (Number(pred.motion_energy || 0) / 0.06) * 100);
      els.meterMarker.style.left = `${Math.min(98, (thresh / 0.06) * 100).toFixed(1)}%`;
      els.meterFill.style.width = `${ratio.toFixed(1)}%`;
      els.meterFill.classList.toggle("active", (pred.motion_energy || 0) >= thresh);
    } else {
      const pct = Math.min(100, Number(pred.hold_progress || 0) * 100);
      els.meterFill.style.width = `${pct.toFixed(1)}%`;
      els.meterFill.classList.toggle("active", pct >= 95);
    }

    // Highlight vocabulary pill
    const active = hasLabel ? pred.label.toLowerCase() : "";
    for (const chip of els.vocabGrid.children) {
      const isMatch = active && chip.dataset.label === active;
      chip.classList.toggle("confirmed", Boolean(isMatch && pred.confirmed_token));
      chip.classList.toggle("preview", Boolean(isMatch && !pred.confirmed_token));
    }
  }

  function polishAslGloss(tokens) {
    if (!tokens.length) return "";
    const rawWords = tokens.map((t) => (t.kind === "letter" ? t.text.toUpperCase() : t.text.toLowerCase()));
    const joined = rawWords.join(" ").trim();
    const mapRules = [
      [/^hello(\s+friend)?\s+want\s+water(\s+please)?$/, (_, f, p) => `Hello${f ? " friend" : ""}, I want water${p ? ", please" : ""}.`],
      [/^want\s+([a-z]+)(\s+please)?$/, (_, obj, p) => `I want ${obj}${p ? ", please" : ""}.`],
      [/^more\s+([a-z]+)(\s+please)?$/, (_, obj, p) => `I would like more ${obj}${p ? ", please" : ""}.`],
      [/^thank you\s+help$/, () => "Thank you for your help."],
      [/^help(\s+please)?$/, (_, p) => `Could you help me${p ? ", please" : ""}?`],
      [/^stop(\s+please)?$/, (_, p) => `${p ? "Please stop" : "Stop"}.`],
      [/^please\s+([a-z\s]+)$/, (_, rest) => `Please ${rest.trim()}.`],
    ];
    for (const [pat, fmt] of mapRules) {
      const m = joined.match(pat);
      if (m) return fmt(...m);
    }
    const clean = joined
      .replace(/\bwant\b/g, "I want")
      .replace(/\blove\b/g, "I love")
      .replace(/\s+/g, " ")
      .trim();
    const titled = clean.charAt(0).toUpperCase() + clean.slice(1);
    return /[.!?]$/.test(titled) ? titled : `${titled}.`;
  }

  function getActiveTranscriptText() {
    const rawText = state.tokens.map((t) => t.text).join(" ").trim();
    if (!rawText) return "";
    return state.polishEnglish ? polishAslGloss(state.tokens) : rawText;
  }

  function renderTranscript() {
    const rawText = state.tokens.map((t) => t.text).join(" ");
    if (!rawText) {
      els.transcriptBox.textContent = "Confirmed words and fingerspelled letters appear here…";
      els.transcriptBox.classList.add("empty");
      els.transcriptRaw.classList.add("hidden");
      els.transcriptRaw.textContent = "";
      return;
    }
    if (state.polishEnglish) {
      els.transcriptBox.textContent = polishAslGloss(state.tokens);
      els.transcriptBox.classList.remove("empty");
      els.transcriptRaw.textContent = `Raw gloss: ${rawText}`;
      els.transcriptRaw.classList.remove("hidden");
    } else {
      els.transcriptBox.textContent = state.pendingSpace ? `${rawText} ·` : rawText;
      els.transcriptBox.classList.remove("empty");
      els.transcriptRaw.classList.add("hidden");
      els.transcriptRaw.textContent = "";
    }
  }

  function appendToken(token, kind) {
    if (!token || !state.autoAppend) return;
    state.pendingSpace = false;
    if (kind === "letter") {
      const clean = token.trim().toUpperCase();
      if (!clean) return;
      if (state.joinLetters && state.tokens.length && state.tokens[state.tokens.length - 1].kind === "letter") {
        state.tokens[state.tokens.length - 1].text += clean;
      } else {
        state.tokens.push({ text: clean, kind: "letter" });
      }
      state.joinLetters = true;
    } else {
      const clean = token.trim().toLowerCase();
      if (!clean) return;
      state.tokens.push({ text: clean, kind: "word" });
      state.joinLetters = false;
    }
    renderTranscript();
  }

  function clearStageOverlay() {
    const ctx = els.overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, els.overlayCanvas.width, els.overlayCanvas.height);
  }

  function drawOverlay(overlay) {
    const canvas = els.overlayCanvas;
    const ctx = canvas.getContext("2d");
    const rect = els.stageBox.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    if (!overlay || !els.chkSkeleton.checked) return;

    const mirror = els.chkMirror.checked;
    const mapPt = (pt) => [(mirror ? 1 - pt[0] : pt[0]) * w, pt[1] * h];

    if (overlay.pose) {
      ctx.strokeStyle = "rgba(37, 99, 235, 0.72)";
      ctx.lineWidth = 2;
      for (const [a, b] of UPPER_POSE_EDGES) {
        const [x1, y1] = mapPt(overlay.pose[a]);
        const [x2, y2] = mapPt(overlay.pose[b]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    const hands = [
      [overlay.left_hand, "#059669"],
      [overlay.right_hand, "#2563eb"],
    ];
    for (const [pts, col] of hands) {
      if (!pts) continue;
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.lineWidth = 2;
      for (const [a, b] of HAND_CONNECTIONS) {
        const [x1, y1] = mapPt(pts[a]);
        const [x2, y2] = mapPt(pts[b]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
      for (let i = 0; i < pts.length; i++) {
        const [px, py] = mapPt(pts[i]);
        const r = [4, 8, 12, 16, 20].includes(i) ? 4 : 2.5;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (overlay.hand_bbox) {
      let [x1, y1, x2, y2] = overlay.hand_bbox;
      if (mirror) [x1, x2] = [1 - x2, 1 - x1];
      ctx.strokeStyle = "#0ea5e9";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
    }
  }

  function syncSegmenterButtons() {
    els.btnAutoMode.textContent = state.autoWordMode ? "Auto Motion" : "Manual";
    els.btnAutoMode.classList.toggle("btn-primary", state.autoWordMode);
    els.btnAutoMode.classList.toggle("btn-outline", !state.autoWordMode);
    els.btnManualCapture.classList.toggle("hidden", state.autoWordMode);
    els.btnManualCapture.disabled = !state.streaming;
    els.btnManualCapture.textContent = state.manualRecording ? "Stop & Predict" : "Start Sign";
    els.btnManualCapture.classList.toggle("btn-danger-outline", state.manualRecording);
    els.btnManualCapture.classList.toggle("btn-emerald", !state.manualRecording);
    els.lblMeterHint.textContent = state.mode === "words" ? (state.autoWordMode ? "Auto gate" : "Manual") : "Dynamic J & Z active";
  }

  function sendWsControl(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
    }
  }

  async function refreshCameraList() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      if (cams.length <= 1) {
        els.cameraSelect.classList.add("hidden");
        return;
      }
      els.cameraSelect.innerHTML = "";
      cams.forEach((cam, idx) => {
        const opt = document.createElement("option");
        opt.value = cam.deviceId;
        opt.textContent = cam.label || `Camera ${idx + 1}`;
        els.cameraSelect.appendChild(opt);
      });
      if (state.selectedDeviceId) els.cameraSelect.value = state.selectedDeviceId;
      els.cameraSelect.classList.remove("hidden");
    } catch (_) {}
  }

  function triggerConfirmedFlash() {
    els.stageBox.classList.add("confirmed");
    if (state.confirmedFlashTimer) clearTimeout(state.confirmedFlashTimer);
    state.confirmedFlashTimer = setTimeout(() => els.stageBox.classList.remove("confirmed"), 650);
  }

  async function startCamera(deviceId = state.selectedDeviceId) {
    if (!navigator.mediaDevices?.getUserMedia) {
      els.lblStatus.textContent = "Camera API requires localhost or HTTPS in this browser.";
      return;
    }
    try {
      const videoConstraints = deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 960 }, height: { ideal: 720 } }
        : { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" };
      const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
      const activeTrack = stream.getVideoTracks()[0];
      if (activeTrack?.getSettings()?.deviceId) {
        state.selectedDeviceId = activeTrack.getSettings().deviceId;
      }
      state.mediaStream = stream;
      els.liveVideo.srcObject = stream;
      await els.liveVideo.play();
      await refreshCameraList();
      if (els.liveVideo.videoWidth > 0 && els.liveVideo.videoHeight > 0) {
        els.stageBox.style.aspectRatio = `${els.liveVideo.videoWidth} / ${els.liveVideo.videoHeight}`;
      }

      els.previewImg.classList.add("hidden");
      els.stageEmpty.classList.add("hidden");
      els.liveVideo.classList.remove("hidden");
      els.overlayCanvas.classList.remove("hidden");

      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
      ws.binaryType = "arraybuffer";
      state.ws = ws;
      state.streaming = true;
      state.inFlight = false;
      state.fpsTimes = [];

      els.btnCamera.textContent = "Stop Camera";
      els.btnCamera.classList.remove("btn-primary");
      els.btnCamera.classList.add("btn-danger-outline");
      syncSegmenterButtons();

      ws.onopen = () => {
        sendWsControl({
          type: "config",
          mode: state.mode,
          auto_word_mode: state.autoWordMode,
          return_overlay: els.chkSkeleton.checked,
          ...state.tuning,
        });
        els.stagePill.textContent = "LIVE";
        state.rafId = requestAnimationFrame(pumpFrames);
      };

      ws.onmessage = (evt) => {
        state.inFlight = false;
        const data = JSON.parse(evt.data);
        if (data.type === "prediction") {
          const now = performance.now();
          state.fpsTimes.push(now);
          while (state.fpsTimes.length && now - state.fpsTimes[0] > 1000) state.fpsTimes.shift();
          els.stagePill.textContent = `LIVE · ${state.fpsTimes.length} FPS`;

          if (data.segmenter) {
            state.autoWordMode = Boolean(data.segmenter.auto_mode);
            state.manualRecording = Boolean(data.segmenter.manual_recording);
            const isSigning = Boolean(data.segmenter.manual_recording || data.segmenter.word_state === "SIGNING");
            els.stageBox.classList.toggle("signing", isSigning);
            syncSegmenterButtons();
          }

          updatePredictionUI(data.pred);
          if (data.pred?.confirmed_token) {
            triggerConfirmedFlash();
            appendToken(data.pred.confirmed_token, data.pred.confirmed_kind);
          }
          drawOverlay(data.overlay);
        }
      };

      ws.onclose = () => {
        if (state.streaming) stopCamera();
      };
    } catch (err) {
      els.lblStatus.textContent = `Camera error: ${err.message || "Permission denied or no camera found"}`;
      stopCamera();
    }
  }

  function pumpFrames() {
    if (!state.streaming) return;
    const now = performance.now();
    const video = els.liveVideo;

    if (
      state.ws &&
      state.ws.readyState === WebSocket.OPEN &&
      !state.inFlight &&
      now - state.lastSendAt >= 50 &&
      video.readyState >= 2 &&
      video.videoWidth > 0
    ) {
      const targetW = 640;
      const targetH = Math.max(240, Math.round((video.videoHeight / video.videoWidth) * targetW));
      if (state.captureCanvas.width !== targetW || state.captureCanvas.height !== targetH) {
        state.captureCanvas.width = targetW;
        state.captureCanvas.height = targetH;
      }
      const ctx = state.captureCanvas.getContext("2d");
      ctx.drawImage(video, 0, 0, targetW, targetH);

      state.inFlight = true;
      state.lastSendAt = now;
      state.captureCanvas.toBlob(
        (blob) => {
          if (blob && state.ws && state.ws.readyState === WebSocket.OPEN && state.streaming) {
            state.ws.send(blob);
          } else {
            state.inFlight = false;
          }
        },
        "image/jpeg",
        0.75
      );
    }
    state.rafId = requestAnimationFrame(pumpFrames);
  }

  function stopCamera() {
    state.streaming = false;
    state.inFlight = false;
    state.manualRecording = false;
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
    if (state.ws) {
      try {
        state.ws.close();
      } catch (_) {}
      state.ws = null;
    }
    if (state.mediaStream) {
      for (const t of state.mediaStream.getTracks()) t.stop();
      state.mediaStream = null;
    }
    els.liveVideo.srcObject = null;
    els.liveVideo.classList.add("hidden");
    els.overlayCanvas.classList.add("hidden");
    els.stageBox.classList.remove("signing", "confirmed");
    els.stageBox.style.aspectRatio = "4 / 3";
    clearStageOverlay();
    if (els.previewImg.classList.contains("hidden")) {
      els.stageEmpty.classList.remove("hidden");
      els.stagePill.textContent = "IDLE";
    }
    els.btnCamera.textContent = "Start Camera";
    els.btnCamera.classList.add("btn-primary");
    els.btnCamera.classList.remove("btn-danger-outline");
    syncSegmenterButtons();
  }

  async function handleFileUpload(file) {
    if (!file) return;
    if (state.streaming) stopCamera();

    const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name);
    const useWords = isVideo ? true : false;
    if (useWords && state.mode !== "words") switchMode("words");
    if (!useWords && state.mode !== "letters") switchMode("letters");

    els.stagePill.textContent = "ANALYZING…";
    els.lblStatus.textContent = `Processing ${file.name}…`;

    const form = new FormData();
    form.append("file", file);
    const endpoint = useWords ? "/api/predict/video" : "/api/predict/image";

    try {
      const resp = await fetch(endpoint, { method: "POST", body: form });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Upload failed");

      if (data.preview_url) {
        els.previewImg.src = data.preview_url;
        els.previewImg.classList.remove("hidden");
        els.stageEmpty.classList.add("hidden");
      }
      els.stagePill.textContent = useWords
        ? `CLIP · ${data.active_frames || 0} ACTIVE FRAMES`
        : data.hand_present ? "IMAGE · HAND FOUND" : "IMAGE · NO HAND";

      if (data.ok) {
        const kind = useWords ? "word" : "letter";
        const confirmed = useWords ? (data.confident ? data.label : null) : data.label;
        updatePredictionUI({
          label: data.label,
          confidence: data.confidence,
          topk: data.topk,
          latency_ms: data.latency_ms,
          motion_energy: 0,
          hold_progress: 1,
          status: `${useWords ? "Clip top prediction" : "Letter prediction"} · ${file.name}`,
          confirmed_token: confirmed,
        });
        if (confirmed) appendToken(confirmed, kind);
      } else {
        updatePredictionUI({
          label: "",
          confidence: 0,
          topk: [],
          latency_ms: 0,
          motion_energy: 0,
          hold_progress: 0,
          status: data.message || "Could not classify upload.",
          confirmed_token: null,
        });
      }
    } catch (err) {
      els.stagePill.textContent = "ERROR";
      els.lblStatus.textContent = err.message;
    }
  }

  function renderDemoGrid(filterText = "") {
    const isWords = state.demoSubTab === "words";
    els.demoSubWords.classList.toggle("active", isWords);
    els.demoSubLetters.classList.toggle("active", !isWords);
    els.demoSubWords.setAttribute("aria-selected", String(isWords));
    els.demoSubLetters.setAttribute("aria-selected", String(!isWords));
    els.btnTryWords.textContent = isWords ? "Try Live in Words" : "Try Live in Letters";
    els.demoLetterChartCard.classList.toggle("hidden", isWords || !els.demoLetterChartImg.getAttribute("src"));

    els.demoGrid.innerHTML = "";
    const q = filterText.trim().toLowerCase();

    if (isWords) {
      if (!state.demoSigns) return;
      const items = q
        ? state.demoSigns.filter(
            (item) => item.word.toLowerCase().includes(q) || item.how_to.toLowerCase().includes(q)
          )
        : state.demoSigns;

      for (const item of items) {
        const card = document.createElement("article");
        card.className = "demo-card";

        const media = document.createElement("div");
        media.className = "demo-media";
        if (item.image_url) {
          const img = document.createElement("img");
          img.src = item.image_url;
          img.alt = `ASL sign for ${item.word}`;
          img.loading = "lazy";
          media.appendChild(img);
        }

        const body = document.createElement("div");
        body.className = "demo-body";

        const top = document.createElement("div");
        top.className = "demo-card-top";

        const wordEl = document.createElement("div");
        wordEl.className = "demo-word";
        wordEl.textContent = item.word;

        const tryBtn = document.createElement("button");
        tryBtn.type = "button";
        tryBtn.className = "btn btn-outline demo-try-btn";
        tryBtn.textContent = "Practice";
        tryBtn.addEventListener("click", () => switchTab("words", item.word));

        top.appendChild(wordEl);
        top.appendChild(tryBtn);

        const desc = document.createElement("p");
        desc.className = "demo-howto";
        desc.textContent = item.how_to;

        body.appendChild(top);
        body.appendChild(desc);
        card.appendChild(media);
        card.appendChild(body);
        els.demoGrid.appendChild(card);
      }
      return;
    }

    if (!state.demoLetters) return;
    const items = q
      ? state.demoLetters.filter(
          (item) => item.letter.toLowerCase().includes(q) || item.how_to.toLowerCase().includes(q)
        )
      : state.demoLetters;

    for (const item of items) {
      const card = document.createElement("article");
      card.className = "demo-card";

      const badge = document.createElement("div");
      badge.className = "letter-demo-badge";
      const chEl = document.createElement("div");
      chEl.className = "letter-demo-char";
      chEl.textContent = item.letter;
      badge.appendChild(chEl);
      if (item.dynamic) {
        const tag = document.createElement("span");
        tag.className = "letter-motion-tag";
        tag.textContent = "Motion";
        badge.appendChild(tag);
      }

      const body = document.createElement("div");
      body.className = "demo-body";
      const top = document.createElement("div");
      top.className = "demo-card-top";
      const lbl = document.createElement("div");
      lbl.className = "demo-word";
      lbl.textContent = `Letter ${item.letter}`;

      const tryBtn = document.createElement("button");
      tryBtn.type = "button";
      tryBtn.className = "btn btn-outline demo-try-btn";
      tryBtn.textContent = "Practice";
      tryBtn.addEventListener("click", () => switchTab("letters", item.letter));
      top.appendChild(lbl);
      top.appendChild(tryBtn);

      const desc = document.createElement("p");
      desc.className = "demo-howto";
      desc.textContent = item.how_to;

      body.appendChild(top);
      body.appendChild(desc);
      card.appendChild(badge);
      card.appendChild(body);
      els.demoGrid.appendChild(card);
    }
  }

  async function ensureDemoSignsLoaded() {
    if (state.demoSigns && state.demoLetters) return;
    const resp = await fetch("/api/demo-signs");
    if (!resp.ok) throw new Error("Failed to load demo signs");
    const data = await resp.json();
    state.demoSigns = data.signs || [];
    state.demoLetters = data.letters || [];
    if (data.letter_chart_url) {
      els.demoLetterChartImg.src = data.letter_chart_url;
    } else {
      els.demoLetterChartImg.removeAttribute("src");
    }
    renderDemoGrid(els.demoFilter.value);
  }

  function syncCalibrationControls() {
    if (!state.status) return;
    const labels = state.mode === "words" ? state.status.words.labels : state.status.letters.labels;
    const prev = els.calLabelSelect.value;
    els.calLabelSelect.innerHTML = "";
    for (const lbl of labels) {
      const opt = document.createElement("option");
      opt.value = lbl;
      opt.textContent = lbl.toUpperCase();
      els.calLabelSelect.appendChild(opt);
    }
    if (labels.includes(prev)) els.calLabelSelect.value = prev;
    els.btnCalRecord.textContent = state.mode === "words" ? "Record 2s Clip" : "Capture Snapshot";
    els.calTrimRow.classList.toggle("hidden", state.mode !== "words");
    updateCalCountBadge();
  }

  function getActiveCalLabel() {
    const custom = (els.calCustomLabel.value || "").trim();
    return custom || els.calLabelSelect.value;
  }

  function updateCalCountBadge() {
    const lbl = getActiveCalLabel();
    const total = Number(state.calCounts?.annotated_total || 0);
    if (!state.calCounts || !lbl) {
      els.calCountBadge.textContent = `${total} total`;
      return;
    }
    const bucket = state.mode === "words" ? state.calCounts.words : state.calCounts.letters;
    const norm = state.mode === "words" ? lbl.toLowerCase() : lbl.toUpperCase();
    const cnt = Number(bucket?.[norm] || 0);
    els.calCountBadge.textContent = `${cnt} label · ${total} total`;
  }

  function renderRecentAnnotations(rows) {
    els.calRecentList.innerHTML = "";
    if (!Array.isArray(rows) || !rows.length) {
      return;
    }
    for (const row of rows.slice(0, 8)) {
      const item = document.createElement("div");
      item.className = "cal-recent-item";
      const main = document.createElement("span");
      main.className = "cal-recent-main";
      main.innerHTML = `<strong>${row.label}</strong> (${row.mode || "clip"}) · ${row.signer_id} · ${row.lighting}/${row.background}`;
      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "cal-del-btn";
      delBtn.textContent = "Delete";
      delBtn.addEventListener("click", async () => {
        delBtn.disabled = true;
        try {
          const resp = await fetch(`/api/calibration/sample/${encodeURIComponent(row.sample_id)}`, { method: "DELETE" });
          const data = await resp.json();
          if (resp.ok) {
            state.calCounts = data.counts;
            updateCalCountBadge();
            renderRecentAnnotations(data.annotations || []);
            els.calStatusMsg.textContent = `Deleted sample ${row.label} (${row.sample_id}).`;
          }
        } catch (_) {}
      });
      item.appendChild(main);
      item.appendChild(delBtn);
      els.calRecentList.appendChild(item);
    }
  }

  async function fetchCalibrationCounts() {
    try {
      const resp = await fetch("/api/calibration/annotations?limit=25");
      if (!resp.ok) return;
      const data = await resp.json();
      state.calCounts = data.counts;
      updateCalCountBadge();
      renderRecentAnnotations(data.annotations || []);
    } catch (_) {}
  }

  function grabUnmirroredFrameBlob() {
    return new Promise((resolve) => {
      const v = els.liveVideo;
      if (!state.streaming || v.readyState < 2 || v.videoWidth <= 0) {
        resolve(null);
        return;
      }
      const w = 640;
      const h = Math.max(240, Math.round((v.videoHeight / v.videoWidth) * w));
      if (state.captureCanvas.width !== w || state.captureCanvas.height !== h) {
        state.captureCanvas.width = w;
        state.captureCanvas.height = h;
      }
      const ctx = state.captureCanvas.getContext("2d");
      ctx.drawImage(v, 0, 0, w, h);
      state.captureCanvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.90);
    });
  }

  async function runBrowserCalibrationCapture() {
    if (state.calBusy) return;
    if (!state.streaming) {
      els.calStatusMsg.textContent = "Start your webcam first before recording calibration samples.";
      return;
    }
    const customText = (els.calCustomLabel.value || "").trim();
    const label = customText || els.calLabelSelect.value;
    const allowCustom = Boolean(customText);
    const signerId = (els.calSignerId.value || "anonymous").trim() || "anonymous";
    const lighting = els.calLighting.value || "normal";
    const background = els.calBackground.value || "plain";
    const handedness = els.calHandedness.value || "right";
    if (!label) return;
    try { localStorage.setItem("asl_signer_id", signerId); } catch (_) {}
    state.calBusy = true;
    els.btnCalRecord.disabled = true;

    try {
      if (state.mode === "letters") {
        els.calStatusMsg.textContent = `Hold "${label}" steady… capturing snapshot.`;
        const blob = await grabUnmirroredFrameBlob();
        if (!blob) throw new Error("Failed to grab webcam frame.");
        const form = new FormData();
        form.append("label", label);
        form.append("file", blob, "snapshot.jpg");
        form.append("signer_id", signerId);
        form.append("lighting", lighting);
        form.append("background", background);
        form.append("handedness", handedness);
        form.append("allow_custom", String(allowCustom));
        const resp = await fetch("/api/calibration/letter", { method: "POST", body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Save failed");
        state.calCounts = data.counts;
        updateCalCountBadge();
        renderRecentAnnotations(data.annotations || []);
        els.calStatusMsg.textContent = `Saved letter ${data.label} (${data.path}) by ${signerId}.`;
      } else {
        for (const sec of ["1.0s", "0.5s"]) {
          els.calStatusMsg.textContent = `Get ready for "${label.toUpperCase()}" (${sec})…`;
          await new Promise((r) => setTimeout(r, 450));
        }
        els.calStatusMsg.textContent = `Recording "${label.toUpperCase()}" (2s)…`;
        const blobs = [];
        for (let i = 0; i < 28; i++) {
          const b = await grabUnmirroredFrameBlob();
          if (b) blobs.push(b);
          await new Promise((r) => setTimeout(r, 67));
        }
        const trimStart = Number(els.rngTrimStart.value || 0) / 100.0;
        const trimEnd = Number(els.rngTrimEnd.value || 100) / 100.0;
        const form = new FormData();
        form.append("label", label);
        form.append("fps", "15");
        form.append("signer_id", signerId);
        form.append("lighting", lighting);
        form.append("background", background);
        form.append("handedness", handedness);
        form.append("trim_start_pct", String(trimStart));
        form.append("trim_end_pct", String(trimEnd));
        form.append("allow_custom", String(allowCustom));
        blobs.forEach((b, i) => form.append("files", b, `f_${i}.jpg`));
        const resp = await fetch("/api/calibration/word", { method: "POST", body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || "Save failed");
        state.calCounts = data.counts;
        updateCalCountBadge();
        renderRecentAnnotations(data.annotations || []);
        els.calStatusMsg.textContent = `Saved ${data.frames_saved}f clip for "${data.label.toUpperCase()}" by ${signerId}.`;
      }
    } catch (err) {
      els.calStatusMsg.textContent = err.message || "Calibration failed.";
    } finally {
      state.calBusy = false;
      els.btnCalRecord.disabled = false;
    }
  }

  function applyTuningControls(send = true) {
    state.tuning.word_conf = Number(els.rngWordConf.value);
    state.tuning.word_motion = Number(els.rngWordMotion.value);
    state.tuning.letter_conf = Number(els.rngLetterConf.value);
    state.tuning.letter_hold = Number(els.rngLetterHold.value);
    els.valWordConf.textContent = `${Math.round(state.tuning.word_conf * 100)}%`;
    els.valWordMotion.textContent = state.tuning.word_motion.toFixed(3);
    els.valLetterConf.textContent = `${Math.round(state.tuning.letter_conf * 100)}%`;
    els.valLetterHold.textContent = `${state.tuning.letter_hold.toFixed(2)}s`;
    if (state.mode === "words") {
      els.meterMarker.style.left = `${Math.min(98, (state.tuning.word_motion / 0.06) * 100).toFixed(1)}%`;
    }
    if (send) sendWsControl({ type: "config", ...state.tuning });
  }

  function switchTab(target, practiceLabel = "") {
    if (target === "demo") {
      if (state.streaming) stopCamera();
      state.activeView = "demo";
      renderModeShell();
      ensureDemoSignsLoaded().catch((err) => {
        els.demoGrid.textContent = err.message;
      });
      return;
    }

    const prevMode = state.mode;
    state.activeView = target;
    state.mode = target;
    state.manualRecording = false;
    if (prevMode !== target) {
      sendWsControl({ type: "config", mode: target, return_overlay: els.chkSkeleton.checked, ...state.tuning });
    }
    renderModeShell();
    syncSegmenterButtons();

    if (practiceLabel) {
      els.lblStatus.textContent = `Ready to practice "${practiceLabel.toUpperCase()}" · Start camera or upload a sample`;
      for (const chip of els.vocabGrid.children) {
        chip.classList.toggle("preview", chip.dataset.label === practiceLabel.toLowerCase());
      }
    }
  }

  function switchMode(newMode) {
    switchTab(newMode);
  }

  // Event wiring
  els.tabWords.addEventListener("click", () => switchTab("words"));
  els.tabLetters.addEventListener("click", () => switchTab("letters"));
  els.tabDemo.addEventListener("click", () => switchTab("demo"));
  els.demoSubWords.addEventListener("click", () => {
    state.demoSubTab = "words";
    renderDemoGrid(els.demoFilter.value);
  });
  els.demoSubLetters.addEventListener("click", () => {
    state.demoSubTab = "letters";
    renderDemoGrid(els.demoFilter.value);
  });
  els.btnTryWords.addEventListener("click", () => switchTab(state.demoSubTab === "letters" ? "letters" : "words"));
  els.demoFilter.addEventListener("input", () => renderDemoGrid(els.demoFilter.value));

  els.btnTune.addEventListener("click", () => {
    els.tunePanel.classList.toggle("hidden");
  });
  els.btnCalibrate.addEventListener("click", () => {
    const nowOpen = els.calibratePanel.classList.toggle("hidden") === false;
    if (nowOpen) fetchCalibrationCounts();
  });
  els.calLabelSelect.addEventListener("change", updateCalCountBadge);
  els.calCustomLabel.addEventListener("input", updateCalCountBadge);
  els.rngTrimStart.addEventListener("input", () => {
    if (Number(els.rngTrimStart.value) >= Number(els.rngTrimEnd.value) - 10) {
      els.rngTrimStart.value = String(Math.max(0, Number(els.rngTrimEnd.value) - 15));
    }
    els.valTrimStart.textContent = `${els.rngTrimStart.value}%`;
  });
  els.rngTrimEnd.addEventListener("input", () => {
    if (Number(els.rngTrimEnd.value) <= Number(els.rngTrimStart.value) + 10) {
      els.rngTrimEnd.value = String(Math.min(100, Number(els.rngTrimStart.value) + 15));
    }
    els.valTrimEnd.textContent = `${els.rngTrimEnd.value}%`;
  });
  els.btnCalRecord.addEventListener("click", runBrowserCalibrationCapture);

  els.btnOnboardingDemo.addEventListener("click", () => switchTab("demo"));
  els.btnDismissTip.addEventListener("click", () => {
    els.onboardingStrip.classList.add("hidden");
    try { localStorage.setItem("asl_hide_onboarding", "1"); } catch (_) {}
  });
  els.packSelect.addEventListener("change", async () => {
    try {
      const resp = await fetch("/api/packs/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pack_id: els.packSelect.value }),
      });
      if (resp.ok) applyStatusPayload(await resp.json());
    } catch (_) {}
  });
  els.badgeDeploy.addEventListener("click", async () => {
    const curInt8 = Boolean(state.status?.deployment?.prefer_int8);
    try {
      const resp = await fetch("/api/precision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prefer_int8: !curInt8 }),
      });
      if (resp.ok) applyStatusPayload(await resp.json());
    } catch (_) {}
  });

  [els.rngWordConf, els.rngWordMotion, els.rngLetterConf, els.rngLetterHold].forEach((rng) => {
    rng.addEventListener("input", () => applyTuningControls(true));
  });
  els.btnTuneReset.addEventListener("click", () => {
    els.rngWordConf.value = "0.68";
    els.rngWordMotion.value = "0.014";
    els.rngLetterConf.value = "0.72";
    els.rngLetterHold.value = "0.65";
    applyTuningControls(true);
  });

  els.cameraSelect.addEventListener("change", () => {
    state.selectedDeviceId = els.cameraSelect.value;
    if (state.streaming) {
      stopCamera();
      startCamera(state.selectedDeviceId);
    }
  });

  els.btnReload.addEventListener("click", async () => {
    els.btnReload.disabled = true;
    try {
      await fetchStatus("/api/reload", "POST");
    } finally {
      els.btnReload.disabled = false;
    }
  });

  els.btnCamera.addEventListener("click", () => (state.streaming ? stopCamera() : startCamera()));
  els.btnUpload.addEventListener("click", () => {
    els.fileInput.value = "";
    els.fileInput.click();
  });
  els.fileInput.addEventListener("change", (e) => handleFileUpload(e.target.files?.[0]));

  els.stageBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.stageBox.classList.add("dragover");
  });
  els.stageBox.addEventListener("dragleave", () => els.stageBox.classList.remove("dragover"));
  els.stageBox.addEventListener("drop", (e) => {
    e.preventDefault();
    els.stageBox.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileUpload(file);
  });

  els.btnAutoMode.addEventListener("click", () => {
    state.autoWordMode = !state.autoWordMode;
    state.manualRecording = false;
    syncSegmenterButtons();
    sendWsControl({ type: "config", auto_word_mode: state.autoWordMode });
  });

  els.btnManualCapture.addEventListener("click", () => {
    sendWsControl({ type: "manual_capture" });
  });

  els.chkMirror.addEventListener("change", () => {
    els.liveVideo.classList.toggle("mirrored", els.chkMirror.checked);
  });

  els.liveVideo.addEventListener("loadedmetadata", () => {
    if (els.liveVideo.videoWidth > 0 && els.liveVideo.videoHeight > 0) {
      els.stageBox.style.aspectRatio = `${els.liveVideo.videoWidth} / ${els.liveVideo.videoHeight}`;
    }
  });

  els.chkSkeleton.addEventListener("change", () => {
    if (!els.chkSkeleton.checked) clearStageOverlay();
    sendWsControl({ type: "config", return_overlay: els.chkSkeleton.checked });
  });

  els.chkAutoAppend.addEventListener("change", () => {
    state.autoAppend = els.chkAutoAppend.checked;
  });

  els.chkPolish.addEventListener("change", () => {
    state.polishEnglish = els.chkPolish.checked;
    renderTranscript();
  });

  els.btnSpace.addEventListener("click", () => {
    state.joinLetters = false;
    state.pendingSpace = state.tokens.length > 0;
    renderTranscript();
  });

  els.btnUndo.addEventListener("click", () => {
    state.pendingSpace = false;
    if (!state.tokens.length) return;
    const last = state.tokens[state.tokens.length - 1];
    if (last.kind === "letter" && last.text.length > 1) {
      last.text = last.text.slice(0, -1);
      state.joinLetters = true;
    } else {
      state.tokens.pop();
      state.joinLetters = Boolean(state.tokens.length && state.tokens[state.tokens.length - 1].kind === "letter");
    }
    renderTranscript();
  });

  els.btnClear.addEventListener("click", () => {
    state.tokens = [];
    state.joinLetters = false;
    state.pendingSpace = false;
    renderTranscript();
  });

  els.btnCopy.addEventListener("click", async () => {
    const text = getActiveTranscriptText();
    if (!text) return;
    await navigator.clipboard?.writeText(text);
    const orig = els.btnCopy.textContent;
    els.btnCopy.textContent = "Copied";
    setTimeout(() => (els.btnCopy.textContent = orig), 1200);
  });

  els.btnSpeak.addEventListener("click", () => {
    const text = getActiveTranscriptText();
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  });

  // Init
  try {
    if (localStorage.getItem("asl_hide_onboarding") === "1") {
      els.onboardingStrip.classList.add("hidden");
    }
    const savedSigner = localStorage.getItem("asl_signer_id");
    if (savedSigner) els.calSignerId.value = savedSigner;
  } catch (_) {}
  applyTuningControls(false);
  fetchStatus().catch((err) => {
    els.lblStatus.textContent = err.message;
  });
})();
