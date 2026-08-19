(function () {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const openButton = byId("scanPyramidButton");
  const dialog = byId("ppaiScanDialog");
  const video = byId("ppaiScanVideo");
  const captureButton = byId("ppaiCaptureButton");
  const stopButton = byId("ppaiStopCameraButton");
  const closeButton = byId("ppaiCloseScanButton");
  const retakeButton = byId("ppaiRetakeButton");
  const keepButton = byId("ppaiKeepCaptureButton");
  const detectButton = byId("ppaiDetectGeometryButton");
  const previewPanel = byId("ppaiPreviewPanel");
  const cameraPanel = byId("ppaiCameraPanel");
  const canvas = byId("ppaiCaptureCanvas");
  const cvProof = byId("ppaiCvProof");
  const geometrySummary = byId("ppaiGeometrySummary");
  const cvStatus = byId("opencvStatus");
  const scanStatus = byId("ppaiScanStatus");

  let cameraStream = null;
  let cvReady = false;
  let captured = false;
  let lastGeometry = null;
  let originalCapturedImageData = null;

  function setCvStatus(text, kind) {
    if (!cvStatus) return;
    cvStatus.textContent = text;
    cvStatus.className = "opencv-status" + (kind ? " " + kind : "");
  }

  function setScanStatus(text, kind) {
    if (!scanStatus) return;
    scanStatus.textContent = text;
    scanStatus.className = "scan-status" + (kind ? " " + kind : "");
  }

  function setGeometrySummary(html, kind) {
    if (!geometrySummary) return;
    geometrySummary.innerHTML = html;
    geometrySummary.className = "geometry-summary" + (kind ? " " + kind : "");
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
    }
    if (video) {
      try { video.pause(); } catch (_) {}
      video.srcObject = null;
    }
    if (captureButton) captureButton.disabled = true;
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setScanStatus("This browser does not provide an in-page camera.", "error");
      return;
    }

    stopCamera();
    captured = false;
    lastGeometry = null;
    originalCapturedImageData = null;
    keepButton.disabled = true;
    if (previewPanel) previewPanel.hidden = true;
    if (cameraPanel) cameraPanel.hidden = false;
    setScanStatus("Requesting rear-camera permission…", "working");

    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });
      video.srcObject = cameraStream;
      await video.play();
      captureButton.disabled = false;
      setScanStatus(
        "Keep the full 28-tile pyramid visible. You do not need exact manual alignment.",
        "ready"
      );
    } catch (error) {
      console.error(error);
      stopCamera();
      const message = error && error.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access for this site in Safari settings."
        : `Camera could not start: ${error.message || error}`;
      setScanStatus(message, "error");
    }
  }

  function restoreOriginalCapture() {
    if (!originalCapturedImageData) return;
    canvas.width = originalCapturedImageData.width;
    canvas.height = originalCapturedImageData.height;
    canvas.getContext("2d").putImageData(originalCapturedImageData, 0, 0);
  }

  function runOpenCvProof() {
    if (!cvReady || !window.cv) {
      cvProof.textContent = "Photo captured. OpenCV is not ready, so the proof operation was skipped.";
      return;
    }

    let src = null;
    let gray = null;
    let edges = null;
    try {
      src = window.cv.imread(canvas);
      gray = new window.cv.Mat();
      edges = new window.cv.Mat();
      window.cv.cvtColor(src, gray, window.cv.COLOR_RGBA2GRAY);
      window.cv.Canny(gray, edges, 70, 150);

      let edgePixels = 0;
      const data = edges.data;
      for (let i = 0; i < data.length; i += 1) {
        if (data[i]) edgePixels += 1;
      }

      cvProof.textContent =
        `OpenCV proof passed: ${src.cols}×${src.rows} frame; ${edgePixels.toLocaleString()} edge pixels detected.`;
    } catch (error) {
      console.error(error);
      cvProof.textContent = `OpenCV proof operation failed: ${error.message || error}`;
    } finally {
      if (edges) edges.delete();
      if (gray) gray.delete();
      if (src) src.delete();
    }
  }

  function captureFrame() {
    if (!video || !video.videoWidth || !video.videoHeight) {
      setScanStatus("The camera preview is not ready yet. Wait a moment and try again.", "error");
      return;
    }

    const maxWidth = 1400;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));

    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    originalCapturedImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    captured = true;
    lastGeometry = null;
    keepButton.disabled = true;
    stopCamera();
    cameraPanel.hidden = true;
    previewPanel.hidden = false;
    setGeometrySummary(
      "Photo captured. Press <strong>Detect 28 Tile Centers</strong> to fit the pyramid geometry.",
      ""
    );
    runOpenCvProof();

    try {
      window.ppaiLastPyramidCapture = canvas.toDataURL("image/jpeg", 0.92);
      sessionStorage.setItem("ppaiLastPyramidCapture", window.ppaiLastPyramidCapture);
    } catch (error) {
      console.warn("Could not cache captured image.", error);
    }
  }

  function makeIntegralFromBinary(mat) {
    const width = mat.cols, height = mat.rows, stride = width + 1;
    const integral = new Uint32Array((width + 1) * (height + 1));
    for (let y = 0; y < height; y += 1) {
      const row = mat.ucharPtr(y);
      let running = 0;
      const out = (y + 1) * stride;
      const prev = y * stride;
      for (let x = 0; x < width; x += 1) {
        running += row[x] ? 1 : 0;
        integral[out + x + 1] = integral[prev + x + 1] + running;
      }
    }
    return { data: integral, width, height, stride };
  }

  function rectSum(ii, x0, y0, x1, y1) {
    x0 = Math.max(0, Math.min(ii.width, Math.round(x0)));
    x1 = Math.max(0, Math.min(ii.width, Math.round(x1)));
    y0 = Math.max(0, Math.min(ii.height, Math.round(y0)));
    y1 = Math.max(0, Math.min(ii.height, Math.round(y1)));
    if (x1 <= x0 || y1 <= y0) return 0;
    const a = ii.data[y0 * ii.stride + x0];
    const b = ii.data[y0 * ii.stride + x1];
    const c = ii.data[y1 * ii.stride + x0];
    const d = ii.data[y1 * ii.stride + x1];
    return d - b - c + a;
  }

  function rectRatio(ii, x0, y0, x1, y1) {
    const w = Math.max(1, Math.round(x1) - Math.round(x0));
    const h = Math.max(1, Math.round(y1) - Math.round(y0));
    return rectSum(ii, x0, y0, x1, y1) / (w * h);
  }

  function pyramidTemplateGeometry(centerX, bottomY, pitch, vFactor, topScale) {
    const centers = [];
    let tileId = 1;
    for (let row = 1; row <= 7; row += 1) {
      const t = (row - 1) / 6;
      const rowPitch = pitch * (topScale + (1 - topScale) * t);
      const y = bottomY - (7 - row) * pitch * vFactor;
      for (let col = 1; col <= row; col += 1) {
        centers.push({
          tileId, row, col,
          x: centerX + (col - (row + 1) / 2) * rowPitch,
          y,
          r: pitch * 0.39,
          inferred: false
        });
        tileId += 1;
      }
    }
    return { centerX, bottomY, pitch, vFactor, topScale, centers };
  }

  function edgePatch(ii, x, y, half) {
    return rectRatio(ii, x - half, y - half, x + half + 1, y + half + 1);
  }

  function scoreTemplateCenter(ii, c, pitch) {
    const r = pitch * 0.39;
    const patch = Math.max(1.5, pitch * 0.055);
    const angles = [0,Math.PI/4,Math.PI/2,3*Math.PI/4,Math.PI,5*Math.PI/4,3*Math.PI/2,7*Math.PI/4];

    let outer = 0, inner = 0;
    for (const a of angles) {
      outer += edgePatch(ii, c.x + Math.cos(a)*r, c.y + Math.sin(a)*r, patch);
      inner += edgePatch(ii, c.x + Math.cos(a)*r*0.78, c.y + Math.sin(a)*r*0.78, patch);
    }
    outer /= angles.length;
    inner /= angles.length;

    const centerBusy = rectRatio(ii, c.x-r*0.5, c.y-r*0.5, c.x+r*0.5, c.y+r*0.5);
    const score = outer*0.58 + inner*0.30 + centerBusy*0.12;
    return { score, outer, inner, centerBusy };
  }

  function scorePyramidTemplate(ii, g) {
    const supports = g.centers.map(c => ({...c, ...scoreTemplateCenter(ii,c,g.pitch)}));
    const mean = supports.reduce((s,c)=>s+c.score,0)/28;
    const bottom = supports.filter(c=>c.row===7);
    const bottomMean = bottom.reduce((s,c)=>s+c.score,0)/7;
    const apex = supports[0].score;
    const sorted = supports.map(c=>c.score).sort((a,b)=>a-b);
    const lowQuartile = sorted.slice(0,7).reduce((a,b)=>a+b,0)/7;
    return {
      score: mean*5.5 + bottomMean*2.2 + apex*0.9 + lowQuartile*2.4,
      mean, bottomMean, apex, lowQuartile, supports
    };
  }

  function fitFixedPyramidTemplate(edgeMask) {
    const ii = makeIntegralFromBinary(edgeMask);
    const w = edgeMask.cols, h = edgeMask.rows;
    let best = null;
    const coarse = [];

    function consider(centerX,bottomY,pitch,vFactor,topScale) {
      const g = pyramidTemplateGeometry(centerX,bottomY,pitch,vFactor,topScale);
      const left = Math.min(...g.centers.map(c=>c.x-c.r));
      const right = Math.max(...g.centers.map(c=>c.x+c.r));
      const top = Math.min(...g.centers.map(c=>c.y-c.r));
      const bottom = Math.max(...g.centers.map(c=>c.y+c.r));
      if (left<0 || top<0 || right>=w || bottom>=h) return;
      const metrics = scorePyramidTemplate(ii,g);
      const c = {geometry:g,metrics};
      coarse.push(c);
      if (!best || metrics.score > best.metrics.score) best = c;
    }

    const minPitch = Math.round(w*0.075), maxPitch = Math.round(w*0.145);
    const pitchStep = Math.max(4,Math.round(w*0.007));
    const centerStep = Math.max(5,Math.round(w*0.010));
    const bottomStep = Math.max(6,Math.round(h*0.010));

    for (let pitch=minPitch; pitch<=maxPitch; pitch+=pitchStep) {
      for (let cx=Math.round(w*0.34); cx<=Math.round(w*0.66); cx+=centerStep) {
        for (let by=Math.round(h*0.38); by<=Math.round(h*0.72); by+=bottomStep) {
          for (const vf of [0.76,0.80,0.84,0.88,0.92,0.96]) {
            for (const ts of [0.92,0.96,1.00]) consider(cx,by,pitch,vf,ts);
          }
        }
      }
    }
    if (!best) throw new Error("No fixed pyramid-template candidate fit inside the image.");

    const b = best.geometry;
    let fineBest = best;
    for (let pitch=b.pitch-6; pitch<=b.pitch+6; pitch+=2) {
      for (let cx=b.centerX-10; cx<=b.centerX+10; cx+=2) {
        for (let by=b.bottomY-12; by<=b.bottomY+12; by+=2) {
          for (let vf=b.vFactor-0.04; vf<=b.vFactor+0.04; vf+=0.01) {
            for (let ts=b.topScale-0.03; ts<=b.topScale+0.03; ts+=0.01) {
              const g = pyramidTemplateGeometry(cx,by,pitch,vf,ts);
              const left = Math.min(...g.centers.map(c=>c.x-c.r));
              const right = Math.max(...g.centers.map(c=>c.x+c.r));
              const top = Math.min(...g.centers.map(c=>c.y-c.r));
              const bottom = Math.max(...g.centers.map(c=>c.y+c.r));
              if (left<0 || top<0 || right>=w || bottom>=h) continue;
              const metrics = scorePyramidTemplate(ii,g);
              if (metrics.score > fineBest.metrics.score) fineBest = {geometry:g,metrics};
            }
          }
        }
      }
    }

    const alternatives = coarse.filter(c=>{
      const a=c.geometry,g=fineBest.geometry;
      return Math.abs(a.centerX-g.centerX)>g.pitch*0.55 ||
             Math.abs(a.bottomY-g.bottomY)>g.pitch*0.55 ||
             Math.abs(a.pitch-g.pitch)>g.pitch*0.15;
    }).sort((a,b)=>b.metrics.score-a.metrics.score);

    const second = alternatives[0] || null;
    const margin = second ? fineBest.metrics.score-second.metrics.score : fineBest.metrics.score;
    const normalizedMargin = margin/Math.max(0.0001,fineBest.metrics.score);
    const supportThreshold = Math.max(0.018,fineBest.metrics.mean*0.42);

    const centers = fineBest.metrics.supports.map(s=>({
      tileId:s.tileId,row:s.row,col:s.col,x:s.x,y:s.y,r:s.r,
      inferred:s.score<supportThreshold,
      templateSupport:s.score
    }));
    const supportedCount = centers.filter(c=>!c.inferred).length;
    const bottomSupported = centers.filter(c=>c.row===7&&!c.inferred).length;
    const apexSupported = !centers[0].inferred;

    const rowDiagnostics = [];
    for (let row=1; row<=7; row+=1) {
      const rr=centers.filter(c=>c.row===row);
      rowDiagnostics.push({row,supported:rr.filter(c=>!c.inferred).length,expected:row});
    }

    let quality="bad", locked=false;
    if (supportedCount>=24 && bottomSupported>=6 && apexSupported && normalizedMargin>=0.02) {
      quality="good"; locked=true;
    } else if (supportedCount>=21 && bottomSupported>=5 && apexSupported) {
      quality="warn";
    }

    return {
      ok:true,locked,quality,centers,supportedCount,bottomSupported,apexSupported,rowDiagnostics,
      detector:"fixed-28-position-pyramid-template",
      templateScore:fineBest.metrics.score,
      secondScore:second?second.metrics.score:null,
      normalizedMargin,
      workGeometry:fineBest.geometry
    };
  }

  function detectCirclesWithOpenCv() {
    if (!cvReady || !window.cv) throw new Error("OpenCV is not ready.");
    restoreOriginalCapture();

    let src=null, small=null, gray=null, blurred=null, edges=null;
    try {
      const cv=window.cv;
      src=cv.imread(canvas);
      const targetWidth=Math.min(640,src.cols);
      const scale=targetWidth/src.cols;
      const targetHeight=Math.max(1,Math.round(src.rows*scale));

      small=new cv.Mat();
      cv.resize(src,small,new cv.Size(targetWidth,targetHeight),0,0,cv.INTER_AREA);

      gray=new cv.Mat();
      cv.cvtColor(small,gray,cv.COLOR_RGBA2GRAY);

      blurred=new cv.Mat();
      cv.GaussianBlur(gray,blurred,new cv.Size(5,5),0);

      edges=new cv.Mat();
      cv.Canny(blurred,edges,42,126);

      const result=fitFixedPyramidTemplate(edges);
      result.scaleToCanvasX=canvas.width/targetWidth;
      result.scaleToCanvasY=canvas.height/targetHeight;
      result.workWidth=targetWidth;
      result.workHeight=targetHeight;
      return result;
    } finally {
      if(edges)edges.delete(); if(blurred)blurred.delete(); if(gray)gray.delete(); if(small)small.delete(); if(src)src.delete();
    }
  }

  function drawGeometryOverlay(geometry) {
    restoreOriginalCapture();
    const ctx = canvas.getContext("2d");
    const sx = geometry.scaleToCanvasX || 1;
    const sy = geometry.scaleToCanvasY || 1;

    ctx.save();
    ctx.lineWidth = Math.max(2, canvas.width / 420);
    ctx.font = `bold ${Math.max(13, Math.round(canvas.width / 45))}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (const c of geometry.centers) {
      const x = c.x * sx;
      const y = c.y * sy;
      const r = Math.max(12, c.r * (sx + sy) / 2 * 0.72);

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.strokeStyle = c.inferred ? "#ffd400" : "#24e36a";
      ctx.stroke();

      const labelR = Math.max(10, canvas.width / 55);
      ctx.beginPath();
      ctx.arc(x, y, labelR, 0, Math.PI * 2);
      ctx.fillStyle = c.inferred ? "rgba(120,85,0,.88)" : "rgba(0,70,28,.88)";
      ctx.fill();

      ctx.fillStyle = "#fff";
      ctx.fillText(String(c.tileId), x, y + 0.5);
    }
    ctx.restore();
  }

  function geometryHtml(g) {
    if (!g.ok) return `<strong>Pyramid-template fit failed.</strong><br>${g.reason || "No template fit."}`;
    const chips=g.rowDiagnostics.map(r=>`<div class="geometry-chip">Row ${r.row}: ${r.supported}/${r.expected}</div>`).join("");
    const marginPct=Math.round((g.normalizedMargin||0)*1000)/10;
    if (g.locked) {
      return `<strong>28-position pyramid template LOCKED.</strong><br>
        ${g.supportedCount}/28 strongly supported; bottom ${g.bottomSupported}/7; apex supported.<br>
        Best-vs-second template separation: ${marginPct}%.
        <div class="geometry-grid">${chips}</div>`;
    }
    if (g.quality==="warn") {
      return `<strong>Template fit found — CHECK ALIGNMENT.</strong><br>
        ${g.supportedCount}/28 strongly supported; bottom ${g.bottomSupported}/7; apex ${g.apexSupported?"supported":"weak"}.<br>
        The 28 legal positions come from the fixed template, but confidence is not high enough to continue automatically.
        <div class="geometry-grid">${chips}</div>`;
    }
    return `<strong>Template fit uncertain — RETAKE / RETRY.</strong><br>
      ${g.supportedCount}/28 strongly supported; bottom ${g.bottomSupported}/7; apex ${g.apexSupported?"supported":"weak"}.
      <div class="geometry-grid">${chips}</div>`;
  }

  function detectGeometry() {
    if (!captured) {
      setGeometrySummary("Capture a pyramid photo first.", "bad");
      return;
    }
    if (!cvReady) {
      setGeometrySummary("OpenCV is not ready yet.", "bad");
      return;
    }

    detectButton.disabled = true;
    keepButton.disabled = true;
    setGeometrySummary("Fitting the canonical 28-position pyramid template over the photographed board…", "");

    // Yield once so Safari paints the status before the heavier OpenCV work.
    window.setTimeout(() => {
      try {
        const geometry = detectCirclesWithOpenCv();
        lastGeometry = geometry;

        if (geometry.ok) {
          drawGeometryOverlay(geometry);
          setGeometrySummary(geometryHtml(geometry), geometry.locked ? "good" : (geometry.quality || "warn"));
          keepButton.disabled = !geometry.locked;

          if (geometry.locked) {
            try {
              const normalized = geometry.centers.map((c) => ({
                tileId: c.tileId,
                row: c.row,
                col: c.col,
                x: c.x / geometry.workWidth,
                y: c.y / geometry.workHeight,
                r: c.r / geometry.workWidth,
                inferred: !!c.inferred
              }));
              window.ppaiLastPyramidGeometry = normalized;
              sessionStorage.setItem("ppaiLastPyramidGeometry", JSON.stringify(normalized));
            } catch (error) {
              console.warn("Could not cache geometry.", error);
            }
          }
        } else {
          restoreOriginalCapture();
          setGeometrySummary(geometryHtml(geometry), "bad");
        }
      } catch (error) {
        console.error(error);
        restoreOriginalCapture();
        setGeometrySummary(`Geometry detection failed: ${error.message || error}`, "bad");
      } finally {
        detectButton.disabled = false;
      }
    }, 40);
  }

  function closeDialog() {
    stopCamera();
    if (dialog) dialog.hidden = true;
  }

  async function openScanner() {
    if (dialog) dialog.hidden = false;
    await startCamera();
  }

  async function retake() {
    captured = false;
    lastGeometry = null;
    originalCapturedImageData = null;
    cvProof.textContent = "";
    keepButton.disabled = true;
    await startCamera();
  }

  function keepCapture() {
    if (!captured || !lastGeometry || !lastGeometry.ok || !lastGeometry.locked) {
      setGeometrySummary("A high-confidence LOCK is required before continuing.", "bad");
      return;
    }
    closeDialog();
    const status = byId("status");
    if (status) {
      status.textContent =
        "Pyramid geometry captured: 28/28 tile centers mapped. Next phase: classify each center as emoji1–emoji18 and populate Tiles 1–28.";
    }
  }

  function markCvReady() {
    const cv = window.cv;
    const required = ["Mat","imread","resize","cvtColor","GaussianBlur","Canny"];
    const missing = required.filter((name) => typeof cv?.[name] === "undefined");
    if (missing.length) {
      cvReady = false;
      setCvStatus(
        `OpenCV loaded, but this build is missing: ${missing.join(", ")}`,
        "error"
      );
      return;
    }
    cvReady = true;
    setCvStatus("OpenCV ready — FreeCell-style fixed-template matcher available.", "ready");
  }

  function markCvError(message) {
    cvReady = false;
    setCvStatus(`OpenCV error: ${message || "initialization failed"}`, "error");
  }

  window.addEventListener("ppai-opencv-ready", markCvReady);
  window.addEventListener("ppai-opencv-error", (event) => {
    markCvError(event.detail && event.detail.message);
  });
  window.addEventListener("ppai-opencv-loading", () => {
    setCvStatus("Loading OpenCV…", "working");
  });

  if (window.ppaiCvReady) {
    window.ppaiCvReady.then(markCvReady).catch((error) => {
      markCvError(error.message || String(error));
    });
  }

  if (openButton) openButton.addEventListener("click", openScanner);
  if (captureButton) captureButton.addEventListener("click", captureFrame);
  if (detectButton) detectButton.addEventListener("click", detectGeometry);
  if (stopButton) stopButton.addEventListener("click", stopCamera);
  if (closeButton) closeButton.addEventListener("click", closeDialog);
  if (retakeButton) retakeButton.addEventListener("click", retake);
  if (keepButton) keepButton.addEventListener("click", keepCapture);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopCamera();
  });
  window.addEventListener("pagehide", stopCamera);
}());
