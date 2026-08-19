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

  function median(values) {
    if (!values.length) return 0;
    const a = [...values].sort((x, y) => x - y);
    const mid = Math.floor(a.length / 2);
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }

  function clusterRows(circles) {
    if (!circles.length) return [];
    const radii = circles.map((c) => c.r);
    const rMed = Math.max(4, median(radii));
    const tolerance = rMed * 1.35;

    const sorted = [...circles].sort((a, b) => a.y - b.y);
    const rows = [];

    for (const c of sorted) {
      let best = null;
      let bestDy = Infinity;
      for (const row of rows) {
        const dy = Math.abs(c.y - row.yMean);
        if (dy < tolerance && dy < bestDy) {
          best = row;
          bestDy = dy;
        }
      }

      if (!best) {
        rows.push({ items: [c], yMean: c.y });
      } else {
        best.items.push(c);
        best.yMean = best.items.reduce((s, q) => s + q.y, 0) / best.items.length;
      }
    }

    rows.sort((a, b) => a.yMean - b.yMean);
    for (const row of rows) row.items.sort((a, b) => a.x - b.x);
    return rows;
  }

  function choosePyramidRows(rows, width, height) {
    // Keep plausible rows in the central game area.
    const plausible = rows.filter((row) => {
      const y = row.yMean;
      return y > height * 0.13 && y < height * 0.73 && row.items.length >= 1;
    });

    // Score every 7-row subsequence. Ideal counts are 1,2,...,7.
    let best = null;
    for (let start = 0; start <= plausible.length - 7; start += 1) {
      const block = plausible.slice(start, start + 7);
      let score = 0;
      let valid = true;
      let previousY = null;
      const spacings = [];

      for (let i = 0; i < 7; i += 1) {
        const expected = i + 1;
        const count = block[i].items.length;
        score += Math.abs(count - expected) * 7;
        if (count < expected) score += (expected - count) * 12;

        if (previousY !== null) spacings.push(block[i].yMean - previousY);
        previousY = block[i].yMean;

        const xs = block[i].items.map((c) => c.x);
        const xMean = xs.reduce((s, x) => s + x, 0) / xs.length;
        score += Math.abs(xMean - width / 2) / Math.max(20, width * 0.03);
      }

      const spacingMed = median(spacings);
      if (spacingMed <= 0) valid = false;
      for (const s of spacings) {
        score += Math.abs(s - spacingMed) / Math.max(4, spacingMed) * 5;
      }

      if (valid && (!best || score < best.score)) {
        best = { rows: block, score, spacing: spacingMed };
      }
    }
    return best;
  }

  function selectRowCenters(row, expectedCount, centerX, nominalSpacing) {
    const items = [...row.items].sort((a, b) => a.x - b.x);
    if (items.length === expectedCount) return items;

    // Build expected x positions from the row center and nominal horizontal spacing.
    const rowCenter = items.length
      ? items.reduce((s, c) => s + c.x, 0) / items.length
      : centerX;
    const spacing = nominalSpacing;
    const expectedXs = [];
    for (let i = 0; i < expectedCount; i += 1) {
      expectedXs.push(rowCenter + (i - (expectedCount - 1) / 2) * spacing);
    }

    const chosen = [];
    const available = [...items];
    for (const ex of expectedXs) {
      let bestIndex = -1;
      let bestDist = Infinity;
      for (let i = 0; i < available.length; i += 1) {
        const d = Math.abs(available[i].x - ex);
        if (d < bestDist) {
          bestDist = d;
          bestIndex = i;
        }
      }
      if (bestIndex >= 0) {
        chosen.push(available.splice(bestIndex, 1)[0]);
      }
    }
    chosen.sort((a, b) => a.x - b.x);
    return chosen;
  }

  function fitPyramidGeometry(circles, workWidth, workHeight) {
    const rows = clusterRows(circles);
    const choice = choosePyramidRows(rows, workWidth, workHeight);
    if (!choice) {
      return { ok: false, reason: `Could not find seven pyramid-like circle rows. Raw circle candidates: ${circles.length}.`, rawCount: circles.length };
    }

    const selectedRows = choice.rows;

    // Estimate horizontal spacing from rows with multiple candidates.
    const horizontalGaps = [];
    for (const row of selectedRows) {
      const xs = row.items.map((c) => c.x).sort((a, b) => a - b);
      for (let i = 1; i < xs.length; i += 1) {
        const gap = xs[i] - xs[i - 1];
        if (gap > 8 && gap < workWidth * 0.25) horizontalGaps.push(gap);
      }
    }
    const hSpacing = median(horizontalGaps) || choice.spacing || workWidth * 0.09;

    const centers = [];
    let tileId = 1;
    const rowDiagnostics = [];

    for (let rowIndex = 0; rowIndex < 7; rowIndex += 1) {
      const expectedCount = rowIndex + 1;
      const row = selectedRows[rowIndex];
      const selected = selectRowCenters(row, expectedCount, workWidth / 2, hSpacing);

      rowDiagnostics.push({
        row: rowIndex + 1,
        raw: row.items.length,
        selected: selected.length
      });

      // If Hough missed one or more circles, infer missing lattice positions from
      // row center + spacing. This keeps geometry useful even with a few misses.
      const y = row.yMean;
      const sourceCenterX = selected.length
        ? selected.reduce((s, c) => s + c.x, 0) / selected.length
        : workWidth / 2;

      // Blend row center toward the overall frame center only lightly.
      const rowCenterX = sourceCenterX * 0.85 + (workWidth / 2) * 0.15;
      const selectedByX = [...selected].sort((a, b) => a.x - b.x);

      for (let col = 0; col < expectedCount; col += 1) {
        const idealX = rowCenterX + (col - (expectedCount - 1) / 2) * hSpacing;
        let actual = null;
        let bestD = Infinity;
        for (const c of selectedByX) {
          const d = Math.abs(c.x - idealX);
          if (d < bestD) {
            bestD = d;
            actual = c;
          }
        }

        const useActual = actual && bestD < hSpacing * 0.48;
        centers.push({
          tileId,
          row: rowIndex + 1,
          col: col + 1,
          x: useActual ? actual.x : idealX,
          y: useActual ? actual.y : y,
          r: useActual ? actual.r : median(circles.map((c) => c.r)),
          inferred: !useActual
        });
        tileId += 1;
      }
    }

    const inferredCount = centers.filter((c) => c.inferred).length;
    const quality =
      inferredCount === 0 ? "good" :
      inferredCount <= 4 ? "warn" : "bad";

    return {
      ok: centers.length === 28 && inferredCount <= 8,
      centers,
      inferredCount,
      rowDiagnostics,
      rawCount: circles.length,
      rowScore: choice.score,
      hSpacing,
      vSpacing: choice.spacing,
      quality
    };
  }

  function detectCirclesWithOpenCv() {
    if (!cvReady || !window.cv) {
      throw new Error("OpenCV is not ready.");
    }

    restoreOriginalCapture();

    let src = null;
    let small = null;
    let gray = null;
    let blur = null;
    let circlesMat = null;

    try {
      src = window.cv.imread(canvas);

      const targetWidth = Math.min(760, src.cols);
      const scale = targetWidth / src.cols;
      const targetHeight = Math.max(1, Math.round(src.rows * scale));

      small = new window.cv.Mat();
      window.cv.resize(
        src,
        small,
        new window.cv.Size(targetWidth, targetHeight),
        0,
        0,
        window.cv.INTER_AREA
      );

      gray = new window.cv.Mat();
      blur = new window.cv.Mat();
      window.cv.cvtColor(small, gray, window.cv.COLOR_RGBA2GRAY);
      window.cv.GaussianBlur(
        gray,
        blur,
        new window.cv.Size(7, 7),
        1.6,
        1.6,
        window.cv.BORDER_DEFAULT
      );

      circlesMat = new window.cv.Mat();

      const minRadius = Math.max(10, Math.round(targetWidth * 0.026));
      const maxRadius = Math.max(minRadius + 4, Math.round(targetWidth * 0.075));
      const minDist = Math.max(18, Math.round(targetWidth * 0.055));

      window.cv.HoughCircles(
        blur,
        circlesMat,
        window.cv.HOUGH_GRADIENT,
        1.15,
        minDist,
        110,
        24,
        minRadius,
        maxRadius
      );

      const circles = [];
      const data = circlesMat.data32F || [];
      for (let i = 0; i + 2 < data.length; i += 3) {
        const x = data[i];
        const y = data[i + 1];
        const r = data[i + 2];

        // Filter obvious UI/background circles.
        if (x < targetWidth * 0.05 || x > targetWidth * 0.95) continue;
        if (y < targetHeight * 0.10 || y > targetHeight * 0.76) continue;

        circles.push({ x, y, r });
      }

      const geometry = fitPyramidGeometry(circles, targetWidth, targetHeight);
      geometry.scaleToCanvasX = canvas.width / targetWidth;
      geometry.scaleToCanvasY = canvas.height / targetHeight;
      geometry.workWidth = targetWidth;
      geometry.workHeight = targetHeight;
      return geometry;
    } finally {
      if (circlesMat) circlesMat.delete();
      if (blur) blur.delete();
      if (gray) gray.delete();
      if (small) small.delete();
      if (src) src.delete();
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
    if (!g.ok) {
      return `<strong>Geometry not locked.</strong><br>${g.reason || "The pyramid fit was not reliable enough."}<br>
        Try retaking the photo with all 28 tile circles visible and less glare.`;
    }

    const inferredText = g.inferredCount
      ? `${g.inferredCount} center${g.inferredCount === 1 ? "" : "s"} inferred from the fitted lattice`
      : "all 28 centers directly supported by detected circles";

    const chips = g.rowDiagnostics.map((r) =>
      `<div class="geometry-chip">Row ${r.row}: ${r.selected}/${r.row}</div>`
    ).join("");

    return `<strong>28-tile pyramid geometry locked.</strong><br>
      ${g.rawCount} circle candidates found; ${inferredText}.<br>
      Green = directly supported center. Yellow = center inferred from the regular pyramid lattice.
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
    setGeometrySummary("Analyzing circle candidates and fitting seven pyramid rows…", "");

    // Yield once so Safari paints the status before the heavier OpenCV work.
    window.setTimeout(() => {
      try {
        const geometry = detectCirclesWithOpenCv();
        lastGeometry = geometry;

        if (geometry.ok) {
          drawGeometryOverlay(geometry);
          setGeometrySummary(geometryHtml(geometry), geometry.quality || "good");
          keepButton.disabled = false;

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
    if (!captured || !lastGeometry || !lastGeometry.ok) {
      setGeometrySummary("Detect and lock the 28 tile centers first.", "bad");
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
    cvReady = true;
    setCvStatus("OpenCV ready — local known-good build loaded.", "ready");
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
