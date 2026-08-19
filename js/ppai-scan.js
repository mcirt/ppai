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

  function average(values) {
    if (!values.length) return 0;
    return values.reduce((s, v) => s + v, 0) / values.length;
  }

  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function rotatePoint(x, y, cx, cy, angle) {
    const ca = Math.cos(angle);
    const sa = Math.sin(angle);
    const dx = x - cx;
    const dy = y - cy;
    return {
      x: cx + dx * ca - dy * sa,
      y: cy + dx * sa + dy * ca
    };
  }

  function estimateBottomRowCandidates(circles, width, height) {
    // Bottom pyramid row normally lives in the lower-middle region.
    const pool = circles.filter((c) =>
      c.y > height * 0.42 &&
      c.y < height * 0.70 &&
      c.x > width * 0.06 &&
      c.x < width * 0.94
    );

    // Try every reasonably horizontal 7-candidate subset created from sorted x.
    const sorted = [...pool].sort((a, b) => a.x - b.x);
    const results = [];

    if (sorted.length < 7) return results;

    for (let i = 0; i <= sorted.length - 7; i += 1) {
      const group = sorted.slice(i, i + 7);

      const gaps = [];
      for (let j = 1; j < group.length; j += 1) {
        gaps.push(group[j].x - group[j - 1].x);
      }
      const spacing = median(gaps);
      if (spacing <= 0) continue;

      const yMean = average(group.map((c) => c.y));
      const ySpread = Math.max(...group.map((c) => Math.abs(c.y - yMean)));

      const gapError = average(gaps.map((g) => Math.abs(g - spacing) / spacing));
      const centerX = (group[0].x + group[6].x) / 2;
      const centerPenalty = Math.abs(centerX - width / 2) / Math.max(1, width);
      const widthSpan = group[6].x - group[0].x;
      const expectedSpan = spacing * 6;
      const spanError = Math.abs(widthSpan - expectedSpan) / Math.max(1, expectedSpan);

      // Infer row tilt from first-to-last candidate.
      const angle = Math.atan2(group[6].y - group[0].y, group[6].x - group[0].x);

      const score =
        gapError * 9 +
        (ySpread / Math.max(1, spacing)) * 3 +
        centerPenalty * 2 +
        spanError * 4;

      results.push({
        group,
        spacing,
        centerX,
        centerY: yMean,
        angle,
        score
      });
    }

    return results.sort((a, b) => a.score - b.score);
  }

  function buildIdealPyramid(bottomFit, verticalSpacingFactor = 0.88) {
    const positions = [];

    const centerX = bottomFit.centerX;
    const bottomY = bottomFit.centerY;
    const hSpacing = bottomFit.spacing;
    const vSpacing = hSpacing * verticalSpacingFactor;
    const tilt = bottomFit.angle;

    // Build an untilted lattice first, then rotate around bottom-row center.
    let tileId = 1;
    for (let row = 1; row <= 7; row += 1) {
      const count = row;
      const y = bottomY - (7 - row) * vSpacing;
      for (let col = 1; col <= count; col += 1) {
        const x = centerX + (col - (count + 1) / 2) * hSpacing;
        const rotated = rotatePoint(x, y, centerX, bottomY, tilt);
        positions.push({
          tileId,
          row,
          col,
          x: rotated.x,
          y: rotated.y
        });
        tileId += 1;
      }
    }

    return { positions, hSpacing, vSpacing, tilt, centerX, bottomY };
  }

  function nearestCircleForPosition(position, circles, maxDistance, used) {
    let best = null;
    let bestIndex = -1;
    let bestD = Infinity;

    for (let i = 0; i < circles.length; i += 1) {
      if (used.has(i)) continue;
      const d = distance(position, circles[i]);
      if (d < bestD && d <= maxDistance) {
        best = circles[i];
        bestIndex = i;
        bestD = d;
      }
    }

    return { circle: best, index: bestIndex, distance: bestD };
  }

  function scoreLatticeAgainstCircles(lattice, circles) {
    const used = new Set();
    const matched = [];
    const maxDistance = lattice.hSpacing * 0.42;

    let totalError = 0;
    let bottomSupported = 0;

    for (const p of lattice.positions) {
      const hit = nearestCircleForPosition(p, circles, maxDistance, used);

      if (hit.circle) {
        used.add(hit.index);
        const normalizedError = hit.distance / Math.max(1, lattice.hSpacing);
        totalError += normalizedError;

        if (p.row === 7) bottomSupported += 1;

        matched.push({
          ...p,
          x: hit.circle.x,
          y: hit.circle.y,
          r: hit.circle.r,
          supported: true,
          inferred: false,
          error: hit.distance
        });
      } else {
        totalError += 0.80;
        matched.push({
          ...p,
          r: median(circles.map((c) => c.r)) || lattice.hSpacing * 0.35,
          supported: false,
          inferred: true,
          error: null
        });
      }
    }

    const supportedCount = matched.filter((m) => m.supported).length;
    const inferredCount = 28 - supportedCount;

    // Heavy penalties for weak bottom-row support and too many inferred centers.
    const score =
      totalError +
      inferredCount * 0.55 +
      Math.max(0, 7 - bottomSupported) * 1.6;

    return {
      matched,
      supportedCount,
      inferredCount,
      bottomSupported,
      score
    };
  }

  function fitPyramidGeometry(circles, workWidth, workHeight) {
    if (circles.length < 12) {
      return {
        ok: false,
        locked: false,
        reason: `Too few usable contour centers (${circles.length}). Retake with the full pyramid visible and less glare.`,
        rawCount: circles.length
      };
    }

    const bottomFits = estimateBottomRowCandidates(circles, workWidth, workHeight);
    if (!bottomFits.length) {
      return {
        ok: false,
        locked: false,
        reason: "Could not establish a credible seven-tile bottom row.",
        rawCount: circles.length
      };
    }

    let best = null;

    // Evaluate several vertical-spacing factors because camera perspective changes
    // row spacing slightly from photo to photo.
    const verticalFactors = [0.78, 0.82, 0.86, 0.90, 0.94, 0.98];

    for (const bottomFit of bottomFits.slice(0, 12)) {
      // The bottom row itself must already be fairly regular.
      if (bottomFit.score > 4.5) continue;

      for (const vf of verticalFactors) {
        const lattice = buildIdealPyramid(bottomFit, vf);
        const scored = scoreLatticeAgainstCircles(lattice, circles);

        const combinedScore = scored.score + bottomFit.score * 1.3;

        if (!best || combinedScore < best.combinedScore) {
          best = {
            bottomFit,
            lattice,
            scored,
            combinedScore
          };
        }
      }
    }

    if (!best) {
      return {
        ok: false,
        locked: false,
        reason: "No globally consistent 28-position pyramid lattice could be fitted.",
        rawCount: circles.length
      };
    }

    const { scored, lattice } = best;
    const centers = scored.matched.map((m) => ({
      tileId: m.tileId,
      row: m.row,
      col: m.col,
      x: m.x,
      y: m.y,
      r: m.r,
      inferred: m.inferred
    }));

    const rowDiagnostics = [];
    for (let row = 1; row <= 7; row += 1) {
      const rowCenters = scored.matched.filter((m) => m.row === row);
      const supported = rowCenters.filter((m) => m.supported).length;
      rowDiagnostics.push({
        row,
        supported,
        expected: row
      });
    }

    const supportedCount = scored.supportedCount;
    const inferredCount = scored.inferredCount;
    const bottomSupported = scored.bottomSupported;

    // Stronger confidence gating than v0.10.4.
    let quality = "bad";
    let locked = false;

    if (supportedCount >= 26 && bottomSupported >= 6) {
      quality = "good";
      locked = true;
    } else if (supportedCount >= 23 && bottomSupported >= 6) {
      quality = "warn";
      locked = false;
    } else {
      quality = "bad";
      locked = false;
    }

    return {
      ok: centers.length === 28,
      locked,
      centers,
      supportedCount,
      inferredCount,
      bottomSupported,
      rowDiagnostics,
      rawCount: circles.length,
      quality,
      hSpacing: lattice.hSpacing,
      vSpacing: lattice.vSpacing,
      detector: "bottom-row anchored constrained lattice",
      combinedScore: best.combinedScore
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
    let blurred = null;
    let edges = null;
    let closed = null;
    let contours = null;
    let hierarchy = null;
    let kernel = null;

    try {
      const cv = window.cv;
      src = cv.imread(canvas);

      const targetWidth = Math.min(760, src.cols);
      const scale = targetWidth / src.cols;
      const targetHeight = Math.max(1, Math.round(src.rows * scale));

      small = new cv.Mat();
      cv.resize(
        src,
        small,
        new cv.Size(targetWidth, targetHeight),
        0,
        0,
        cv.INTER_AREA
      );

      // This is intentionally the same family of operations that the working
      // FreeCell photo scanner already uses with this exact OpenCV build.
      gray = new cv.Mat();
      cv.cvtColor(small, gray, cv.COLOR_RGBA2GRAY);

      blurred = new cv.Mat();
      cv.GaussianBlur(gray, blurred, new cv.Size(7, 7), 0);

      edges = new cv.Mat();
      cv.Canny(blurred, edges, 45, 135);

      // Close small breaks in the circular emblem outlines so findContours()
      // sees them as compact objects instead of fragmented arcs.
      closed = new cv.Mat();
      kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
      cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);

      contours = new cv.MatVector();
      hierarchy = new cv.Mat();
      cv.findContours(
        closed,
        contours,
        hierarchy,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE
      );

      const imageArea = targetWidth * targetHeight;
      const minDim = Math.min(targetWidth, targetHeight);

      // The photographed circular emblems are typically a few percent of the
      // working width. Keep a deliberately broad range because camera distance
      // and perspective vary.
      const minBox = Math.max(15, targetWidth * 0.025);
      const maxBox = Math.max(minBox + 5, targetWidth * 0.13);
      const circles = [];
      const debugCandidates = [];

      for (let i = 0; i < contours.size(); i += 1) {
        const contour = contours.get(i);
        try {
          const rect = cv.boundingRect(contour);
          const w = rect.width;
          const h = rect.height;

          if (w < minBox || h < minBox || w > maxBox || h > maxBox) continue;

          const aspect = w / Math.max(1, h);
          if (aspect < 0.66 || aspect > 1.50) continue;

          // Restrict candidate search to the broad area where the pyramid
          // appears in real camera captures. This removes most UI controls.
          const cx = rect.x + w / 2;
          const cy = rect.y + h / 2;
          if (cx < targetWidth * 0.07 || cx > targetWidth * 0.93) continue;
          if (cy < targetHeight * 0.10 || cy > targetHeight * 0.78) continue;

          const area = Math.abs(cv.contourArea(contour));
          const boxArea = w * h;
          if (boxArea <= 0) continue;

          // Circular/ring contours do not always fill the rectangle, especially
          // after Canny, so use a forgiving fill range.
          const fill = area / boxArea;
          if (fill < 0.10 || fill > 0.92) continue;

          const perimeter = cv.arcLength(contour, true);
          if (perimeter <= 0) continue;

          // 4*pi*A/P^2 approaches 1 for compact circles. Edge rings and partial
          // contours score lower, so keep a loose threshold.
          const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
          if (circularity < 0.18) continue;

          const radius = (w + h) / 4;
          circles.push({
            x: cx,
            y: cy,
            r: radius,
            aspect,
            fill,
            circularity
          });
          debugCandidates.push({ x: cx, y: cy, w, h, circularity });
        } finally {
          contour.delete();
        }
      }

      // findContours frequently returns nested contours for one emblem.
      // Merge near-duplicate centers before fitting the seven-row lattice.
      circles.sort((a, b) => b.circularity - a.circularity);
      const deduped = [];
      for (const c of circles) {
        const duplicate = deduped.some((d) => {
          const dx = c.x - d.x;
          const dy = c.y - d.y;
          const minR = Math.min(c.r, d.r);
          return Math.hypot(dx, dy) < Math.max(8, minR * 0.70);
        });
        if (!duplicate) deduped.push(c);
      }

      const geometry = fitPyramidGeometry(deduped, targetWidth, targetHeight);
      geometry.rawContourCount = contours.size();
      geometry.rawCount = deduped.length;
      geometry.scaleToCanvasX = canvas.width / targetWidth;
      geometry.scaleToCanvasY = canvas.height / targetHeight;
      geometry.workWidth = targetWidth;
      geometry.workHeight = targetHeight;
      geometry.detector = "Canny + MORPH_CLOSE + findContours";
      return geometry;
    } finally {
      if (kernel) kernel.delete();
      if (hierarchy) hierarchy.delete();
      if (contours) contours.delete();
      if (closed) closed.delete();
      if (edges) edges.delete();
      if (blurred) blurred.delete();
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
      return `<strong>Pyramid alignment failed.</strong><br>
        ${g.reason || "The fit was not reliable enough."}<br>
        Try retaking with all 28 circular emblems visible and the phone roughly centered.`;
    }

    const chips = g.rowDiagnostics.map((r) =>
      `<div class="geometry-chip">Row ${r.row}: ${r.supported}/${r.expected}</div>`
    ).join("");

    if (g.locked) {
      return `<strong>28-tile pyramid geometry LOCKED.</strong><br>
        ${g.supportedCount}/28 centers directly supported; bottom row ${g.bottomSupported}/7 supported.<br>
        Green = directly supported center. Yellow = lattice inference.
        <div class="geometry-grid">${chips}</div>`;
    }

    if (g.quality === "warn") {
      return `<strong>Alignment found, but NOT LOCKED.</strong><br>
        ${g.supportedCount}/28 centers directly supported; bottom row ${g.bottomSupported}/7 supported.<br>
        The geometry is plausible, but confidence is not high enough to continue automatically.
        <div class="geometry-grid">${chips}</div>`;
    }

    return `<strong>Pyramid alignment uncertain — RETAKE / RETRY.</strong><br>
      ${g.supportedCount}/28 centers directly supported; bottom row ${g.bottomSupported}/7 supported.<br>
      v0.10.5 will not claim a lock when the fit is weak.
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
    setGeometrySummary("Anchoring the 7-tile bottom row, projecting the full 28-position lattice, and validating every legal tile position…", "");

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
    const required = [
      "Mat", "MatVector", "imread", "resize", "cvtColor", "GaussianBlur",
      "Canny", "getStructuringElement", "morphologyEx", "findContours",
      "boundingRect", "contourArea", "arcLength"
    ];
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
    setCvStatus("OpenCV ready — FreeCell-compatible contour detector available.", "ready");
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
