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
  const previewPanel = byId("ppaiPreviewPanel");
  const cameraPanel = byId("ppaiCameraPanel");
  const canvas = byId("ppaiCaptureCanvas");
  const cvProof = byId("ppaiCvProof");
  const cvStatus = byId("opencvStatus");
  const scanStatus = byId("ppaiScanStatus");

  let cameraStream = null;
  let cvReady = false;
  let captured = false;

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
        "Keep all 28 pyramid tiles visible. Exact alignment is not required yet; OpenCV alignment comes in the next phase.",
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

    captured = true;
    stopCamera();
    cameraPanel.hidden = true;
    previewPanel.hidden = false;
    runOpenCvProof();

    // Keep the captured still available for the next geometry milestone.
    try {
      window.ppaiLastPyramidCapture = canvas.toDataURL("image/jpeg", 0.92);
      sessionStorage.setItem("ppaiLastPyramidCapture", window.ppaiLastPyramidCapture);
    } catch (error) {
      console.warn("Could not cache captured image.", error);
    }
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
    cvProof.textContent = "";
    await startCamera();
  }

  function keepCapture() {
    if (!captured) {
      setScanStatus("Capture a pyramid photo first.", "error");
      return;
    }
    closeDialog();
    const status = byId("status");
    if (status) {
      status.textContent =
        "Pyramid photo captured successfully. Next development phase: automatic alignment and 28 tile-center detection.";
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

  // Catch the case where OpenCV finished before this script attached listeners.
  if (window.ppaiCvReady) {
    window.ppaiCvReady.then(markCvReady).catch((error) => {
      markCvError(error.message || String(error));
    });
  }

  if (openButton) openButton.addEventListener("click", openScanner);
  if (captureButton) captureButton.addEventListener("click", captureFrame);
  if (stopButton) stopButton.addEventListener("click", stopCamera);
  if (closeButton) closeButton.addEventListener("click", closeDialog);
  if (retakeButton) retakeButton.addEventListener("click", retake);
  if (keepButton) keepButton.addEventListener("click", keepCapture);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopCamera();
  });
  window.addEventListener("pagehide", stopCamera);
}());
