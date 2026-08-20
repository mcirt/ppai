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
  let originalPhotoCanvas = null;

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
    originalPhotoCanvas = null;
    keepButton.disabled = true;
    previewPanel.hidden = true;
    cameraPanel.hidden = false;
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
        "Keep the full phone/game display visible. Exact pyramid alignment is not required.",
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

  function restoreCanvasFrom(source) {
    canvas.width = source.width;
    canvas.height = source.height;
    canvas.getContext("2d", { alpha: false }).drawImage(source, 0, 0);
  }

  function restoreOriginalCapture() {
    if (originalPhotoCanvas) {
      restoreCanvasFrom(originalPhotoCanvas);
      return;
    }
    if (!originalCapturedImageData) return;
    canvas.width = originalCapturedImageData.width;
    canvas.height = originalCapturedImageData.height;
    canvas.getContext("2d").putImageData(originalCapturedImageData, 0, 0);
  }

  function captureFrame() {
    if (!video || !video.videoWidth || !video.videoHeight) {
      setScanStatus("The camera preview is not ready yet.", "error");
      return;
    }

    const maxWidth = 1400;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));

    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    originalCapturedImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    originalPhotoCanvas = document.createElement("canvas");
    originalPhotoCanvas.width = canvas.width;
    originalPhotoCanvas.height = canvas.height;
    originalPhotoCanvas.getContext("2d").drawImage(canvas, 0, 0);

    captured = true;
    lastGeometry = null;
    keepButton.disabled = true;
    stopCamera();
    cameraPanel.hidden = true;
    previewPanel.hidden = false;

    setGeometrySummary(
      "<strong>Photo captured.</strong> Press <em>Detect Pyramid Template</em>. v0.11.0 fits the supplied static 28-circle pyramid reference directly to this photo.",
      ""
    );

    runOpenCvProof();
  }

  function runOpenCvProof() {
    if (!cvReady || !window.cv) {
      cvProof.textContent = "Photo captured. OpenCV is not ready.";
      return;
    }
    let src=null, gray=null, edges=null;
    try {
      const cv=window.cv;
      src=cv.imread(canvas);
      gray=new cv.Mat();
      edges=new cv.Mat();
      cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);
      cv.Canny(gray,edges,70,150);
      let count=0;
      const data=edges.data;
      for(let i=0;i<data.length;i++) if(data[i]) count++;
      cvProof.textContent=`OpenCV proof passed: ${src.cols}×${src.rows} frame; ${count.toLocaleString()} edge pixels detected.`;
    } catch(error) {
      console.error(error);
      cvProof.textContent=`OpenCV proof failed: ${error.message||error}`;
    } finally {
      if(edges)edges.delete();
      if(gray)gray.delete();
      if(src)src.delete();
    }
  }

  // ---- Direct static-reference pyramid registration ----
  function buildScreenTemplate(w,h,params) {
    const ref = window.PPAI_PYRAMID_REFERENCE;
    if (!ref || !ref.pyramid || !Array.isArray(ref.pyramid.tileCenters)) {
      throw new Error("Static pyramid reference geometry is not loaded.");
    }

    // Exact normalized geometry measured from the user-supplied straight-on
    // reference screenshot. Search only a small registration transform around it.
    const base = ref.pyramid.tileCenters;
    const cx = params.cx;
    const cy = params.cy;
    const sx = params.sx;
    const sy = params.sy;

    const centers = base.map((p) => {
      const dx = (p.x - ref.pyramid.centerX);
      const dy = (p.y - ref.pyramid.bottomRowCenterY);
      return {
        tileId: p.tileId,
        row: p.row,
        col: p.col,
        x: (cx + dx * sx) * w,
        y: (cy + dy * sy) * h,
        r: p.r * sx * w,
        inferred: false
      };
    });

    return { centers, cx, cy, sx, sy, xPitchPx: ref.pyramid.horizontalPitch*sx*w, yPitchPx: ref.pyramid.verticalPitch*sy*h };
  }

  function sampleGrayEdge(data,w,h,x,y,rad) {
    const x0=Math.max(0,Math.floor(x-rad)), x1=Math.min(w-1,Math.ceil(x+rad));
    const y0=Math.max(0,Math.floor(y-rad)), y1=Math.min(h-1,Math.ceil(y+rad));
    let sum=0,n=0;
    for(let yy=y0;yy<=y1;yy++) {
      const off=yy*w;
      for(let xx=x0;xx<=x1;xx++) {
        if(data[off+xx])sum++;
        n++;
      }
    }
    return n?sum/n:0;
  }

  function scoreTemplate(edgeData,w,h,t) {
    const angles=[0,Math.PI/4,Math.PI/2,3*Math.PI/4,Math.PI,5*Math.PI/4,3*Math.PI/2,7*Math.PI/4];
    const supports=[];
    const patch=Math.max(1.5,t.xPitchPx*0.035);

    for(const c of t.centers) {
      let ring=0;
      for(const a of angles) {
        ring+=sampleGrayEdge(edgeData,w,h,c.x+Math.cos(a)*c.r,c.y+Math.sin(a)*c.r,patch);
      }
      ring/=angles.length;
      supports.push(ring);
    }

    const mean=supports.reduce((a,b)=>a+b,0)/28;
    const sorted=[...supports].sort((a,b)=>a-b);
    const low=sorted.slice(0,7).reduce((a,b)=>a+b,0)/7;
    const bottom=supports.slice(21,28).reduce((a,b)=>a+b,0)/7;
    const apex=supports[0];

    return {
      score:mean*5 + low*2.5 + bottom*2 + apex,
      supports,mean,low,bottom,apex
    };
  }

  function fitPyramidOnPhoto(sourceCanvas) {
    const cv=window.cv;
    const ref=window.PPAI_PYRAMID_REFERENCE;
    if(!ref) throw new Error("Static pyramid reference is not loaded.");

    let src=null,small=null,gray=null,blurred=null,edges=null;
    try {
      src=cv.imread(sourceCanvas);

      // Keep this light enough for iPhone Safari.
      const targetWidth=Math.min(560,src.cols);
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

      const edgeData=new Uint8Array(edges.data.length);
      for(let i=0;i<edges.data.length;i++) edgeData[i]=edges.data[i]?1:0;

      // Direct registration against the exact normalized geometry measured from
      // the supplied straight-on screenshot.
      //
      // Because the camera photo may include bezel/background, search a broad
      // placement/scale range — but only four transform parameters:
      // translation X/Y and independent X/Y scale.
      const baseCx=ref.pyramid.centerX;
      const baseCy=ref.pyramid.bottomRowCenterY;

      let best=null;
      const candidates=[];

      function evaluate(cx,cy,sx,sy) {
        const t=buildScreenTemplate(targetWidth,targetHeight,{cx,cy,sx,sy});
        const metrics=scoreTemplate(edgeData,targetWidth,targetHeight,t);
        const c={template:t,metrics};
        candidates.push(c);
        if(!best || metrics.score>best.metrics.score) best=c;
      }

      // Coarse direct fit. These ranges intentionally allow the pyramid to occupy
      // a smaller/larger fraction of the photographed frame.
      for(let cx=0.34;cx<=0.66+1e-9;cx+=0.02) {
        for(let cy=0.42;cy<=0.68+1e-9;cy+=0.02) {
          for(let sx=0.55;sx<=1.15+1e-9;sx+=0.06) {
            for(let sy=0.55;sy<=1.15+1e-9;sy+=0.06) {
              evaluate(cx,cy,sx,sy);
            }
          }
        }
      }

      if(!best) throw new Error("Static pyramid registration evaluated no candidates.");

      // Fine fit around the winner.
      let refined=best;
      const b=best.template;
      for(let cx=b.cx-0.025;cx<=b.cx+0.025+1e-9;cx+=0.005) {
        for(let cy=b.cy-0.025;cy<=b.cy+0.025+1e-9;cy+=0.005) {
          for(let sx=b.sx-0.06;sx<=b.sx+0.06+1e-9;sx+=0.015) {
            for(let sy=b.sy-0.06;sy<=b.sy+0.06+1e-9;sy+=0.015) {
              const t=buildScreenTemplate(targetWidth,targetHeight,{cx,cy,sx,sy});
              const metrics=scoreTemplate(edgeData,targetWidth,targetHeight,t);
              if(metrics.score>refined.metrics.score) refined={template:t,metrics};
            }
          }
        }
      }

      const alternatives=candidates.filter(c=>
        Math.abs(c.template.cx-refined.template.cx)>0.05 ||
        Math.abs(c.template.cy-refined.template.cy)>0.05 ||
        Math.abs(c.template.sx-refined.template.sx)>0.12 ||
        Math.abs(c.template.sy-refined.template.sy)>0.12
      ).sort((a,b)=>b.metrics.score-a.metrics.score);

      const second=alternatives[0]||null;
      const margin=second?refined.metrics.score-second.metrics.score:refined.metrics.score;
      const normalizedMargin=margin/Math.max(1e-6,refined.metrics.score);
      const threshold=Math.max(0.010,refined.metrics.mean*0.42);

      const sxCanvas=sourceCanvas.width/targetWidth;
      const syCanvas=sourceCanvas.height/targetHeight;

      const centers=refined.template.centers.map((c,i)=>({
        ...c,
        x:c.x*sxCanvas,
        y:c.y*syCanvas,
        r:c.r*(sxCanvas+syCanvas)/2,
        inferred:refined.metrics.supports[i]<threshold,
        templateSupport:refined.metrics.supports[i]
      }));

      const supportedCount=centers.filter(c=>!c.inferred).length;
      const bottomSupported=centers.filter(c=>c.row===7&&!c.inferred).length;
      const apexSupported=!centers[0].inferred;

      const rowDiagnostics=[];
      for(let row=1;row<=7;row++) {
        const rr=centers.filter(c=>c.row===row);
        rowDiagnostics.push({row,supported:rr.filter(c=>!c.inferred).length,expected:row});
      }

      let quality="bad",locked=false;
      if(supportedCount>=25 && bottomSupported>=6 && apexSupported && normalizedMargin>=0.010) {
        quality="good"; locked=true;
      } else if(supportedCount>=21 && bottomSupported>=5) {
        quality="warn";
      }

      return {
        ok:true,locked,quality,centers,supportedCount,bottomSupported,apexSupported,
        rowDiagnostics,normalizedMargin,
        workWidth:sourceCanvas.width,workHeight:sourceCanvas.height,
        scaleToCanvasX:1,scaleToCanvasY:1,
        detector:"direct-static-reference-registration",
        referenceImage:ref.referenceImage,
        registration:{
          cx:refined.template.cx,
          cy:refined.template.cy,
          sx:refined.template.sx,
          sy:refined.template.sy
        }
      };
    } finally {
      if(edges)edges.delete();
      if(blurred)blurred.delete();
      if(gray)gray.delete();
      if(small)small.delete();
      if(src)src.delete();
    }
  }

  function drawGeometryOverlay(g) {
    restoreOriginalCapture();
    const ctx=canvas.getContext("2d");
    ctx.save();
    ctx.lineWidth=Math.max(2,canvas.width/420);
    ctx.font=`bold ${Math.max(13,Math.round(canvas.width/45))}px Arial`;
    ctx.textAlign="center";
    ctx.textBaseline="middle";

    for(const c of g.centers) {
      ctx.beginPath();
      ctx.arc(c.x,c.y,Math.max(12,c.r*0.72),0,Math.PI*2);
      ctx.strokeStyle=c.inferred?"#ffd400":"#24e36a";
      ctx.stroke();

      const lr=Math.max(10,canvas.width/55);
      ctx.beginPath();
      ctx.arc(c.x,c.y,lr,0,Math.PI*2);
      ctx.fillStyle=c.inferred?"rgba(120,85,0,.88)":"rgba(0,70,28,.88)";
      ctx.fill();
      ctx.fillStyle="#fff";
      ctx.fillText(String(c.tileId),c.x,c.y+0.5);
    }
    ctx.restore();
  }

  function geometryHtml(g) {
    const chips=g.rowDiagnostics.map(r=>
      `<div class="geometry-chip">Row ${r.row}: ${r.supported}/${r.expected}</div>`
    ).join("");
    const marginPct=Math.round((g.normalizedMargin||0)*1000)/10;

    if(g.locked) {
      return `<strong>PYRAMID TEMPLATE LOCKED.</strong><br>
        ${g.supportedCount}/28 positions supported; bottom ${g.bottomSupported}/7; apex ${g.apexSupported?"supported":"weak"}.<br>
        Best-vs-second template separation: ${marginPct}%.
        <div class="geometry-grid">${chips}</div>`;
    }
    if(g.quality==="warn") {
      return `<strong>Template fit found — CHECK PYRAMID ALIGNMENT.</strong><br>
        ${g.supportedCount}/28 positions supported; bottom ${g.bottomSupported}/7.
        <div class="geometry-grid">${chips}</div>`;
    }
    return `<strong>Pyramid template fit is uncertain.</strong><br>
      ${g.supportedCount}/28 positions supported; bottom ${g.bottomSupported}/7.
      <div class="geometry-grid">${chips}</div>`;
  }

  function runDirectTemplateRegistration() {
    if(!captured || !cvReady || !originalPhotoCanvas) return;

    detectButton.disabled=true;
    keepButton.disabled=true;
    setGeometrySummary(
      "Fitting the exact static pyramid reference directly to the camera photo — no four-corner or screen-rectification step…",
      ""
    );

    window.setTimeout(()=>{
      try {
        const g=fitPyramidOnPhoto(originalPhotoCanvas);
        lastGeometry=g;
        drawGeometryOverlay(g);
        setGeometrySummary(geometryHtml(g),g.locked?"good":(g.quality==="warn"?"warn":"bad"));
        keepButton.disabled=!g.locked;

        if(g.locked) {
          try {
            const normalized=g.centers.map(c=>({
              tileId:c.tileId,row:c.row,col:c.col,
              x:c.x/g.workWidth,y:c.y/g.workHeight,r:c.r/g.workWidth,
              inferred:!!c.inferred
            }));
            sessionStorage.setItem("ppaiLastPyramidGeometry",JSON.stringify(normalized));
            sessionStorage.setItem("ppaiLastPyramidCapture",originalPhotoCanvas.toDataURL("image/jpeg",0.94));
          } catch(e) {
            console.warn("Could not cache direct-registration result.",e);
          }
        }
      } catch(error) {
        console.error(error);
        setGeometrySummary(`Direct pyramid template registration failed: ${error.message||error}`,"bad");
      } finally {
        detectButton.disabled=false;
      }
    },40);
  }


  function closeDialog() {
    stopCamera();
    dialog.hidden=true;
  }

  async function retake() {
    await startCamera();
  }

  function keepCapture() {
    if(!lastGeometry || !lastGeometry.locked) {
      setGeometrySummary("A high-confidence pyramid LOCK is required before continuing.","bad");
      return;
    }
    closeDialog();
    const status=byId("status");
    if(status)status.textContent="Pyramid geometry captured: 28 template positions mapped after screen perspective correction. Next phase: emoji recognition for Tiles 1–28.";
  }

  function markCvReady() {
    const cv=window.cv;
    const required=["Mat","imread","resize","cvtColor","GaussianBlur","Canny"];
    const missing=required.filter(name=>typeof cv?.[name]==="undefined");
    if(missing.length) {
      cvReady=false;
      setCvStatus(`OpenCV loaded, but this build is missing: ${missing.join(", ")}`,"error");
      return;
    }
    cvReady=true;
    setCvStatus("OpenCV ready — direct static pyramid reference registration available.","ready");
  }

  function markCvError(message) {
    cvReady=false;
    setCvStatus(`OpenCV error: ${message||"initialization failed"}`,"error");
  }

  window.addEventListener("ppai-opencv-ready",markCvReady);
  window.addEventListener("ppai-opencv-error",e=>markCvError(e.detail&&e.detail.message));
  window.addEventListener("ppai-opencv-loading",()=>setCvStatus("Loading OpenCV…","working"));
  if(window.ppaiCvReady)window.ppaiCvReady.then(markCvReady).catch(e=>markCvError(e.message||String(e)));

  openButton?.addEventListener("click",async()=>{dialog.hidden=false;await startCamera();});
  captureButton?.addEventListener("click",captureFrame);
  detectButton?.addEventListener("click",runDirectTemplateRegistration);
  stopButton?.addEventListener("click",stopCamera);
  closeButton?.addEventListener("click",closeDialog);
  retakeButton?.addEventListener("click",retake);
  keepButton?.addEventListener("click",keepCapture);

  document.addEventListener("visibilitychange",()=>{if(document.hidden)stopCamera();});
  window.addEventListener("pagehide",stopCamera);
}());
