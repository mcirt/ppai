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
    const width=mat.cols,height=mat.rows,stride=width+1;
    const integral=new Uint32Array((width+1)*(height+1));
    for(let y=0;y<height;y+=1){
      const row=mat.ucharPtr(y); let running=0;
      const out=(y+1)*stride,prev=y*stride;
      for(let x=0;x<width;x+=1){
        running+=row[x]?1:0;
        integral[out+x+1]=integral[prev+x+1]+running;
      }
    }
    return {data:integral,width,height,stride};
  }

  function rectSum(ii,x0,y0,x1,y1){
    x0=Math.max(0,Math.min(ii.width,Math.round(x0))); x1=Math.max(0,Math.min(ii.width,Math.round(x1)));
    y0=Math.max(0,Math.min(ii.height,Math.round(y0))); y1=Math.max(0,Math.min(ii.height,Math.round(y1)));
    if(x1<=x0||y1<=y0)return 0;
    const a=ii.data[y0*ii.stride+x0],b=ii.data[y0*ii.stride+x1],c=ii.data[y1*ii.stride+x0],d=ii.data[y1*ii.stride+x1];
    return d-b-c+a;
  }
  function rectRatio(ii,x0,y0,x1,y1){
    const w=Math.max(1,Math.round(x1)-Math.round(x0)),h=Math.max(1,Math.round(y1)-Math.round(y0));
    return rectSum(ii,x0,y0,x1,y1)/(w*h);
  }

  function pyramidCanonicalPoints(){
    const pts=[]; let tileId=1;
    for(let row=1;row<=7;row+=1){
      const y=-(7-row);
      for(let col=1;col<=row;col+=1){pts.push({tileId,row,col,x:col-(row+1)/2,y});tileId+=1;}
    }
    return pts;
  }
  const PYRAMID_CANONICAL=pyramidCanonicalPoints();

  function transformCanonicalPoint(p,q){
    const rowT=(p.row-1)/6;
    const rowScale=q.topScale+(1-q.topScale)*rowT;
    const lx=p.x*q.pitch*rowScale,ly=p.y*q.pitch*q.vFactor;
    const sx=lx+q.shear*ly,ca=Math.cos(q.angle),sa=Math.sin(q.angle);
    return {x:q.centerX+sx*ca-ly*sa,y:q.bottomY+sx*sa+ly*ca};
  }
  function buildRegisteredTemplate(q){
    const centers=PYRAMID_CANONICAL.map(p=>{const t=transformCanonicalPoint(p,q);return {...p,x:t.x,y:t.y,r:q.pitch*0.39,inferred:false};});
    return {...q,centers};
  }
  function edgePatch(ii,x,y,half){return rectRatio(ii,x-half,y-half,x+half+1,y+half+1);}
  function scoreRingSupport(ii,c,pitch){
    const patch=Math.max(1.5,pitch*0.05),radii=[pitch*0.31,pitch*0.39,pitch*0.47];
    const angles=[0,Math.PI/6,Math.PI/3,Math.PI/2,2*Math.PI/3,5*Math.PI/6,Math.PI,7*Math.PI/6,4*Math.PI/3,3*Math.PI/2,5*Math.PI/3,11*Math.PI/6];
    let best=0;
    for(const r of radii){let sum=0;for(const a of angles)sum+=edgePatch(ii,c.x+Math.cos(a)*r,c.y+Math.sin(a)*r,patch);best=Math.max(best,sum/angles.length);}
    const busy=rectRatio(ii,c.x-pitch*0.24,c.y-pitch*0.24,c.x+pitch*0.24,c.y+pitch*0.24);
    return best*0.82+busy*0.18;
  }
  function buildOuterSilhouetteSegments(t){
    const rows=[];for(let r=1;r<=7;r++)rows[r]=t.centers.filter(c=>c.row===r).sort((a,b)=>a.x-b.x);
    const pts=[rows[1][0]];for(let r=2;r<=7;r++)pts.push(rows[r][0]);for(let i=1;i<rows[7].length;i++)pts.push(rows[7][i]);for(let r=6;r>=2;r--)pts.push(rows[r][rows[r].length-1]);pts.push(rows[1][0]);
    const seg=[];for(let i=0;i<pts.length-1;i++)seg.push([pts[i],pts[i+1]]);return seg;
  }
  function scoreSegment(ii,a,b,half,samples=14){let sum=0;for(let i=0;i<samples;i++){const t=samples===1?.5:i/(samples-1);sum+=edgePatch(ii,a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,half);}return sum/samples;}
  function scoreRegisteredTemplate(ii,t){
    const supports=t.centers.map(c=>scoreRingSupport(ii,c,t.pitch));
    const ringMean=supports.reduce((a,b)=>a+b,0)/28,sorted=[...supports].sort((a,b)=>a-b),low=sorted.slice(0,7).reduce((a,b)=>a+b,0)/7;
    const bottom=supports.slice(21,28).reduce((a,b)=>a+b,0)/7,apex=supports[0];
    const segments=buildOuterSilhouetteSegments(t),half=Math.max(1.5,t.pitch*0.045);
    const outer=segments.reduce((s,[a,b])=>s+scoreSegment(ii,a,b,half,18),0)/segments.length;
    return {score:ringMean*5+low*2.2+bottom*1.7+apex*.8+outer*3,ringSupports:supports,ringMean,lowQuartile:low,bottomMean:bottom,apex,outerMean:outer};
  }
  function templateInsideImage(t,w,h){const pad=t.pitch*.48;return t.centers.every(c=>c.x-pad>=0&&c.y-pad>=0&&c.x+pad<w&&c.y+pad<h);}

  function fitPyramidRegistration(edgeMask){
    const ii=makeIntegralFromBinary(edgeMask),w=edgeMask.cols,h=edgeMask.rows,coarse=[];let best=null;
    function evalQ(q,store=true){const t=buildRegisteredTemplate(q);if(!templateInsideImage(t,w,h))return null;const m=scoreRegisteredTemplate(ii,t),c={template:t,metrics:m};if(store)coarse.push(c);if(!best||m.score>best.metrics.score)best=c;return c;}
    const pmin=Math.round(w*.075),pmax=Math.round(w*.145),pstep=Math.max(4,Math.round(w*.008)),cxstep=Math.max(6,Math.round(w*.012)),bystep=Math.max(7,Math.round(h*.012));
    for(let pitch=pmin;pitch<=pmax;pitch+=pstep)for(let centerX=Math.round(w*.32);centerX<=Math.round(w*.68);centerX+=cxstep)for(let bottomY=Math.round(h*.40);bottomY<=Math.round(h*.74);bottomY+=bystep)for(const vFactor of [.78,.84,.90,.96,1.02])for(const topScale of [.92,.96,1])for(const angle of [-.06,-.03,0,.03,.06])for(const shear of [-.06,-.03,0,.03,.06])evalQ({centerX,bottomY,pitch,vFactor,topScale,angle,shear});
    if(!best)throw new Error('No complete pyramid registration template fit inside the image.');
    let current=best;
    for(const st of [{dp:6,dc:12,dy:14,dv:.05,dt:.04,da:.025,ds:.025,step:2},{dp:3,dc:6,dy:7,dv:.025,dt:.02,da:.012,ds:.012,step:1}]){
      const g=current.template;let sb=current;
      for(let pitch=g.pitch-st.dp;pitch<=g.pitch+st.dp;pitch+=st.step)for(let cx=g.centerX-st.dc;cx<=g.centerX+st.dc;cx+=st.step)for(let by=g.bottomY-st.dy;by<=g.bottomY+st.dy;by+=st.step)for(let vf=g.vFactor-st.dv;vf<=g.vFactor+st.dv+1e-9;vf+=st.dv/2)for(let ts=g.topScale-st.dt;ts<=g.topScale+st.dt+1e-9;ts+=st.dt/2)for(let a=g.angle-st.da;a<=g.angle+st.da+1e-9;a+=st.da/2)for(let sh=g.shear-st.ds;sh<=g.shear+st.ds+1e-9;sh+=st.ds/2){const t=buildRegisteredTemplate({centerX:cx,bottomY:by,pitch,vFactor:vf,topScale:ts,angle:a,shear:sh});if(!templateInsideImage(t,w,h))continue;const m=scoreRegisteredTemplate(ii,t);if(m.score>sb.metrics.score)sb={template:t,metrics:m};}
      current=sb;
    }
    best=current;
    const alternatives=coarse.filter(c=>{const a=c.template,b=best.template;return Math.abs(a.centerX-b.centerX)>b.pitch*.55||Math.abs(a.bottomY-b.bottomY)>b.pitch*.55||Math.abs(a.pitch-b.pitch)>b.pitch*.16||Math.abs(a.angle-b.angle)>.045;}).sort((a,b)=>b.metrics.score-a.metrics.score);
    const second=alternatives[0]||null,margin=second?best.metrics.score-second.metrics.score:best.metrics.score,norm=margin/Math.max(.0001,best.metrics.score);
    const sup=best.metrics.ringSupports,thr=Math.max(.016,best.metrics.ringMean*.43);
    const centers=best.template.centers.map((c,i)=>({...c,inferred:sup[i]<thr,templateSupport:sup[i]}));
    const supportedCount=centers.filter(c=>!c.inferred).length,bottomSupported=centers.filter(c=>c.row===7&&!c.inferred).length,apexSupported=!centers[0].inferred;
    const rowDiagnostics=[];for(let row=1;row<=7;row++){const rr=centers.filter(c=>c.row===row);rowDiagnostics.push({row,supported:rr.filter(c=>!c.inferred).length,expected:row});}
    let quality='bad',locked=false;
    if(supportedCount>=25&&bottomSupported>=6&&apexSupported&&best.metrics.outerMean>=best.metrics.ringMean*.38&&norm>=.02){quality='good';locked=true;}else if(supportedCount>=22&&bottomSupported>=5&&apexSupported)quality='warn';
    return {ok:true,locked,quality,centers,supportedCount,bottomSupported,apexSupported,rowDiagnostics,detector:'full-pyramid-template-registration',registrationScore:best.metrics.score,outerMean:best.metrics.outerMean,ringMean:best.metrics.ringMean,normalizedMargin:norm,transform:{centerX:best.template.centerX,bottomY:best.template.bottomY,pitch:best.template.pitch,vFactor:best.template.vFactor,topScale:best.template.topScale,angle:best.template.angle,shear:best.template.shear}};
  }

  function detectCirclesWithOpenCv(){
    if(!cvReady||!window.cv)throw new Error('OpenCV is not ready.');restoreOriginalCapture();
    let src=null,small=null,gray=null,blurred=null,edges=null;
    try{const cv=window.cv;src=cv.imread(canvas);const targetWidth=Math.min(640,src.cols),scale=targetWidth/src.cols,targetHeight=Math.max(1,Math.round(src.rows*scale));small=new cv.Mat();cv.resize(src,small,new cv.Size(targetWidth,targetHeight),0,0,cv.INTER_AREA);gray=new cv.Mat();cv.cvtColor(small,gray,cv.COLOR_RGBA2GRAY);blurred=new cv.Mat();cv.GaussianBlur(gray,blurred,new cv.Size(5,5),0);edges=new cv.Mat();cv.Canny(blurred,edges,42,126);const result=fitPyramidRegistration(edges);result.scaleToCanvasX=canvas.width/targetWidth;result.scaleToCanvasY=canvas.height/targetHeight;result.workWidth=targetWidth;result.workHeight=targetHeight;return result;}finally{if(edges)edges.delete();if(blurred)blurred.delete();if(gray)gray.delete();if(small)small.delete();if(src)src.delete();}
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
    if(!g.ok)return `<strong>Pyramid registration failed.</strong><br>${g.reason||"No registration fit."}`;
    const chips=g.rowDiagnostics.map(r=>`<div class="geometry-chip">Row ${r.row}: ${r.supported}/${r.expected}</div>`).join("");
    const marginPct=Math.round((g.normalizedMargin||0)*1000)/10;
    if(g.locked)return `<strong>FULL PYRAMID TEMPLATE LOCKED.</strong><br>${g.supportedCount}/28 emoji-circle positions strongly supported; bottom ${g.bottomSupported}/7; apex supported.<br>Outer silhouette and all 28 circles were scored together. Best-vs-second registration separation: ${marginPct}%.<div class="geometry-grid">${chips}</div>`;
    if(g.quality==="warn")return `<strong>Registration found — CHECK ALIGNMENT.</strong><br>${g.supportedCount}/28 circle positions supported; bottom ${g.bottomSupported}/7; apex ${g.apexSupported?"supported":"weak"}.<br>The fixed 28-circle pyramid mask is placed, but confidence is not high enough to continue automatically.<div class="geometry-grid">${chips}</div>`;
    return `<strong>Registration uncertain — RETAKE / RETRY.</strong><br>${g.supportedCount}/28 circle positions supported; bottom ${g.bottomSupported}/7; apex ${g.apexSupported?"supported":"weak"}.<div class="geometry-grid">${chips}</div>`;
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
    setGeometrySummary("Registering the full pyramid mask: outer silhouette + all 28 expected emoji circles…", "");

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
    setCvStatus("OpenCV ready — full-pyramid template registration available.", "ready");
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
