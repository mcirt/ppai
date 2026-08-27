(function(){
"use strict";

const REF=window.PPAI_INVENTORY_REFERENCE_V0135;
const sourceCanvas=document.getElementById("sourceCanvas");
const overlayCanvas=document.getElementById("overlayCanvas");
const normalizedCanvas=document.getElementById("normalizedCanvas");
const statusEl=document.getElementById("status");
const photoInput=document.getElementById("photoInput");
const reanalyzeButton=document.getElementById("reanalyzeButton");
const stage=document.getElementById("stage");
const metrics=document.getElementById("metrics");
const normalizedCard=document.getElementById("normalizedCard");
const opencvStatus=document.getElementById("opencvStatus");
const video=document.getElementById("cameraVideo");
const cameraWrap=document.getElementById("cameraWrap");
const startCameraButton=document.getElementById("startCameraButton");
const takePhotoButton=document.getElementById("takePhotoButton");
const stopCameraButton=document.getElementById("stopCameraButton");

let lastImage=null;
let stream=null;
let templateEdgePoints=null;
let cvReady=false;

function cvHasRequiredApis(){
  const cv=window.cv;
  const required=[
    "Mat","MatVector","imread","resize","cvtColor","GaussianBlur","Canny",
    "getStructuringElement","morphologyEx","findContours","boundingRect",
    "contourArea","arcLength"
  ];
  return !!cv && required.every(name=>typeof cv[name]!=="undefined");
}

function markCvReady(){
  cvReady=cvHasRequiredApis();
  if(cvReady){
    opencvStatus.textContent="OpenCV ready — v0.13.5 geometry-first localizer available.";
    opencvStatus.style.color="#238636";
  }else{
    opencvStatus.textContent="OpenCV loaded, but required contour APIs are missing.";
    opencvStatus.style.color="#b42318";
  }
}

function markCvError(message){
  cvReady=false;
  opencvStatus.textContent=`OpenCV error: ${message||"failed to initialize"}`;
  opencvStatus.style.color="#b42318";
}


function setStatus(text,kind=""){statusEl.textContent=text;statusEl.className=kind;}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pct(v){return `${Math.round(clamp(v,0,1)*100)}%`;}

window.addEventListener("ppai-opencv-ready",markCvReady);
window.addEventListener("ppai-opencv-error",e=>markCvError(e.detail?.message));

if(window.ppaiCvIsReady || cvHasRequiredApis()){
  markCvReady();
}else if(window.ppaiCvReady && typeof window.ppaiCvReady.then==="function"){
  window.ppaiCvReady.then(markCvReady).catch(err=>markCvError(err?.message||String(err)));
}else{
  opencvStatus.textContent="Waiting for OpenCV loader…";
}

function loadImage(url){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>resolve(img);
    img.onerror=()=>reject(new Error(`Could not load ${url}`));
    img.src=url;
  });
}

function drawImageContained(img,canvas,maxWidth=820){
  const sw=img.naturalWidth||img.width, sh=img.naturalHeight||img.height;
  const scale=Math.min(1,maxWidth/sw);
  canvas.width=Math.max(1,Math.round(sw*scale));
  canvas.height=Math.max(1,Math.round(sh*scale));
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
}

function cannyData(canvas){
  const cv=window.cv;
  const src=cv.imread(canvas), gray=new cv.Mat(), blur=new cv.Mat(), edges=new cv.Mat();
  try{
    cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray,blur,new cv.Size(5,5),0,0,cv.BORDER_DEFAULT);
    cv.Canny(blur,edges,45,135);
    return {width:edges.cols,height:edges.rows,data:new Uint8Array(edges.data)};
  }finally{src.delete();gray.delete();blur.delete();edges.delete();}
}

function dilateBinary(edge,w,h,radius=2){
  const out=new Uint8Array(w*h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    if(!edge[y*w+x])continue;
    for(let dy=-radius;dy<=radius;dy++){
      const yy=y+dy;if(yy<0||yy>=h)continue;
      for(let dx=-radius;dx<=radius;dx++){
        const xx=x+dx;if(xx>=0&&xx<w)out[yy*w+xx]=1;
      }
    }
  }
  return out;
}

async function buildTemplateEdgePoints(){
  if(templateEdgePoints)return templateEdgePoints;
  const [img,maskImg]=await Promise.all([
    loadImage("images/inventory_reference_v0135.png?v=0135"),
    loadImage("images/inventory_reference_mask_v0135.png?v=0135")
  ]);
  const tc=document.createElement("canvas");
  tc.width=REF.canonicalWidth;tc.height=REF.canonicalHeight;
  tc.getContext("2d",{willReadFrequently:true}).drawImage(img,0,0);
  const mc=document.createElement("canvas");
  mc.width=REF.canonicalWidth;mc.height=REF.canonicalHeight;
  mc.getContext("2d",{willReadFrequently:true}).drawImage(maskImg,0,0);

  const edge=cannyData(tc);
  const md=mc.getContext("2d",{willReadFrequently:true}).getImageData(0,0,mc.width,mc.height).data;
  const pts=[];
  for(let y=1;y<edge.height-1;y++)for(let x=1;x<edge.width-1;x++){
    const i=y*edge.width+x;
    if(edge.data[i] && md[i*4]>127)pts.push([x,y]);
  }
  const target=700;
  const sampled=[];
  if(pts.length<=target)sampled.push(...pts);
  else for(let i=0;i<target;i++)sampled.push(pts[Math.floor(i*(pts.length-1)/(target-1))]);
  templateEdgePoints=sampled;
  return sampled;
}

function orderTriplet(g){return [...g].sort((a,b)=>a.x-b.x);}

function detectCircleTriplet(canvas){
  const cv=window.cv;
  let src=null,small=null,gray=null,blurred=null,edges=null,closed=null,kernel=null,contours=null,hierarchy=null;
  try{
    src=cv.imread(canvas);
    const targetWidth=Math.min(720,src.cols);
    const scale=targetWidth/src.cols;
    const targetHeight=Math.max(1,Math.round(src.rows*scale));
    small=new cv.Mat(); cv.resize(src,small,new cv.Size(targetWidth,targetHeight),0,0,cv.INTER_AREA);
    gray=new cv.Mat(); cv.cvtColor(small,gray,cv.COLOR_RGBA2GRAY);
    blurred=new cv.Mat(); cv.GaussianBlur(gray,blurred,new cv.Size(7,7),0);
    edges=new cv.Mat(); cv.Canny(blurred,edges,42,132);
    closed=new cv.Mat(); kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,new cv.Size(5,5));
    cv.morphologyEx(edges,closed,cv.MORPH_CLOSE,kernel);
    contours=new cv.MatVector(); hierarchy=new cv.Mat();
    cv.findContours(closed,contours,hierarchy,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE);

    const minBox=targetWidth*0.07, maxBox=targetWidth*0.32;
    const candidates=[];
    for(let i=0;i<contours.size();i++){
      const contour=contours.get(i);
      try{
        const rect=cv.boundingRect(contour);
        if(rect.width<minBox||rect.height<minBox||rect.width>maxBox||rect.height>maxBox)continue;
        const aspect=rect.width/Math.max(1,rect.height);
        if(aspect<0.68||aspect>1.42)continue;
        const area=Math.abs(cv.contourArea(contour));
        const perimeter=cv.arcLength(contour,true);
        if(perimeter<=0)continue;
        const circularity=4*Math.PI*area/(perimeter*perimeter);
        if(circularity<0.20)continue;
        candidates.push({
          x:rect.x+rect.width/2,y:rect.y+rect.height/2,
          r:(rect.width+rect.height)/4,circularity
        });
      }finally{contour.delete();}
    }

    candidates.sort((a,b)=>b.circularity-a.circularity);
    const dedup=[];
    for(const c of candidates){
      const duplicate=dedup.some(q=>Math.hypot(c.x-q.x,c.y-q.y)<Math.min(c.r,q.r)*0.70);
      if(!duplicate)dedup.push(c);
    }

    const canon=REF.circles;
    const cX=canon[0],cY=canon[1],cZ=canon[2];
    const canonAngle=Math.atan2(cZ.cy-cX.cy,cZ.cx-cX.cx);
    const canonSpan=Math.hypot(cZ.cx-cX.cx,cZ.cy-cX.cy);
    let best=null;

    for(let i=0;i<dedup.length-2;i++)for(let j=i+1;j<dedup.length-1;j++)for(let k=j+1;k<dedup.length;k++){
      const g=orderTriplet([dedup[i],dedup[j],dedup[k]]);
      const span=Math.hypot(g[2].x-g[0].x,g[2].y-g[0].y);
      const rMean=(g[0].r+g[1].r+g[2].r)/3;
      if(span<rMean*2.7||span>rMean*5.2)continue;

      const scale=span/canonSpan;
      const angle=Math.atan2(g[2].y-g[0].y,g[2].x-g[0].x)-canonAngle;
      const ca=Math.cos(angle),sa=Math.sin(angle);
      function project(p){
        const dx=p.cx-cX.cx,dy=p.cy-cX.cy;
        return {x:g[0].x+(dx*ca-dy*sa)*scale,y:g[0].y+(dx*sa+dy*ca)*scale};
      }
      const py=project(cY);
      const yErr=Math.hypot(py.x-g[1].x,py.y-g[1].y)/(rMean+1e-6);
      if(yErr>0.72)continue;

      const rVar=(Math.abs(g[0].r-rMean)+Math.abs(g[1].r-rMean)+Math.abs(g[2].r-rMean))/(3*rMean+1e-6);
      if(rVar>0.34)continue;

      const expectedR=[cX.rimR,cY.rimR,cZ.rimR].map(r=>r*scale);
      const radiusErr=(Math.abs(g[0].r-expectedR[0])+Math.abs(g[1].r-expectedR[1])+Math.abs(g[2].r-expectedR[2]))/(expectedR.reduce((a,b)=>a+b,0)+1e-6);
      const circ=(g[0].circularity+g[1].circularity+g[2].circularity)/3;
      const score=yErr*2.2+rVar*1.4+radiusErr*0.9+(1-circ)*0.35;
      if(!best||score<best.score)best={circles:g,score,scale,angle,project};
    }

    if(!best)throw new Error(`No valid X/Y/Z circle triplet found (${dedup.length} circle candidates).`);

    const sx=canvas.width/targetWidth, sy=canvas.height/targetHeight;
    const mapped=best.circles.map(c=>({x:c.x*sx,y:c.y*sy,r:c.r*(sx+sy)/2,circularity:c.circularity}));
    return {
      circles:mapped,
      smallCircles:best.circles,
      scale:best.scale,
      angle:best.angle,
      candidateCount:dedup.length,
      targetWidth,targetHeight
    };
  }finally{
    [hierarchy,contours,kernel,closed,edges,blurred,gray,small,src].forEach(m=>{if(m&&typeof m.delete==="function")m.delete();});
  }
}

function transformFromDetected(detection){
  const X=REF.circles[0], Z=REF.circles[2];
  const dX=detection.circles[0], dZ=detection.circles[2];
  const canonAngle=Math.atan2(Z.cy-X.cy,Z.cx-X.cx);
  const sceneAngle=Math.atan2(dZ.y-dX.y,dZ.x-dX.x);
  const canonSpan=Math.hypot(Z.cx-X.cx,Z.cy-X.cy);
  const sceneSpan=Math.hypot(dZ.x-dX.x,dZ.y-dX.y);
  const scale=sceneSpan/canonSpan;
  const angle=sceneAngle-canonAngle;
  const c=Math.cos(angle),s=Math.sin(angle);
  return {
    scale,angle,
    map(x,y){
      const dx=x-X.cx,dy=y-X.cy;
      return {x:dX.x+(dx*c-dy*s)*scale,y:dX.y+(dx*s+dy*c)*scale};
    }
  };
}

function maskSupport(edge,w,h,transform,points){
  let hit=0,n=0;
  for(const q of points){
    const p=transform.map(q[0],q[1]);
    const x=Math.round(p.x),y=Math.round(p.y);
    if(x<0||x>=w||y<0||y>=h)continue;
    n++; if(edge[y*w+x])hit++;
  }
  return n?hit/n:0;
}

function drawOverlay(detection,transform,maskScore){
  overlayCanvas.width=sourceCanvas.width;overlayCanvas.height=sourceCanvas.height;
  const ctx=overlayCanvas.getContext("2d");
  ctx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);

  ctx.lineWidth=Math.max(3,sourceCanvas.width/260);
  ctx.strokeStyle="#ff2b2b";
  for(const outline of REF.plaqueOutlines){
    ctx.beginPath();
    outline.forEach((q,i)=>{
      const p=transform.map(q[0],q[1]);
      if(i===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);
    });
    ctx.stroke();
  }

  const colors=["#35e5ff","#ffd84a","#56e36d"];
  REF.circles.forEach((c,i)=>{
    const p=transform.map(c.cx,c.cy);
    const rr=c.rimR*transform.scale;
    ctx.beginPath();ctx.arc(p.x,p.y,rr,0,Math.PI*2);
    ctx.strokeStyle=colors[i];ctx.stroke();
    const half=c.cropSide*transform.scale/2;
    ctx.strokeRect(p.x-half,p.y-half,half*2,half*2);
    ctx.fillStyle="rgba(0,0,0,.72)";
    ctx.fillRect(p.x-42,p.y-19,84,30);
    ctx.fillStyle="#fff";ctx.font=`bold ${Math.max(14,sourceCanvas.width/42)}px Arial`;
    ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText(c.slot,p.x,p.y-4);
  });

  ctx.fillStyle="rgba(0,0,0,.76)";
  ctx.fillRect(8,8,Math.min(340,sourceCanvas.width-16),34);
  ctx.fillStyle="#fff";ctx.textAlign="left";
  ctx.font=`bold ${Math.max(13,sourceCanvas.width/48)}px Arial`;
  ctx.fillText(`permanent-geometry support ${pct(maskScore)}`,16,25);
}

function buildNormalized(transform){
  normalizedCanvas.width=REF.canonicalWidth;normalizedCanvas.height=REF.canonicalHeight;
  const ctx=normalizedCanvas.getContext("2d");
  const c=Math.cos(transform.angle),s=Math.sin(transform.angle),scale=transform.scale;
  const X=REF.circles[0], dX=transform.map(X.cx,X.cy);

  // Inverse affine mapping from source to canonical.
  const A=c/scale, C=s/scale, B=-s/scale, D=c/scale;
  const E=X.cx-A*dX.x-C*dX.y;
  const F=X.cy-B*dX.x-D*dX.y;
  ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,normalizedCanvas.width,normalizedCanvas.height);
  ctx.setTransform(A,B,C,D,E,F);ctx.drawImage(sourceCanvas,0,0);
  ctx.setTransform(1,0,0,1,0,0);

  for(const circle of REF.circles){
    const cc=document.getElementById(`crop${circle.slot}`);
    cc.width=circle.cropSide;cc.height=circle.cropSide;
    const cctx=cc.getContext("2d");
    cctx.clearRect(0,0,cc.width,cc.height);
    cctx.drawImage(normalizedCanvas,circle.cx-circle.cropSide/2,circle.cy-circle.cropSide/2,
      circle.cropSide,circle.cropSide,0,0,cc.width,cc.height);
  }
}

async function analyze(){
  if(!lastImage)return;
  reanalyzeButton.disabled=true;metrics.hidden=true;normalizedCard.hidden=true;
  setStatus("Finding the three inventory rims first…","working");
  try{
    if(!cvReady){
      if(window.ppaiCvReady && typeof window.ppaiCvReady.then==="function"){
        await window.ppaiCvReady;
        markCvReady();
      }
    }
    if(!cvReady)throw new Error("OpenCV is not ready yet.");
    drawImageContained(lastImage,sourceCanvas,820);
    overlayCanvas.width=sourceCanvas.width;overlayCanvas.height=sourceCanvas.height;stage.hidden=false;

    const detection=detectCircleTriplet(sourceCanvas);
    const transform=transformFromDetected(detection);
    const points=await buildTemplateEdgePoints();
    const raw=cannyData(sourceCanvas);
    const edge=dilateBinary(raw.data,raw.width,raw.height,3);
    const maskScore=maskSupport(edge,raw.width,raw.height,transform,points);

    const canonY=REF.circles[1], predictedY=transform.map(canonY.cx,canonY.cy), actualY=detection.circles[1];
    const yError=Math.hypot(predictedY.x-actualY.x,predictedY.y-actualY.y)/Math.max(1,actualY.r);
    const radii=detection.circles.map(c=>c.r), rMean=radii.reduce((a,b)=>a+b,0)/3;
    const radiusSpread=radii.reduce((s,r)=>s+Math.abs(r-rMean),0)/(3*rMean+1e-6);
    const geometryScore=clamp(1-yError/0.72,0,1)*0.60+clamp(1-radiusSpread/0.34,0,1)*0.40;
    const confidence=clamp(geometryScore*0.68+clamp((maskScore-0.16)/0.42,0,1)*0.32,0,1);

    drawOverlay(detection,transform,maskScore);
    buildNormalized(transform);

    document.getElementById("fitScore").textContent=pct(maskScore);
    document.getElementById("overallScore").textContent=pct(confidence);
    document.getElementById("scaleScore").textContent=`${transform.scale.toFixed(3)}×`;
    document.getElementById("rotationScore").textContent=`${(transform.angle*180/Math.PI).toFixed(1)}°`;
    document.getElementById("rimX").textContent=`detected rim · circularity ${pct(detection.circles[0].circularity)}`;
    document.getElementById("rimY").textContent=`detected rim · circularity ${pct(detection.circles[1].circularity)}`;
    document.getElementById("rimZ").textContent=`detected rim · circularity ${pct(detection.circles[2].circularity)}`;
    metrics.hidden=false;normalizedCard.hidden=false;

    const locked=yError<=0.48 && radiusSpread<=0.27 && maskScore>=0.28;
    if(locked){
      setStatus(`INVENTORY LOCKED. Three-rim geometry passed; permanent plaque geometry support ${pct(maskScore)}. X/Y/Z crops are derived from the detected rims.`,"good");
    }else{
      setStatus(`Candidate rejected. Rim geometry error ${(yError).toFixed(2)}, radius spread ${pct(radiusSpread)}, permanent geometry support ${pct(maskScore)}. Retake closer and centered.`,"bad");
    }
  }catch(err){
    console.error(err);
    setStatus(`NO MATCH: ${err.message||err}`,"bad");
  }finally{reanalyzeButton.disabled=false;}
}

async function startCamera(){
  if(!navigator.mediaDevices?.getUserMedia){
    setStatus("This browser does not provide an in-page camera. Use Choose Existing Photo.","bad");
    return;
  }
  try{
    stopCamera();
    setStatus("Opening rear camera… OpenCV is not required for the camera.","working");
    stream=await navigator.mediaDevices.getUserMedia({
      audio:false,
      video:{facingMode:{ideal:"environment"},width:{ideal:1920},height:{ideal:1080}}
    });
    video.srcObject=stream;cameraWrap.hidden=false;video.hidden=false;
    await video.play();
    takePhotoButton.disabled=false;stopCameraButton.disabled=false;
    setStatus("Camera ready. Center the three inventory plaques and tap Take Photo.");
  }catch(err){setStatus(`Camera could not start: ${err.message||err}`,"bad");}
}
function stopCamera(){
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  if(video){try{video.pause();}catch(_){ } video.srcObject=null;}
  takePhotoButton.disabled=true;stopCameraButton.disabled=true;
}
function captureCamera(){
  if(!stream||video.readyState<2){setStatus("Camera is not ready yet.","bad");return;}
  const c=document.createElement("canvas");
  c.width=video.videoWidth;c.height=video.videoHeight;
  c.getContext("2d").drawImage(video,0,0);
  const img=new Image();
  img.onload=()=>{lastImage=img;stopCamera();cameraWrap.hidden=true;analyze();};
  img.src=c.toDataURL("image/jpeg",0.94);
}

photoInput.addEventListener("change",()=>{
  const file=photoInput.files&&photoInput.files[0];if(!file)return;
  const url=URL.createObjectURL(file),img=new Image();
  img.onload=()=>{URL.revokeObjectURL(url);lastImage=img;stopCamera();cameraWrap.hidden=true;analyze();};
  img.onerror=()=>{URL.revokeObjectURL(url);setStatus("Could not read that image.","bad");};
  img.src=url;
});
reanalyzeButton.addEventListener("click",analyze);
startCameraButton.addEventListener("click",startCamera);
takePhotoButton.addEventListener("click",captureCamera);
stopCameraButton.addEventListener("click",()=>{stopCamera();cameraWrap.hidden=true;setStatus("Camera stopped.");});
window.addEventListener("pagehide",stopCamera);
}());
