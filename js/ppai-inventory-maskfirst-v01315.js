(function(){
"use strict";

const VERSION="0.13.15";
const sourceCanvas=document.getElementById("sourceCanvas");
const maskCanvas=document.getElementById("maskCanvas");
const cleanedCanvas=document.getElementById("cleanedCanvas");
const candidateCanvas=document.getElementById("candidateCanvas");
const statusEl=document.getElementById("status");
const debugEl=document.getElementById("maskDebug");
const opencvStatus=document.getElementById("opencvStatus");
const photoInput=document.getElementById("photoInput");
const reanalyzeButton=document.getElementById("reanalyzeButton");
const presetSelect=document.getElementById("maskPreset");
const video=document.getElementById("cameraVideo");
const cameraWrap=document.getElementById("cameraWrap");
const startCameraButton=document.getElementById("startCameraButton");
const takePhotoButton=document.getElementById("takePhotoButton");
const stopCameraButton=document.getElementById("stopCameraButton");
const resultGrid=document.getElementById("resultGrid");
const candidateInfo=document.getElementById("candidateInfo");

let lastImage=null;
let stream=null;
let cvReady=false;

function setStatus(text,kind=""){statusEl.textContent=text;statusEl.className=kind;}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pct(v){return `${Math.round(clamp(v,0,1)*100)}%`;}

function cvHasRequiredApis(){
  const cv=window.cv;
  const required=["Mat","imread","imshow","cvtColor","getStructuringElement","morphologyEx","findContours","boundingRect","contourArea"];
  return !!cv && required.every(name=>typeof cv[name]!=="undefined");
}
function markCvReady(){
  cvReady=cvHasRequiredApis();
  if(cvReady){
    opencvStatus.textContent=`OpenCV ready — v${VERSION} mask-first inventory test available.`;
    opencvStatus.style.color="#238636";
  }else{
    opencvStatus.textContent="OpenCV loaded, but this build is missing a required basic contour API.";
    opencvStatus.style.color="#b42318";
  }
}
function markCvError(message){
  cvReady=false;
  opencvStatus.textContent=`OpenCV error: ${message||"failed to initialize"}`;
  opencvStatus.style.color="#b42318";
}
window.addEventListener("ppai-opencv-ready",markCvReady);
window.addEventListener("ppai-opencv-error",e=>markCvError(e.detail?.message));
if(window.ppaiCvIsReady||cvHasRequiredApis()) markCvReady();
else if(window.ppaiCvReady&&typeof window.ppaiCvReady.then==="function"){
  window.ppaiCvReady.then(markCvReady).catch(err=>markCvError(err?.message||String(err)));
}

function rgbToHsv(r,g,b){
  r/=255; g/=255; b/=255;
  const max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min;
  let h=0;
  if(d){
    if(max===r)h=((g-b)/d)%6;
    else if(max===g)h=(b-r)/d+2;
    else h=(r-g)/d+4;
    h*=60;if(h<0)h+=360;
  }
  return [h,max?d/max:0,max];
}

function drawImageContained(img,canvas,maxWidth=900){
  const sw=img.naturalWidth||img.width,sh=img.naturalHeight||img.height;
  const scale=Math.min(1,maxWidth/sw);
  canvas.width=Math.max(1,Math.round(sw*scale));
  canvas.height=Math.max(1,Math.round(sh*scale));
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
}

function preset(){
  // Tuned to the pale gray/cream plaque material, not the changing emoji colors.
  const p=presetSelect.value;
  if(p==="strict") return {maxSat:.31,minVal:.56,close:5,open:3};
  if(p==="loose")  return {maxSat:.47,minVal:.39,close:9,open:3};
  return {maxSat:.39,minVal:.48,close:7,open:3}; // balanced
}

function buildRawMask(){
  const cfg=preset();
  const w=sourceCanvas.width,h=sourceCanvas.height;
  maskCanvas.width=w;maskCanvas.height=h;
  const sctx=sourceCanvas.getContext("2d",{willReadFrequently:true});
  const src=sctx.getImageData(0,0,w,h);
  const out=new ImageData(w,h);

  // The short viewfinder is intentionally used as the search image.
  // A mild central prior is applied only to the mask edges, not to the actual pixels.
  for(let y=0;y<h;y++){
    for(let x=0;x<w;x++){
      const i=(y*w+x)*4;
      const r=src.data[i],g=src.data[i+1],b=src.data[i+2];
      const [,sat,val]=rgbToHsv(r,g,b);

      // Pale plaque material tends to be reasonably bright and relatively low-saturation.
      // Also admit slightly warmer cream pixels when bright.
      const creamBonus=(r>g*.92 && g>b*.84 && val>.48);
      const isPlaque=(val>=cfg.minVal && sat<=cfg.maxSat) || (creamBonus && sat<.52 && val>.58);

      const v=isPlaque?255:0;
      out.data[i]=out.data[i+1]=out.data[i+2]=v;
      out.data[i+3]=255;
    }
  }
  maskCanvas.getContext("2d",{willReadFrequently:true}).putImageData(out,0,0);
}

function cleanMaskAndFindCandidate(){
  const cv=window.cv;
  const raw=cv.imread(maskCanvas);
  const gray=new cv.Mat();
  const cleaned=new cv.Mat();
  const contours=new cv.MatVector();
  const hierarchy=new cv.Mat();
  let kernel=null;

  try{
    cv.cvtColor(raw,gray,cv.COLOR_RGBA2GRAY);
    const cfg=preset();
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,new cv.Size(cfg.close,cfg.close));
    cv.morphologyEx(gray,cleaned,cv.MORPH_CLOSE,kernel);
    kernel.delete();
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,new cv.Size(cfg.open,cfg.open));
    cv.morphologyEx(cleaned,cleaned,cv.MORPH_OPEN,kernel);
    cv.imshow(cleanedCanvas,cleaned);

    cv.findContours(cleaned,contours,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE);

    const W=cleaned.cols,H=cleaned.rows;
    const cx0=W/2,cy0=H/2;
    let best=null;

    for(let i=0;i<contours.size();i++){
      const c=contours.get(i);
      const rect=cv.boundingRect(c);
      const area=cv.contourArea(c);
      c.delete();

      if(rect.width<40||rect.height<40) continue;
      const cx=rect.x+rect.width/2,cy=rect.y+rect.height/2;
      const wf=rect.width/W,hf=rect.height/H;
      const ar=rect.width/Math.max(1,rect.height);
      const fill=area/Math.max(1,rect.width*rect.height);
      const centerDist=Math.hypot((cx-cx0)/(W*.5),(cy-cy0)/(H*.5));

      // Expected cluster is central, medium-sized, wider than tall.
      if(wf<.22||wf>.86||hf<.18||hf>.86) continue;
      if(ar<1.05||ar>2.65) continue;
      if(fill<.16||fill>.96) continue;

      const arScore=1-clamp(Math.abs(ar-1.58)/1.05,0,1);
      const centerScore=1-clamp(centerDist/1.05,0,1);
      const sizeScore=1-clamp(Math.abs(wf-.48)/.45,0,1);
      const fillScore=1-clamp(Math.abs(fill-.55)/.55,0,1);
      const score=arScore*.38+centerScore*.32+sizeScore*.18+fillScore*.12;

      if(!best||score>best.score) best={rect,area,fill,ar,score,centerScore};
    }

    drawCandidate(cleaned,best);
    return best;
  } finally {
    if(kernel) kernel.delete();
    raw.delete();gray.delete();cleaned.delete();contours.delete();hierarchy.delete();
  }
}

function drawCandidate(cleanedMat,best){
  const W=cleanedMat.cols,H=cleanedMat.rows;
  candidateCanvas.width=W;candidateCanvas.height=H;
  const ctx=candidateCanvas.getContext("2d");
  ctx.clearRect(0,0,W,H);
  ctx.drawImage(cleanedCanvas,0,0);

  // Darken non-white mask a little so candidate box reads clearly.
  ctx.strokeStyle=best?"#35d05b":"#ff4141";
  ctx.lineWidth=Math.max(3,W/220);
  if(best){
    const r=best.rect;
    ctx.strokeRect(r.x,r.y,r.width,r.height);
    ctx.fillStyle="rgba(0,0,0,.78)";
    ctx.fillRect(r.x,Math.max(0,r.y-34),Math.min(r.width,430),30);
    ctx.fillStyle="#fff";
    ctx.font=`bold ${Math.max(12,W/52)}px Arial`;
    ctx.textBaseline="middle";
    ctx.fillText(`best central plaque-like blob · shape ${pct(best.score)}`,r.x+7,Math.max(15,r.y-19));
  }
}

async function analyze(){
  if(!lastImage)return;
  reanalyzeButton.disabled=true;
  setStatus("Converting the exact viewfinder photo into a black-and-white plaque mask…","working");
  try{
    if(!cvReady){
      if(window.ppaiCvReady&&typeof window.ppaiCvReady.then==="function"){
        await window.ppaiCvReady; markCvReady();
      }
    }
    if(!cvReady) throw new Error("OpenCV is not ready.");

    drawImageContained(lastImage,sourceCanvas,900);
    buildRawMask();
    const best=cleanMaskAndFindCandidate();
    resultGrid.hidden=false;

    if(best){
      candidateInfo.innerHTML=
        `<strong>Candidate shape score: ${pct(best.score)}</strong><br>`+
        `bbox aspect ${best.ar.toFixed(2)} · fill ${pct(best.fill)} · centrality ${pct(best.centerScore)}.`;
      debugEl.textContent=
        `This build is intentionally testing only mask formation + connected-shape localization. `+
        `It does not trust emoji recognition until this mask stage is visually reliable.`;

      if(best.score>=.62){
        setStatus("MASK CANDIDATE FOUND. Inspect the middle and right panels: the white blob should correspond to the three inventory plaques, not DRAW/HOLD/background.","good");
      }else{
        setStatus("A central plaque-like blob was found, but its silhouette is weak. Inspect the B&W mask before trusting any localization.","bad");
      }
    }else{
      candidateInfo.innerHTML="<strong>No plausible central three-plaque blob found.</strong>";
      debugEl.textContent="Try Balanced, Strict, or Loose. The goal is for the three pale inventory plaques to become one dominant white object.";
      setStatus("NO MASK CANDIDATE. The threshold did not produce a plausible central inventory silhouette.","bad");
    }
  }catch(err){
    console.error(err);
    setStatus(`MASK TEST ERROR: ${err.message||err}`,"bad");
  }finally{
    reanalyzeButton.disabled=false;
  }
}

async function startCamera(){
  if(!navigator.mediaDevices?.getUserMedia){
    setStatus("This browser does not provide an in-page camera. Use Choose Existing Photo.","bad");return;
  }
  try{
    stopCamera();
    setStatus("Opening rear camera…","working");
    stream=await navigator.mediaDevices.getUserMedia({
      audio:false,
      video:{facingMode:{ideal:"environment"},width:{ideal:1920},height:{ideal:1080}}
    });
    video.srcObject=stream;
    cameraWrap.hidden=false;video.hidden=false;
    await video.play();
    takePhotoButton.disabled=false;stopCameraButton.disabled=false;
    setStatus("Camera ready. Put the three inventory plaques in the short viewfinder and tap Take Photo.");
  }catch(err){
    setStatus(`Camera could not start: ${err.message||err}`,"bad");
  }
}

function stopCamera(){
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  if(video){try{video.pause();}catch(_){ }video.srcObject=null;}
  takePhotoButton.disabled=true;stopCameraButton.disabled=true;
}

function captureCamera(){
  if(!stream||video.readyState<2){setStatus("Camera is not ready yet.","bad");return;}
  // Known-good behavior: capture exactly the part of the camera image visible in the short viewfinder.
  const vw=video.videoWidth,vh=video.videoHeight;
  const boxW=Math.max(1,cameraWrap.clientWidth),boxH=Math.max(1,cameraWrap.clientHeight);
  const coverScale=Math.max(boxW/vw,boxH/vh);
  const visibleW=boxW/coverScale,visibleH=boxH/coverScale;
  const sx=(vw-visibleW)/2,sy=(vh-visibleH)/2;
  const targetW=Math.min(1600,Math.max(900,Math.round(visibleW)));
  const targetH=Math.round(targetW*(boxH/boxW));

  const c=document.createElement("canvas");
  c.width=targetW;c.height=targetH;
  c.getContext("2d").drawImage(video,sx,sy,visibleW,visibleH,0,0,targetW,targetH);

  const img=new Image();
  img.onload=()=>{
    lastImage=img;
    stopCamera();
    cameraWrap.hidden=true;
    analyze();
  };
  img.src=c.toDataURL("image/jpeg",.95);
}

photoInput.addEventListener("change",()=>{
  const file=photoInput.files&&photoInput.files[0];if(!file)return;
  const url=URL.createObjectURL(file);
  const img=new Image();
  img.onload=()=>{URL.revokeObjectURL(url);lastImage=img;stopCamera();cameraWrap.hidden=true;analyze();};
  img.onerror=()=>{URL.revokeObjectURL(url);setStatus("Could not read that image.","bad");};
  img.src=url;
});

reanalyzeButton.addEventListener("click",analyze);
presetSelect.addEventListener("change",()=>{if(lastImage)analyze();});
startCameraButton.addEventListener("click",startCamera);
takePhotoButton.addEventListener("click",captureCamera);
stopCameraButton.addEventListener("click",()=>{stopCamera();cameraWrap.hidden=true;setStatus("Camera stopped.");});
window.addEventListener("pagehide",stopCamera);
}());
