(function(){
"use strict";

const VERSION="0.13.23";
const REF=window.PPAI_INVENTORY_CLUSTER_REFERENCE_V01323;
const sourceCanvas=document.getElementById("sourceCanvas");
const maskCanvas=document.getElementById("maskCanvas");
const cleanedCanvas=document.getElementById("cleanedCanvas");
const candidateCanvas=document.getElementById("candidateCanvas");
const sourceOverlayCanvas=document.getElementById("sourceOverlayCanvas");
const statusEl=document.getElementById("status");
const debugEl=document.getElementById("clusterDebug");
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
const recognitionSummary=document.getElementById("recognitionSummary");

let lastImage=null;
let stream=null;
let cvReady=false;
let emojiReferences=null;
let templateData=null;

function setStatus(text,kind=""){statusEl.textContent=text;statusEl.className=kind;}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pct(v){return `${Math.round(clamp(v,0,1)*100)}%`;}

function cvHasRequiredApis(){
  const cv=window.cv;
  const required=["Mat","imread","imshow","cvtColor","getStructuringElement","morphologyEx","GaussianBlur","Canny"];
  return !!cv && required.every(name=>typeof cv[name]!=="undefined");
}
function markCvReady(){
  cvReady=cvHasRequiredApis();
  if(cvReady){
    opencvStatus.textContent=`OpenCV ready — v${VERSION} dark-hole primary + silhouette fallback available.`;
    opencvStatus.style.color="#238636";
  }else{
    opencvStatus.textContent="OpenCV loaded, but required basic mask APIs are missing.";
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
  r/=255;g/=255;b/=255;
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
function loadImage(url){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error(`Could not load ${url}`));img.src=url;});}

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
  const p=presetSelect.value;
  if(p==="strict") return {maxSat:.31,minVal:.56,close:5,open:3};
  if(p==="loose") return {maxSat:.47,minVal:.39,close:9,open:3};
  return {maxSat:.39,minVal:.48,close:7,open:3};
}
function buildRawMask(){
  const cfg=preset(),w=sourceCanvas.width,h=sourceCanvas.height;
  maskCanvas.width=w;maskCanvas.height=h;
  const src=sourceCanvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,w,h);
  const out=new ImageData(w,h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const i=(y*w+x)*4,r=src.data[i],g=src.data[i+1],b=src.data[i+2];
    const [,sat,val]=rgbToHsv(r,g,b);
    const creamBonus=(r>g*.92&&g>b*.84&&val>.48);
    const isPlaque=(val>=cfg.minVal&&sat<=cfg.maxSat)||(creamBonus&&sat<.52&&val>.58);
    const v=isPlaque?255:0;
    out.data[i]=out.data[i+1]=out.data[i+2]=v;out.data[i+3]=255;
  }
  maskCanvas.getContext("2d",{willReadFrequently:true}).putImageData(out,0,0);
}
function cleanMask(){
  const cv=window.cv,raw=cv.imread(maskCanvas),gray=new cv.Mat(),cleaned=new cv.Mat();
  let k1=null,k2=null;
  try{
    cv.cvtColor(raw,gray,cv.COLOR_RGBA2GRAY);
    const cfg=preset();
    k1=cv.getStructuringElement(cv.MORPH_ELLIPSE,new cv.Size(cfg.close,cfg.close));
    cv.morphologyEx(gray,cleaned,cv.MORPH_CLOSE,k1);
    k2=cv.getStructuringElement(cv.MORPH_ELLIPSE,new cv.Size(cfg.open,cfg.open));
    cv.morphologyEx(cleaned,cleaned,cv.MORPH_OPEN,k2);
    cv.imshow(cleanedCanvas,cleaned);
  }finally{if(k1)k1.delete();if(k2)k2.delete();raw.delete();gray.delete();cleaned.delete();}
}


// ---------------- Whole-cluster silhouette fallback ----------------
// This is intentionally SECONDARY. v0.13.18 dark-window triplet geometry remains
// the primary path. Only when that path cannot produce a trustworthy X/Y/Z lock do
// we register the known three-plaque silhouette extracted from a real game snapshot.
// The stored cluster transform then predicts the X/Y/Z window centers.

function binaryCanvasData(canvas){
  const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,canvas.width,canvas.height).data;
  const out=new Uint8Array(canvas.width*canvas.height);
  for(let i=0;i<out.length;i++)out[i]=d[i*4]>=128?1:0;
  return out;
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
function boundaryFromBinary(bin,w,h){
  const out=new Uint8Array(w*h);
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){
    const i=y*w+x;if(!bin[i])continue;
    if(!bin[i-1]||!bin[i+1]||!bin[i-w]||!bin[i+w])out[i]=1;
  }
  return out;
}
function edgeBandsFromBinary(bin,w,h){
  const raw=boundaryFromBinary(bin,w,h);
  return {exact:dilateBinary(raw,w,h,1),tight:dilateBinary(raw,w,h,3),tolerance:dilateBinary(raw,w,h,6)};
}
function weightedEdgeSupport(points,transform,bands,w,h){
  let sum=0,n=0;
  for(const q of points){
    const p=transform.map(q[0],q[1]),x=Math.round(p.x),y=Math.round(p.y);
    if(x<0||x>=w||y<0||y>=h)continue;
    const i=y*w+x;
    const sc=bands.exact[i]?1:(bands.tight[i]?0.82:(bands.tolerance[i]?0.52:0));
    sum+=sc;n++;
  }
  return n?sum/n:0;
}
function transformedPointHit(points,transform,data,w,h){
  let sum=0,n=0;
  for(const q of points){
    const p=transform.map(q[0],q[1]);
    const x=Math.round(p.x),y=Math.round(p.y);
    if(x<0||x>=w||y<0||y>=h)continue;
    sum+=data[y*w+x]?1:0;n++;
  }
  return n?sum/n:0;
}

async function buildTemplateData(){
  if(templateData)return templateData;
  // The approved v0.13.21 PNG is intentionally reused BYTE-FOR-BYTE.
  // v0.13.23 keeps the approved silhouette unchanged as a last-resort fallback.
  const silhouetteImg=await loadImage("images/inventory_cluster_silhouette_v01321.png?v=01321");
  const c=document.createElement("canvas");c.width=REF.canonicalWidth;c.height=REF.canonicalHeight;
  const ctx=c.getContext("2d",{willReadFrequently:true});ctx.drawImage(silhouetteImg,0,0);
  const bin=binaryCanvasData(c),boundaryPoints=[],bodyPoints=[],outsidePoints=[];
  const inside=(x,y)=>!!bin[y*c.width+x];
  // Exact OUTER boundary of the approved solid silhouette.
  for(let y=2;y<c.height-2;y+=3)for(let x=2;x<c.width-2;x+=3){
    if(inside(x,y)&&(!inside(x-2,y)||!inside(x+2,y)||!inside(x,y-2)||!inside(x,y+2)))boundaryPoints.push([x,y]);
  }
  // Because BOTH sides are solid filled silhouettes now, every interior region is valid shape evidence.
  for(let y=10;y<c.height-10;y+=14)for(let x=10;x<c.width-10;x+=14)if(inside(x,y))bodyPoints.push([x,y]);
  // Narrow ring just outside the approved shape. A good registration should keep these BLACK.
  for(let y=8;y<c.height-8;y+=14)for(let x=8;x<c.width-8;x+=14){
    if(inside(x,y))continue;
    let near=false;
    for(const [dx,dy] of [[8,0],[-8,0],[0,8],[0,-8],[14,0],[-14,0],[0,14],[0,-14]]){
      const xx=x+dx,yy=y+dy;if(xx>=0&&xx<c.width&&yy>=0&&yy<c.height&&inside(xx,yy)){near=true;break;}
    }
    if(near)outsidePoints.push([x,y]);
  }
  templateData={boundaryPoints,bodyPoints,outsidePoints};
  return templateData;
}

function makeTransform(scale,angle,tx,ty){
  const c=Math.cos(angle),sn=Math.sin(angle),cx=REF.templateCenter[0],cy=REF.templateCenter[1];
  return {scale,angle,tx,ty,map(x,y){const dx=x-cx,dy=y-cy;return {x:tx+(dx*c-dy*sn)*scale,y:ty+(dx*sn+dy*c)*scale};}};
}

function chooseEnvelopeRoi(comps,w,h){
  // Restrict the fallback to the known inventory neighborhood BEFORE contour extraction.
  // Dark components are only used to tighten the ROI; they are not required.
  const useful=comps.filter(c=>c.cx>w*.12&&c.cx<w*.88&&c.cy>h*.10&&c.cy<h*.65)
                    .sort((a,b)=>b.r-a.r).slice(0,5);
  let x0=w*.18,x1=w*.82,y0=h*.14,y1=h*.74;
  if(useful.length>=2){
    const rs=useful.map(c=>c.r).sort((a,b)=>a-b),r=rs[Math.floor(rs.length/2)];
    const xs=useful.map(c=>c.cx),ys=useful.map(c=>c.cy);
    x0=Math.min(...xs)-r*2.3;x1=Math.max(...xs)+r*2.3;
    y0=Math.min(...ys)-r*1.7;y1=Math.max(...ys)+r*4.8;
    // Never let noisy components collapse the search box too tightly.
    const minW=w*.38,minH=h*.34,cx=(x0+x1)/2,cy=(y0+y1)/2;
    if(x1-x0<minW){x0=cx-minW/2;x1=cx+minW/2;}
    if(y1-y0<minH){y0=cy-minH*.38;y1=cy+minH*.62;}
  }
  x0=Math.max(0,Math.floor(x0));y0=Math.max(0,Math.floor(y0));x1=Math.min(w,Math.ceil(x1));y1=Math.min(h,Math.ceil(y1));
  return {x:x0,y:y0,width:Math.max(1,x1-x0),height:Math.max(1,y1-y0)};
}

function extractSolidEnvelope(comps){
  const cv=window.cv,w=cleanedCanvas.width,h=cleanedCanvas.height;
  const src=cv.imread(cleanedCanvas),gray=new cv.Mat();
  let roiMat=null,closed=null,kernel=null,contours=null,hierarchy=null,localMask=null,fullMask=null,dstRoi=null;
  try{
    cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);
    const roi=chooseEnvelopeRoi(comps,w,h);
    roiMat=gray.roi(new cv.Rect(roi.x,roi.y,roi.width,roi.height));
    closed=new cv.Mat();
    let k=Math.max( nineOdd(Math.round(Math.min(w,h)*.028)),  nineOdd(9) );
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,new cv.Size(k,k));
    // Important: CLOSE gaps, do not permanently dilate the shape.
    cv.morphologyEx(roiMat,closed,cv.MORPH_CLOSE,kernel);
    contours=new cv.MatVector();hierarchy=new cv.Mat();
    cv.findContours(closed,contours,hierarchy,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE);
    let best=null;
    for(let i=0;i<contours.size();i++){
      const c=contours.get(i),rect=cv.boundingRect(c),area=cv.contourArea(c);
      const ar=rect.width/Math.max(1,rect.height),fill=area/Math.max(1,rect.width*rect.height);
      const wf=rect.width/w,hf=rect.height/h,cx=roi.x+rect.x+rect.width/2,cy=roi.y+rect.y+rect.height/2;
      const touches=rect.x<=2||rect.y<=2||rect.x+rect.width>=roi.width-2||rect.y+rect.height>=roi.height-2;
      // The actual three-plaque cluster is a medium, central, wider-than-tall object.
      let score=0;
      if(!touches&&wf>=.25&&wf<=.72&&hf>=.18&&hf<=.60&&ar>=1.12&&ar<=1.82&&fill>=.40){
        const arScore=1-clamp(Math.abs(ar-REF.aspectRatio)/.48,0,1);
        const centerScore=1-clamp(Math.hypot((cx-w*.50)/(w*.42),(cy-h*.40)/(h*.40)),0,1);
        const fillScore=1-clamp(Math.abs(fill-.73)/.42,0,1);
        score=arScore*.42+centerScore*.35+fillScore*.23;
      }
      if(score>0&&(!best||score>best.score))best={index:i,rect,area,ar,fill,score,cx,cy};
      c.delete();
    }
    if(!best)return null;
    localMask=cv.Mat.zeros(roi.height,roi.width,cv.CV_8UC1);
    cv.drawContours(localMask,contours,best.index,new cv.Scalar(255),-1);
    fullMask=cv.Mat.zeros(h,w,cv.CV_8UC1);dstRoi=fullMask.roi(new cv.Rect(roi.x,roi.y,roi.width,roi.height));localMask.copyTo(dstRoi);
    const c=document.createElement("canvas");c.width=w;c.height=h;cv.imshow(c,fullMask);
    const bin=binaryCanvasData(c),bands=edgeBandsFromBinary(bin,w,h);
    return {roi,bbox:{x:roi.x+best.rect.x,y:roi.y+best.rect.y,width:best.rect.width,height:best.rect.height},bin,bands,extractScore:best.score,aspect:best.ar,fill:best.fill,canvas:c};
  }finally{
    [dstRoi,fullMask,localMask,hierarchy,contours,kernel,closed,roiMat,gray,src].forEach(m=>{try{if(m&&m.delete)m.delete();}catch(_){}});
  }
}
function nineOdd(v){v=Math.max(3,Math.round(v));return v%2?v:v+1;}

function scoreTransformAgainstEnvelope(transform,env,template,w,h){
  const boundary=weightedEdgeSupport(template.boundaryPoints,transform,env.bands,w,h);
  const body=transformedPointHit(template.bodyPoints,transform,env.bin,w,h);
  const outside=transformedPointHit(template.outsidePoints,transform,env.bin,w,h);
  const score=boundary*.52+body*.36+(1-outside)*.12;
  return {score,contour:boundary,body,seams:0,rims:0,outside};
}

async function findInventoryCluster(canvas,comps){
  const template=await buildTemplateData(),w=canvas.width,h=canvas.height;
  const env=extractSolidEnvelope(comps);
  if(!env)throw new Error("Could not extract one plausible solid plaque-cluster envelope from the B/W mask.");

  // Fit approved silhouette to the EXTRACTED SOLID SHAPE, not to the photograph.
  const wb=REF.whiteBounds,templateH=wb[3]-wb[1]+1;
  const initialScale=env.bbox.height/templateH;
  const cx=env.bbox.x+env.bbox.width/2,cy=env.bbox.y+env.bbox.height/2;
  let best=null;
  const angles=[-8,-6,-4,-2,0,2,4,6,8].map(v=>v*Math.PI/180);
  const scales=[.92,.95,.975,1,1.025,1.05,1.08];
  const dxStep=Math.max(2,Math.round(env.bbox.width*.018)),dyStep=Math.max(2,Math.round(env.bbox.height*.018));
  for(const sf of scales)for(const angle of angles)for(let dy=-2*dyStep;dy<=2*dyStep;dy+=dyStep)for(let dx=-2*dxStep;dx<=2*dxStep;dx+=dxStep){
    const tr=makeTransform(initialScale*sf,angle,cx+dx,cy+dy),sc=scoreTransformAgainstEnvelope(tr,env,template,w,h);
    if(!best||sc.score>best.score)best={...sc,transform:tr,envelope:env};
  }
  if(!best)throw new Error("No silhouette-to-envelope registration candidate was generated.");
  return best;
}

function silhouetteLockPass(found){
  return !!found && found.score>=0.68 && found.contour>=0.58 && found.body>=0.70;
}

function silhouetteTriplet(found){
  const tr=found.transform;
  const slots={};
  for(const c of REF.circles){
    const p=tr.map(c.cx,c.cy);
    const r=c.rimR*tr.scale*0.72; // inner-window-like radius for diagnostics only
    slots[c.slot]={cx:p.x,cy:p.y,r,bw:r*2,bh:r*2,source:"silhouette"};
  }
  const d1=slots.Y.cx-slots.X.cx,d2=slots.Z.cx-slots.Y.cx;
  const rMean=(slots.X.r+slots.Y.r+slots.Z.r)/3;
  return {slots,score:found.score,alignScore:found.contour,sizeScore:found.body,spacingBalance:found.seams,centerScore:found.rims,rMean,d1,d2,source:"silhouette"};
}

function drawSilhouetteFallback(comps,found){
  const w=cleanedCanvas.width,h=cleanedCanvas.height;
  candidateCanvas.width=w;candidateCanvas.height=h;
  const ctx=candidateCanvas.getContext("2d");ctx.drawImage(cleanedCanvas,0,0);
  // Show the actual solid envelope the code extracted from the B/W mask.
  if(found&&found.envelope&&found.envelope.canvas){
    ctx.save();ctx.globalAlpha=.28;ctx.globalCompositeOperation="source-over";
    ctx.drawImage(found.envelope.canvas,0,0);ctx.restore();
    const b=found.envelope.bbox;
    ctx.strokeStyle="#8a2be2";ctx.lineWidth=Math.max(2,w/260);ctx.setLineDash([8,6]);ctx.strokeRect(b.x,b.y,b.width,b.height);ctx.setLineDash([]);
  }
  ctx.lineWidth=Math.max(2,w/260);ctx.strokeStyle="rgba(255,170,0,.55)";
  for(const c of comps)ctx.strokeRect(c.cx-c.bw/2,c.cy-c.bh/2,c.bw,c.bh);
  if(!found)return;
  const tr=found.transform;
  ctx.strokeStyle="#ff3b30";ctx.lineWidth=Math.max(3,w/220);ctx.beginPath();
  REF.outerContour.forEach((q,i)=>{const p=tr.map(q[0],q[1]);if(i===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);});
  const p0=tr.map(REF.outerContour[0][0],REF.outerContour[0][1]);ctx.lineTo(p0.x,p0.y);ctx.stroke();
  const colors={X:"#35e5ff",Y:"#ffd84a",Z:"#56e36d"};
  for(const c of REF.circles){
    const p=tr.map(c.cx,c.cy),r=c.rimR*tr.scale;
    ctx.strokeStyle=colors[c.slot];ctx.lineWidth=Math.max(3,w/220);ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle="rgba(0,0,0,.78)";ctx.fillRect(p.x-23,p.y-13,46,26);ctx.fillStyle="#fff";ctx.font=`bold ${Math.max(12,w/48)}px Arial`;ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(c.slot,p.x,p.y);
  }
}

function cropSlotsFromSilhouette(found){
  sourceOverlayCanvas.width=sourceCanvas.width;sourceOverlayCanvas.height=sourceCanvas.height;
  const octx=sourceOverlayCanvas.getContext("2d");octx.drawImage(sourceCanvas,0,0);
  const colors={X:"#35e5ff",Y:"#ffd84a",Z:"#56e36d"};
  for(const c of REF.circles){
    const p=found.transform.map(c.cx,c.cy);
    const side=c.cropSide*found.transform.scale;
    const crop=document.getElementById(`crop${c.slot}`);crop.width=128;crop.height=128;
    crop.getContext("2d").drawImage(sourceCanvas,p.x-side/2,p.y-side/2,side,side,0,0,128,128);
    octx.strokeStyle=colors[c.slot];octx.lineWidth=Math.max(3,sourceCanvas.width/220);octx.strokeRect(p.x-side/2,p.y-side/2,side,side);
    octx.fillStyle="rgba(0,0,0,.78)";octx.fillRect(p.x-22,p.y-14,44,26);octx.fillStyle="#fff";octx.font=`bold ${Math.max(12,sourceCanvas.width/50)}px Arial`;octx.textAlign="center";octx.textBaseline="middle";octx.fillText(c.slot,p.x,p.y);
  }
}

// Flood-fill BLACK components from the cleaned binary mask. The three emoji windows
// should appear as large enclosed black components surrounded by the pale white plaques.
function findDarkComponents(){
  const w=cleanedCanvas.width,h=cleanedCanvas.height;
  const data=cleanedCanvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,w,h).data;
  const black=new Uint8Array(w*h),seen=new Uint8Array(w*h);
  for(let i=0;i<w*h;i++) black[i]=data[i*4]<128?1:0;
  const comps=[];
  const qx=new Int32Array(w*h),qy=new Int32Array(w*h);
  const stepSeed=2;
  for(let sy=2;sy<h-2;sy+=stepSeed){
    for(let sx=2;sx<w-2;sx+=stepSeed){
      const si=sy*w+sx;if(!black[si]||seen[si])continue;
      let head=0,tail=0;qx[tail]=sx;qy[tail]=sy;tail++;seen[si]=1;
      let area=0,minx=sx,maxx=sx,miny=sy,maxy=sy,touches=false,sumx=0,sumy=0;
      while(head<tail){
        const x=qx[head],y=qy[head];head++;area++;sumx+=x;sumy+=y;
        if(x<2||x>w-3||y<2||y>h-3)touches=true;
        if(x<minx)minx=x;if(x>maxx)maxx=x;if(y<miny)miny=y;if(y>maxy)maxy=y;
        const ns=[[x+1,y],[x-1,y],[x,y+1],[x,y-1]];
        for(const [nx,ny] of ns){
          if(nx<0||nx>=w||ny<0||ny>=h)continue;
          const ni=ny*w+nx;if(black[ni]&&!seen[ni]){seen[ni]=1;qx[tail]=nx;qy[tail]=ny;tail++;}
        }
      }
      const bw=maxx-minx+1,bh=maxy-miny+1,bbox=bw*bh,fill=area/Math.max(1,bbox),aspect=bw/Math.max(1,bh);
      const wf=bw/w,hf=bh/h,cx=sumx/area,cy=sumy/area;
      if(touches)continue;
      if(wf<.045||wf>.23||hf<.08||hf>.38)continue;
      if(aspect<.62||aspect>1.42)continue;
      if(fill<.34||fill>.98)continue;
      if(cx<w*.10||cx>w*.90||cy<h*.08||cy>h*.78)continue;
      comps.push({cx,cy,bw,bh,area,fill,aspect,r:(bw+bh)/4});
    }
  }
  return comps;
}
function chooseTriplet(comps,w,h){
  let best=null;
  for(let i=0;i<comps.length;i++)for(let j=i+1;j<comps.length;j++)for(let k=j+1;k<comps.length;k++){
    const a=[comps[i],comps[j],comps[k]].sort((p,q)=>p.cx-q.cx);
    const [x,y,z]=a;
    const rMin=Math.min(x.r,y.r,z.r),rMax=Math.max(x.r,y.r,z.r);
    // v0.13.23 HARD SIZE GATE: a tiny speck can never become the third inventory window.
    // The three physical windows are uniform, so reject any triplet whose largest radius
    // is more than 1.38x the smallest before any soft scoring happens.
    if(rMin<=0 || rMax/rMin>1.38)continue;
    const rMean=(x.r+y.r+z.r)/3;
    const rSpread=(rMax-rMin)/Math.max(1,rMean);
    const ySpread=(Math.max(x.cy,y.cy,z.cy)-Math.min(x.cy,y.cy,z.cy))/Math.max(1,rMean*2);
    const d1=y.cx-x.cx,d2=z.cx-y.cx,dMean=(d1+d2)/2;
    if(d1<rMean*.90||d2<rMean*.90||d1>rMean*2.75||d2>rMean*2.75)continue;
    const spacingRatio=Math.max(d1,d2)/Math.max(1,Math.min(d1,d2));
    if(spacingRatio>1.42)continue;
    const spacingBalance=1-clamp(Math.abs(d1-d2)/Math.max(1,dMean),0,1);
    const sizeScore=1-clamp(rSpread/.30,0,1);
    const alignScore=1-clamp(ySpread/.62,0,1);
    const center=(x.cx+z.cx)/2,centerScore=1-clamp(Math.abs(center-w/2)/(w*.44),0,1);
    const verticalScore=1-clamp(Math.abs((x.cy+y.cy+z.cy)/3-h*.39)/(h*.42),0,1);
    const score=alignScore*.31+sizeScore*.29+spacingBalance*.22+centerScore*.11+verticalScore*.07;
    if(!best||score>best.score)best={slots:{X:x,Y:y,Z:z},score,alignScore,sizeScore,spacingBalance,centerScore,rMean,d1,d2,source:"three-dark"};
  }
  return best;
}

function chooseBestPair(comps,w,h){
  let best=null;
  for(let i=0;i<comps.length;i++)for(let j=i+1;j<comps.length;j++){
    const a=[comps[i],comps[j]].sort((p,q)=>p.cx-q.cx),left=a[0],right=a[1];
    const rMin=Math.min(left.r,right.r),rMax=Math.max(left.r,right.r);
    if(rMin<=0 || rMax/rMin>1.25)continue; // two anchors must agree very tightly on physical size
    const rMean=(left.r+right.r)/2;
    const dx=right.cx-left.cx,dy=right.cy-left.cy,dist=Math.hypot(dx,dy);
    if(dist<rMean*.95 || dist>rMean*5.0)continue;
    if(Math.abs(dy)>rMean*1.05)continue;
    const unitx=dx/dist,unity=dy/dist;
    const stepRatio=dist/Math.max(1,rMean);
    const hypotheses=[];
    if(stepRatio<=2.75){
      // Adjacent pair. Try X/Y (missing Z) and Y/Z (missing X), then let
      // cluster centrality and plausible bounds decide which geometry fits the game view.
      hypotheses.push({slots:{X:left,Y:right},missing:"Z",pred:{cx:right.cx+dx,cy:right.cy+dy}});
      hypotheses.push({slots:{Y:left,Z:right},missing:"X",pred:{cx:left.cx-dx,cy:left.cy-dy}});
    }else if(stepRatio<=5.0){
      // Likely X/Z with Y hidden or unusable.
      hypotheses.push({slots:{X:left,Z:right},missing:"Y",pred:{cx:(left.cx+right.cx)/2,cy:(left.cy+right.cy)/2}});
    }
    for(const hyp of hypotheses){
      const p=hyp.pred;
      if(p.cx<w*.06||p.cx>w*.94||p.cy<h*.07||p.cy>h*.79)continue;
      const slots={...hyp.slots};
      const bw=(left.bw+right.bw)/2,bh=(left.bh+right.bh)/2;
      slots[hyp.missing]={cx:p.cx,cy:p.cy,r:rMean,bw,bh,area:Math.PI*rMean*rMean,fill:0,aspect:bw/Math.max(1,bh),source:"inferred"};
      const x=slots.X,y=slots.Y,z=slots.Z;
      const d1=Math.hypot(y.cx-x.cx,y.cy-x.cy),d2=Math.hypot(z.cx-y.cx,z.cy-y.cy);
      const spacingBalance=1-clamp(Math.abs(d1-d2)/Math.max(1,(d1+d2)/2),0,1);
      const alignSpread=(Math.max(x.cy,y.cy,z.cy)-Math.min(x.cy,y.cy,z.cy))/Math.max(1,rMean*2);
      const alignScore=1-clamp(alignSpread/.62,0,1);
      const pairSizeScore=1-clamp((rMax-rMin)/Math.max(1,rMean)/.20,0,1);
      const clusterCenter=(x.cx+z.cx)/2;
      const centerScore=1-clamp(Math.abs(clusterCenter-w/2)/(w*.40),0,1);
      const verticalScore=1-clamp(Math.abs((x.cy+y.cy+z.cy)/3-h*.39)/(h*.40),0,1);
      const score=pairSizeScore*.31+alignScore*.27+spacingBalance*.22+centerScore*.13+verticalScore*.07;
      if(!best||score>best.score)best={slots,score,alignScore,sizeScore:pairSizeScore,spacingBalance,centerScore,rMean,d1,d2,source:"two-dark-infer",missing:hyp.missing,pair:[left,right]};
    }
  }
  return best;
}

function refineInferredWithComponent(pairLock,comps){
  if(!pairLock||pairLock.source!=="two-dark-infer")return pairLock;
  const slot=pairLock.missing,pred=pairLock.slots[slot],r=pairLock.rMean;
  let best=null;
  for(const c of comps){
    if(pairLock.pair.includes(c))continue;
    const sizeRatio=Math.max(c.r,r)/Math.max(1,Math.min(c.r,r));
    if(sizeRatio>1.30)continue;
    const d=Math.hypot(c.cx-pred.cx,c.cy-pred.cy);
    if(d>r*1.05)continue;
    const score=(1-d/(r*1.05))*.65+(1-(sizeRatio-1)/.30)*.35;
    if(!best||score>best.score)best={c,score};
  }
  if(best){
    pairLock.slots[slot]={...best.c,source:"refined-dark"};
    const x=pairLock.slots.X,y=pairLock.slots.Y,z=pairLock.slots.Z;
    pairLock.d1=Math.hypot(y.cx-x.cx,y.cy-x.cy);pairLock.d2=Math.hypot(z.cx-y.cx,z.cy-y.cy);
  }
  return pairLock;
}
function drawMaskDetection(comps,triplet){
  const w=cleanedCanvas.width,h=cleanedCanvas.height;
  candidateCanvas.width=w;candidateCanvas.height=h;
  const ctx=candidateCanvas.getContext("2d");ctx.drawImage(cleanedCanvas,0,0);
  ctx.lineWidth=Math.max(2,w/260);
  ctx.strokeStyle="rgba(255,170,0,.7)";
  for(const c of comps){ctx.strokeRect(c.cx-c.bw/2,c.cy-c.bh/2,c.bw,c.bh);}
  if(!triplet)return;
  const colors={X:"#35e5ff",Y:"#ffd84a",Z:"#56e36d"};
  for(const slot of ["X","Y","Z"]){
    const c=triplet.slots[slot];ctx.strokeStyle=colors[slot];ctx.lineWidth=Math.max(3,w/210);
    ctx.beginPath();ctx.arc(c.cx,c.cy,c.r*1.08,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle="rgba(0,0,0,.78)";ctx.fillRect(c.cx-23,c.cy-13,46,26);ctx.fillStyle="#fff";ctx.font=`bold ${Math.max(12,w/48)}px Arial`;ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(slot,c.cx,c.cy);
  }
}
function cropSlots(triplet){
  sourceOverlayCanvas.width=sourceCanvas.width;sourceOverlayCanvas.height=sourceCanvas.height;
  const octx=sourceOverlayCanvas.getContext("2d");octx.drawImage(sourceCanvas,0,0);
  const colors={X:"#35e5ff",Y:"#ffd84a",Z:"#56e36d"};
  for(const slot of ["X","Y","Z"]){
    const c=triplet.slots[slot];
    // The black-hole bbox is inside the bronze rim. Expand enough to capture the emoji symbol but not the plaque body.
    const side=Math.max(c.bw,c.bh)*1.24;
    const crop=document.getElementById(`crop${slot}`);crop.width=128;crop.height=128;
    crop.getContext("2d").drawImage(sourceCanvas,c.cx-side/2,c.cy-side/2,side,side,0,0,128,128);
    octx.strokeStyle=colors[slot];octx.lineWidth=Math.max(3,sourceCanvas.width/220);octx.strokeRect(c.cx-side/2,c.cy-side/2,side,side);
    octx.fillStyle="rgba(0,0,0,.78)";octx.fillRect(c.cx-22,c.cy-14,44,26);octx.fillStyle="#fff";octx.font=`bold ${Math.max(12,sourceCanvas.width/50)}px Arial`;octx.textAlign="center";octx.textBaseline="middle";octx.fillText(slot,c.cx,c.cy);
  }
}

// v0.13.18 adaptive bright-paint classifier.
// Geometry remains frozen from v0.13.16.
// Recognition is now:
//   1) color family
//   2) adaptive bright-paint binary mask
//   3) sibling shape match (partial A / partial B / full)
// Dark same-hue regions no longer count as "filled" merely because they are orange/blue/etc.

const FAMILY_RANGES=[
  {name:"red",ids:[1,2,3],hues:[350,10]},
  {name:"orange",ids:[4,5,6],hues:[24]},
  {name:"yellow",ids:[7,8,9],hues:[48]},
  {name:"green",ids:[10,11,12],hues:[118]},
  {name:"blue",ids:[13,14,15],hues:[205,220]},
  {name:"purple",ids:[16,17,18],hues:[270,285]}
];

function hueDistance(a,b){
  let d=Math.abs(a-b)%360;
  return Math.min(d,360-d);
}
function familyHueAffinity(h,family){
  let best=999;
  for(const target of family.hues)best=Math.min(best,hueDistance(h,target));
  return clamp(1-best/62,0,1);
}

// Same visibility model on live crops and references.
// X/Y are partially hidden by the plaque in front.
function siblingVisibilityWeight(slot,nx,ny){
  const dx=nx-.5,dy=ny-.5,r=Math.hypot(dx,dy);
  if(r>.405)return 0;
  let w=r<.365?1:clamp((.405-r)/.04,0,1);

  if(slot==="X"){
    if(nx>.66)return 0;
    if(nx>.56)w*=clamp((.66-nx)/.10,0,1);
  }else if(slot==="Y"){
    if(nx>.76)return 0;
    if(nx>.65)w*=clamp((.76-nx)/.11,0,1);
  }
  return w;
}

function percentile(sorted,p){
  if(!sorted.length)return 0;
  const idx=(sorted.length-1)*p;
  const lo=Math.floor(idx),hi=Math.ceil(idx),t=idx-lo;
  return sorted[lo]*(1-t)+sorted[hi]*t;
}

// Stage 1 family detector. v0.13.16/17 did this well, so the idea stays.
function familyHueDescriptor(canvas,slot){
  const size=canvas.width;
  const data=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,size,size).data;
  const hue=new Float32Array(18);

  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const nx=(x+.5)/size,ny=(y+.5)/size;
    const w=siblingVisibilityWeight(slot,nx,ny);
    if(w<=0)continue;
    const i=(y*size+x)*4;
    const [H,S,V]=rgbToHsv(data[i],data[i+1],data[i+2]);
    const weight=w*(.18+.82*S)*(.22+.78*V);
    hue[Math.min(17,Math.floor(H/20))]+=weight;
  }
  let sum=0;for(const v of hue)sum+=v;
  sum=sum||1;for(let i=0;i<hue.length;i++)hue[i]/=sum;
  return hue;
}
function histIntersection(a,b){let s=0;for(let i=0;i<a.length;i++)s+=Math.min(a[i],b[i]);return s;}

// Build an adaptive BINARY mask of the actual bright family-colored paint.
// This is the critical change: dark orange/brown, dark navy, etc. do not count as fill.
function brightPaintMask(canvas,slot,family){
  const size=canvas.width;
  const data=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,size,size).data;
  const candidates=[];

  // First pass: gather brightness of pixels that plausibly belong to the family hue.
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const nx=(x+.5)/size,ny=(y+.5)/size;
    const vw=siblingVisibilityWeight(slot,nx,ny);
    if(vw<=0)continue;
    const i=(y*size+x)*4;
    const [H,S,V]=rgbToHsv(data[i],data[i+1],data[i+2]);
    const ha=familyHueAffinity(H,family);
    if(ha>.46 && S>.16) candidates.push(V);
  }

  candidates.sort((a,b)=>a-b);
  const q20=percentile(candidates,.20);
  const q50=percentile(candidates,.50);
  const q82=percentile(candidates,.82);

  // Adaptive threshold separates the bright painted half from the dark same-hue half.
  // Floor avoids very dark photos making brown/navy pixels count as bright paint.
  const dynamic=q20+(q82-q20)*.54;
  const valueThreshold=Math.max(.40,Math.min(.72,dynamic));

  const mask=new Uint8Array(size*size);
  const weightMap=new Float32Array(size*size);
  let active=0,visible=0,sx=0,sy=0;

  const sectors=new Float32Array(8),sectorDen=new Float32Array(8);
  const quadrants=new Float32Array(4),quadDen=new Float32Array(4);
  const grid=new Float32Array(16),gridDen=new Float32Array(16);

  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const nx=(x+.5)/size,ny=(y+.5)/size;
    const vw=siblingVisibilityWeight(slot,nx,ny);
    if(vw<=0)continue;
    visible+=vw;

    const i=(y*size+x)*4;
    const [H,S,V]=rgbToHsv(data[i],data[i+1],data[i+2]);
    const ha=familyHueAffinity(H,family);

    // Bright-family-paint decision.
    const isPaint =
      ha>.48 &&
      S>.18 &&
      V>=valueThreshold &&
      (V-q20)>=Math.max(.06,(q82-q20)*.28);

    const bit=isPaint?1:0;
    mask[y*size+x]=bit;
    weightMap[y*size+x]=vw;

    if(bit){
      active+=vw;sx+=nx*vw;sy+=ny*vw;
    }

    const dx=nx-.5,dy=ny-.5;
    let ang=Math.atan2(dy,dx);if(ang<0)ang+=Math.PI*2;
    const si=Math.min(7,Math.floor(ang/(Math.PI*2)*8));
    const qi=(ny>=.5?2:0)+(nx>=.5?1:0);
    const gx=Math.min(3,Math.floor(nx*4)),gy=Math.min(3,Math.floor(ny*4)),gi=gy*4+gx;

    sectors[si]+=bit*vw;sectorDen[si]+=vw;
    quadrants[qi]+=bit*vw;quadDen[qi]+=vw;
    grid[gi]+=bit*vw;gridDen[gi]+=vw;
  }

  for(let i=0;i<8;i++)sectors[i]/=sectorDen[i]||1;
  for(let i=0;i<4;i++)quadrants[i]/=quadDen[i]||1;
  for(let i=0;i<16;i++)grid[i]/=gridDen[i]||1;

  return {
    mask,weightMap,size,
    coverage:active/(visible||1),
    cx:active?sx/active:.5,
    cy:active?sy/active:.5,
    sectors:[...sectors],
    quadrants:[...quadrants],
    grid:[...grid],
    valueThreshold,
    q20,q50,q82
  };
}

function weightedIoU(a,b){
  let inter=0,union=0;
  for(let i=0;i<a.mask.length;i++){
    const w=Math.min(a.weightMap[i]||0,b.weightMap[i]||0);
    if(w<=0)continue;
    const av=a.mask[i],bv=b.mask[i];
    if(av||bv)union+=w;
    if(av&&bv)inter+=w;
  }
  return union?inter/union:0;
}
function weightedDice(a,b){
  let inter=0,aa=0,bb=0;
  for(let i=0;i<a.mask.length;i++){
    const w=Math.min(a.weightMap[i]||0,b.weightMap[i]||0);
    if(w<=0)continue;
    if(a.mask[i])aa+=w;
    if(b.mask[i])bb+=w;
    if(a.mask[i]&&b.mask[i])inter+=w;
  }
  return (aa+bb)?2*inter/(aa+bb):0;
}

function structuralMaskScore(live,ref){
  const iou=weightedIoU(live,ref);
  const dice=weightedDice(live,ref);
  const coverageScore=1-clamp(Math.abs(live.coverage-ref.coverage)/.36,0,1);
  const centroidScore=1-clamp((Math.abs(live.cx-ref.cx)+Math.abs(live.cy-ref.cy))/.62,0,1);

  let sectorDiff=0;for(let i=0;i<8;i++)sectorDiff+=Math.abs(live.sectors[i]-ref.sectors[i]);
  sectorDiff/=8;
  let quadDiff=0;for(let i=0;i<4;i++)quadDiff+=Math.abs(live.quadrants[i]-ref.quadrants[i]);
  quadDiff/=4;
  const regionScore=1-clamp(sectorDiff*.58+quadDiff*.42,0,1);

  // Binary overlap is now the primary sibling evidence.
  return {
    score:iou*.35+dice*.30+coverageScore*.15+centroidScore*.10+regionScore*.10,
    iou,dice,coverageScore,centroidScore,regionScore
  };
}

function siblingLabel(id){
  const pos=(id-1)%3;
  return pos===0?"partial A":pos===1?"partial B":"full";
}

function renderBrightMask(debugCanvas,paint){
  debugCanvas.width=paint.size;debugCanvas.height=paint.size;
  const ctx=debugCanvas.getContext("2d");
  const im=ctx.createImageData(paint.size,paint.size);
  for(let i=0;i<paint.mask.length;i++){
    const v=paint.mask[i]?255:0;
    im.data[i*4]=im.data[i*4+1]=im.data[i*4+2]=v;
    im.data[i*4+3]=255;
  }
  ctx.putImageData(im,0,0);
}

async function ensureEmojiReferences(){
  if(emojiReferences)return emojiReferences;
  const out={X:[],Y:[],Z:[]};

  for(const family of FAMILY_RANGES){
    for(const id of family.ids){
      const img=await loadImage(`images/emoji${id}.png?v=01318`);
      const sw=img.naturalWidth||img.width,sh=img.naturalHeight||img.height;
      const side=Math.min(sw,sh)*.76;

      for(const slot of ["X","Y","Z"]){
        const c=document.createElement("canvas");
        c.width=128;c.height=128;
        c.getContext("2d",{willReadFrequently:true})
          .drawImage(img,(sw-side)/2,(sh-side)/2,side,side,0,0,128,128);

        out[slot].push({
          id,
          family:family.name,
          hue:familyHueDescriptor(c,slot),
          paint:brightPaintMask(c,slot,family)
        });
      }
    }
  }

  emojiReferences=out;
  return out;
}

async function runRecognition(){
  const refs=await ensureEmojiReferences(),readings={};

  for(const slot of ["X","Y","Z"]){
    const src=document.getElementById(`crop${slot}`);
    const c=document.createElement("canvas");
    c.width=128;c.height=128;
    c.getContext("2d",{willReadFrequently:true}).drawImage(src,0,0,128,128);

    // Stage 1: color family.
    const liveHue=familyHueDescriptor(c,slot);
    const familyScores=FAMILY_RANGES.map(f=>{
      const members=refs[slot].filter(r=>r.family===f.name);
      return {f,score:Math.max(...members.map(r=>histIntersection(liveHue,r.hue)))};
    }).sort((a,b)=>b.score-a.score);

    const family=familyScores[0].f;
    const familyMargin=Math.max(0,familyScores[0].score-(familyScores[1]?.score||0));

    // Stage 2: adaptive BRIGHT paint mask.
    const livePaint=brightPaintMask(c,slot,family);
    const debugCanvas=document.getElementById(`paint${slot}`);
    if(debugCanvas)renderBrightMask(debugCanvas,livePaint);

    // Stage 3: compare only three same-family sibling masks.
    const ranked=refs[slot]
      .filter(r=>r.family===family.name)
      .map(r=>{
        const s=structuralMaskScore(livePaint,r.paint);
        return {
          id:r.id,
          label:siblingLabel(r.id),
          ...s,
          refCoverage:r.paint.coverage
        };
      })
      .sort((a,b)=>b.score-a.score);

    // Hard anti-"full" gate:
    // Full cannot win unless live bright-paint coverage is reasonably close
    // to the full reference coverage. This directly addresses the E6/E15 bias.
    const full=ranked.find(r=>r.label==="full");
    if(full){
      const fullRef=refs[slot].find(r=>r.id===full.id);
      const ratio=livePaint.coverage/Math.max(.001,fullRef.paint.coverage);
      if(ratio<.72){
        full.score*=.72;
        full.fullGate="rejected-low-bright-coverage";
        ranked.sort((a,b)=>b.score-a.score);
      }
    }

    const best=ranked[0],second=ranked[1];
    const margin=Math.max(0,best.score-second.score);
    const confidence=clamp(.20+familyMargin*1.1+margin*3.8+(best.score-.45)*.72,0,1);

    readings[slot]={
      id:best.id,
      family:family.name,
      label:best.label,
      confidence,
      margin,
      coverage:livePaint.coverage,
      threshold:livePaint.valueThreshold,
      top3:ranked.slice(0,3)
    };

    const el=document.getElementById(`read${slot}`);
    el.className=`reading ${(confidence>=.66&&margin>=.025)?"high":"low"}`;

    const topText=ranked.slice(0,3)
      .map(q=>`E${q.id} ${q.label} ${(q.score*100).toFixed(0)}%`)
      .join(" · ");

    el.innerHTML=
      `E${best.id} · ${family.name} · <strong>${best.label}</strong> · ${pct(confidence)}`+
      `<br><span style="font-weight:600">bright fill ${(livePaint.coverage*100).toFixed(0)}% · V≥${livePaint.valueThreshold.toFixed(2)} · Δ ${margin.toFixed(3)}<br>${topText}</span>`;
  }

  recognitionSummary.textContent=
    `Adaptive bright-paint recognition: X=E${readings.X.id} (${readings.X.label}), `+
    `Y=E${readings.Y.id} (${readings.Y.label}), `+
    `Z=E${readings.Z.id} (${readings.Z.label}). Dark same-hue pixels are excluded from fill.`;

  return readings;
}

async function analyze(){
  if(!lastImage)return;reanalyzeButton.disabled=true;resultGrid.hidden=false;recognitionSummary.textContent="";
  setStatus("Building the B&W plaque mask, then trying dark-window X/Y/Z geometry first…","working");
  try{
    if(!cvReady&&window.ppaiCvReady&&typeof window.ppaiCvReady.then==="function"){await window.ppaiCvReady;markCvReady();}
    if(!cvReady)throw new Error("OpenCV is not ready.");
    if(!REF)throw new Error("Whole-cluster silhouette reference is not loaded.");

    drawImageContained(lastImage,sourceCanvas,900);buildRawMask();cleanMask();
    const comps=findDarkComponents();
    const triplet=chooseTriplet(comps,cleanedCanvas.width,cleanedCanvas.height);

    // PRIMARY A: three real dark windows. Hard size/spacing gates are applied inside chooseTriplet.
    if(triplet && triplet.score>=.66){
      drawMaskDetection(comps,triplet);
      const rr=[triplet.slots.X.r,triplet.slots.Y.r,triplet.slots.Z.r];
      const sizeRatio=Math.max(...rr)/Math.max(1,Math.min(...rr));
      candidateInfo.innerHTML=`<strong>3-CIRCLE DARK-WINDOW LOCK: ${pct(triplet.score)}</strong><br>alignment ${pct(triplet.alignScore)} · size match ${pct(triplet.sizeScore)} · spacing ${pct(triplet.spacingBalance)} · radius ratio ${sizeRatio.toFixed(2)}×.`;
      if(debugEl)debugEl.textContent=`Three-window path used. Enclosed dark components: ${comps.length}. X/Y/Z spacing ${triplet.d1.toFixed(1)} / ${triplet.d2.toFixed(1)} px; average radius ${triplet.rMean.toFixed(1)} px. Hard rule: largest/smallest radius must be ≤1.38×. Silhouette fallback was not needed.`;
      cropSlots(triplet);const readings=await runRecognition();
      setStatus(`3-CIRCLE DARK-WINDOW LOCK (${pct(triplet.score)}). Recognition: X=E${readings.X.id}, Y=E${readings.Y.id}, Z=E${readings.Z.id}.`,`good`);
      return;
    }

    // PRIMARY B: two trustworthy same-size windows are enough. Infer the missing third
    // from equal-spacing geometry instead of accepting a tiny speck as a fake window.
    let pairLock=chooseBestPair(comps,cleanedCanvas.width,cleanedCanvas.height);
    pairLock=refineInferredWithComponent(pairLock,comps);
    if(pairLock && pairLock.score>=.64){
      drawMaskDetection(comps,pairLock);
      const inferredSource=pairLock.slots[pairLock.missing].source;
      candidateInfo.innerHTML=`<strong>2-CIRCLE + INFERRED ${pairLock.missing} LOCK: ${pct(pairLock.score)}</strong><br>`+
        `two anchors passed ≤1.25× radius agreement · ${pairLock.missing} ${inferredSource==="refined-dark"?"was locally refined":"was predicted from equal spacing"} · spacing ${pairLock.d1.toFixed(1)} / ${pairLock.d2.toFixed(1)} px.`;
      if(debugEl)debugEl.textContent=`Two-window geometry path used. Missing ${pairLock.missing} was ${inferredSource==="refined-dark"?"refined from a nearby same-size dark component":"inferred geometrically"}. A third candidate cannot be used unless its radius agrees within 1.30× of the trusted pair. Silhouette fallback was not needed.`;
      cropSlots(pairLock);const readings=await runRecognition();
      setStatus(`2-CIRCLE + INFERRED ${pairLock.missing} LOCK (${pct(pairLock.score)}). Recognition: X=E${readings.X.id}, Y=E${readings.Y.id}, Z=E${readings.Z.id}.`,`good`);
      return;
    }

    // LAST RESORT: whole-cluster silhouette registration only when fewer than two
    // trustworthy circle anchors can establish geometry.
    setStatus("Fewer than two trustworthy same-size circle anchors. Trying silhouette fallback…","working");
    const found=await findInventoryCluster(sourceCanvas,comps);
    const locked=silhouetteLockPass(found);
    drawSilhouetteFallback(comps,found);

    const primaryText=triplet
      ? `Three-circle candidate ${pct(triplet.score)} did not reach the 66% lock threshold; two-circle consensus also did not reach 64%.`
      : `No valid three-circle triplet; two-circle consensus also did not reach 64% from ${comps.length} usable enclosed component(s).`;
    candidateInfo.innerHTML=
      `<strong>SILHOUETTE FALLBACK: ${locked?"LOCKED":"candidate below lock threshold"} · ${pct(found.score)}</strong><br>`+
      `${primaryText}<br>`+
      `shape-edge ${pct(found.contour)} · solid-fill ${pct(found.body)} · extraction ${pct(found.envelope?.extractScore||0)} · rotation ${(found.transform.angle*180/Math.PI).toFixed(1)}°.`;

    if(debugEl)debugEl.textContent=
      `Fallback first converts the photo mask into ONE FILLED external plaque-cluster envelope, then matches the approved silhouette against that clean shape. Emoji interiors/rims/seams are ignored. Approved silhouette aspect ${REF.aspectRatio.toFixed(3)}:1; `+
      `stored X/Y/Z centers are transformed from the matched silhouette. Dark interiors are not required, so a glowing Z may still be localized. `+
      `Template score ${pct(found.score)}, contour ${pct(found.contour)}, seams ${pct(found.seams)}.`;

    if(!locked){
      if(triplet){cropSlots(triplet);setStatus(`NO GEOMETRY LOCK. Dark-window candidate was ${pct(triplet.score)} and silhouette fallback was ${pct(found.score)}.`,"bad");}
      else setStatus(`NO GEOMETRY LOCK. Three dark windows were not available and silhouette fallback was only ${pct(found.score)}.`,"bad");
      return;
    }

    cropSlotsFromSilhouette(found);
    const readings=await runRecognition();
    setStatus(`SILHOUETTE FALLBACK LOCKED (${pct(found.score)}). Predicted X/Y/Z from cluster geometry. Recognition: X=E${readings.X.id}, Y=E${readings.Y.id}, Z=E${readings.Z.id}.`,`good`);
  }catch(err){console.error(err);setStatus(`MASK / GEOMETRY TEST ERROR: ${err.message||err}`,"bad");}
  finally{reanalyzeButton.disabled=false;}
}

async function startCamera(){
  if(!navigator.mediaDevices?.getUserMedia){setStatus("This browser does not provide an in-page camera. Use Choose Existing Photo.","bad");return;}
  try{stopCamera();setStatus("Opening rear camera…","working");stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:"environment"},width:{ideal:1920},height:{ideal:1080}}});video.srcObject=stream;cameraWrap.hidden=false;video.hidden=false;await video.play();takePhotoButton.disabled=false;stopCameraButton.disabled=false;setStatus("Camera ready. Put the three inventory plaques in the short viewfinder and tap Take Photo.");}
  catch(err){setStatus(`Camera could not start: ${err.message||err}`,"bad");}
}
function stopCamera(){if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}if(video){try{video.pause();}catch(_){ }video.srcObject=null;}takePhotoButton.disabled=true;stopCameraButton.disabled=true;}
function captureCamera(){
  if(!stream||video.readyState<2){setStatus("Camera is not ready yet.","bad");return;}
  const vw=video.videoWidth,vh=video.videoHeight,boxW=Math.max(1,cameraWrap.clientWidth),boxH=Math.max(1,cameraWrap.clientHeight),coverScale=Math.max(boxW/vw,boxH/vh),visibleW=boxW/coverScale,visibleH=boxH/coverScale,sx=(vw-visibleW)/2,sy=(vh-visibleH)/2,targetW=Math.min(1600,Math.max(900,Math.round(visibleW))),targetH=Math.round(targetW*(boxH/boxW));
  const c=document.createElement("canvas");c.width=targetW;c.height=targetH;c.getContext("2d").drawImage(video,sx,sy,visibleW,visibleH,0,0,targetW,targetH);const img=new Image();img.onload=()=>{lastImage=img;stopCamera();cameraWrap.hidden=true;analyze();};img.src=c.toDataURL("image/jpeg",.95);
}
photoInput.addEventListener("change",()=>{const file=photoInput.files&&photoInput.files[0];if(!file)return;const url=URL.createObjectURL(file),img=new Image();img.onload=()=>{URL.revokeObjectURL(url);lastImage=img;stopCamera();cameraWrap.hidden=true;analyze();};img.onerror=()=>{URL.revokeObjectURL(url);setStatus("Could not read that image.","bad");};img.src=url;});
reanalyzeButton.addEventListener("click",analyze);presetSelect.addEventListener("change",()=>{if(lastImage)analyze();});startCameraButton.addEventListener("click",startCamera);takePhotoButton.addEventListener("click",captureCamera);stopCameraButton.addEventListener("click",()=>{stopCamera();cameraWrap.hidden=true;setStatus("Camera stopped.");});window.addEventListener("pagehide",stopCamera);
}());