(function(){
"use strict";

const REF=window.PPAI_INVENTORY_CLUSTER_REFERENCE_V01314;
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
const debugEl=document.getElementById("clusterDebug");

let lastImage=null;
let stream=null;
let cvReady=false;
let templateData=null;
let emojiReferences=null;

function setStatus(text,kind=""){statusEl.textContent=text;statusEl.className=kind;}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pct(v){return `${Math.round(clamp(v,0,1)*100)}%`;}

function cvHasRequiredApis(){
  const cv=window.cv;
  const required=["Mat","imread","resize","cvtColor","GaussianBlur","Canny"];
  return !!cv && required.every(name=>typeof cv[name]!=="undefined");
}
function markCvReady(){
  cvReady=cvHasRequiredApis();
  if(cvReady){
    opencvStatus.textContent="OpenCV ready — v0.13.14 whole-cluster locator available.";
    opencvStatus.style.color="#238636";
  }else{
    opencvStatus.textContent="OpenCV loaded, but required basic image APIs are missing.";
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

function loadImage(url){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>resolve(img);
    img.onerror=()=>reject(new Error(`Could not load ${url}`));
    img.src=url;
  });
}

function drawImageContained(img,canvas,maxWidth=820){
  const sw=img.naturalWidth||img.width,sh=img.naturalHeight||img.height;
  const scale=Math.min(1,maxWidth/sw);
  canvas.width=Math.max(1,Math.round(sw*scale));
  canvas.height=Math.max(1,Math.round(sh*scale));
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
}

function cannyData(canvas){
  const cv=window.cv;
  const src=cv.imread(canvas),gray=new cv.Mat(),blur=new cv.Mat(),edges=new cv.Mat();
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

function edgeBands(raw,w,h){
  return {
    exact:dilateBinary(raw,w,h,1),
    tight:dilateBinary(raw,w,h,3),
    tolerance:dilateBinary(raw,w,h,6)
  };
}

function weightedEdgeSupport(points,transform,bands,w,h){
  let sum=0,n=0;
  for(const q of points){
    const p=transform.map(q[0],q[1]),x=Math.round(p.x),y=Math.round(p.y);
    if(x<0||x>=w||y<0||y>=h)continue;
    const i=y*w+x;
    // Exact alignment earns full credit; near alignment is still useful but discounted.
    const s=bands.exact[i]?1:(bands.tight[i]?0.78:(bands.tolerance[i]?0.48:0));
    sum+=s;n++;
  }
  return n?sum/n:0;
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

function plaqueLikelihoodData(canvas){
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  const im=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  const out=new Float32Array(canvas.width*canvas.height);
  for(let i=0,p=0;i<im.length;i+=4,p++){
    const r=im[i],g=im[i+1],b=im[i+2];
    const [,s,v]=rgbToHsv(r,g,b);
    // Plaques are light gray/cream: brightness is important, saturation should be modest.
    const brightness=clamp((v-0.34)/0.52,0,1);
    const lowSat=clamp((0.58-s)/0.46,0,1);
    out[p]=brightness*lowSat;
  }
  return out;
}

function sampleCanvasMask(canvas,predicate,maxPoints){
  const data=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,canvas.width,canvas.height).data;
  const pts=[];
  for(let y=3;y<canvas.height-3;y+=5)for(let x=3;x<canvas.width-3;x+=5){
    const i=(y*canvas.width+x)*4;
    if(predicate(data[i],data[i+1],data[i+2],data[i+3],x,y))pts.push([x,y]);
  }
  if(pts.length<=maxPoints)return pts;
  const out=[];
  for(let i=0;i<maxPoints;i++)out.push(pts[Math.floor(i*(pts.length-1)/(maxPoints-1))]);
  return out;
}

async function buildTemplateData(){
  if(templateData)return templateData;

  const [exactImg,toleranceImg,bodyImg]=await Promise.all([
    loadImage("images/inventory_cluster_exact_contour_v01314.png?v=01314"),
    loadImage("images/inventory_cluster_tolerance_band_v01314.png?v=01314"),
    loadImage("images/inventory_cluster_body_probes_v01314.png?v=01314")
  ]);

  function pointsFromMask(img,maxPoints,step=3){
    const c=document.createElement("canvas");
    c.width=REF.canonicalWidth;c.height=REF.canonicalHeight;
    const ctx=c.getContext("2d",{willReadFrequently:true});
    ctx.drawImage(img,0,0);
    const data=ctx.getImageData(0,0,c.width,c.height).data;
    const pts=[];
    for(let y=1;y<c.height-1;y+=step)for(let x=1;x<c.width-1;x+=step){
      if(data[(y*c.width+x)*4]>127)pts.push([x,y]);
    }
    if(pts.length<=maxPoints)return pts;
    const out=[];
    for(let i=0;i<maxPoints;i++)out.push(pts[Math.floor(i*(pts.length-1)/(maxPoints-1))]);
    return out;
  }

  const exactPoints=pointsFromMask(exactImg,520,3);
  const tolerancePoints=pointsFromMask(toleranceImg,620,4);
  const bodyPoints=pointsFromMask(bodyImg,240,5);

  // Separate exact outer contour and overlap-seam point sets so seams get their own weight.
  const contourPoints=[];
  for(let i=0;i<REF.outerContour.length;i++){
    const a=REF.outerContour[i],b=REF.outerContour[(i+1)%REF.outerContour.length];
    const dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy);
    const n=Math.max(2,Math.ceil(len/10));
    for(let k=0;k<n;k++){const t=k/n;contourPoints.push([a[0]+dx*t,a[1]+dy*t]);}
  }
  const seamPoints=[];
  for(const seam of REF.overlapSeams){
    for(let i=0;i<seam.length-1;i++){
      const a=seam[i],b=seam[i+1],dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy);
      const n=Math.max(2,Math.ceil(len/8));
      for(let k=0;k<n;k++){const t=k/n;seamPoints.push([a[0]+dx*t,a[1]+dy*t]);}
    }
  }

  // Outside probes just beyond the exact silhouette; plaque-like hits here are penalized.
  const outsidePoints=[];
  for(let y=30;y<REF.canonicalHeight-30;y+=28)for(let x=30;x<REF.canonicalWidth-30;x+=28){
    const nearLeft=x<70,nearRight=x>REF.canonicalWidth-70,nearTop=y<55,nearBottom=y>REF.canonicalHeight-55;
    if(nearLeft||nearRight||nearTop||nearBottom)outsidePoints.push([x,y]);
  }

  templateData={exactPoints,tolerancePoints,contourPoints,seamPoints,bodyPoints,outsidePoints};
  return templateData;
}

function makeTransform(scale,angle,tx,ty){
  const c=Math.cos(angle),s=Math.sin(angle);
  const cx=REF.canonicalWidth/2,cy=REF.canonicalHeight/2;
  return {
    scale,angle,tx,ty,
    map(x,y){
      const dx=x-cx,dy=y-cy;
      return {x:tx+(dx*c-dy*s)*scale,y:ty+(dx*s+dy*c)*scale};
    }
  };
}

function transformedPointHit(points,transform,data,w,h,isFloat=false){
  let sum=0,n=0;
  for(const q of points){
    const p=transform.map(q[0],q[1]);
    const x=Math.round(p.x),y=Math.round(p.y);
    if(x<0||x>=w||y<0||y>=h)continue;
    sum += isFloat ? data[y*w+x] : (data[y*w+x]?1:0);
    n++;
  }
  return n?sum/n:0;
}

function rimSupport(bands,w,h,transform){
  let total=0;
  for(const c of REF.circles){
    let hit=0,n=0;
    for(let k=0;k<44;k++){
      const a=k/44*Math.PI*2;
      const p=transform.map(c.cx+Math.cos(a)*c.rimR,c.cy+Math.sin(a)*c.rimR);
      const x=Math.round(p.x),y=Math.round(p.y);
      if(x<0||x>=w||y<0||y>=h)continue;
      const i=y*w+x;
      hit += bands.exact[i]?1:(bands.tight[i]?.78:(bands.tolerance[i]?.46:0));
      n++;
    }
    total+=n?hit/n:0;
  }
  return total/REF.circles.length;
}

function scoreTransform(transform,features,template){
  const contour=weightedEdgeSupport(template.contourPoints,transform,features.bands,features.w,features.h);
  const seams=weightedEdgeSupport(template.seamPoints,transform,features.bands,features.w,features.h);
  const body=transformedPointHit(template.bodyPoints,transform,features.plaque,features.w,features.h,true);
  const rims=rimSupport(features.bands,features.w,features.h,transform);
  const outside=transformedPointHit(template.outsidePoints,transform,features.plaque,features.w,features.h,true);

  const w=REF.weights;
  const score=
    contour*w.outerContour+
    seams*w.overlapSeams+
    body*w.bodyFill+
    rims*w.rims+
    (1-outside)*w.outsidePenalty;

  return {score,contour,seams,body,rims,outside};
}

async function findInventoryCluster(canvas){
  const template=await buildTemplateData();
  const rawEdge=cannyData(canvas);
  const bands=edgeBands(rawEdge.data,rawEdge.width,rawEdge.height);
  const plaque=plaqueLikelihoodData(canvas);
  const w=canvas.width,h=canvas.height;
  const features={bands,plaque,w,h};

  // PASS 1 — coarse search: broad scale/rotation grid, central region only.
  let best=null;
  const minClusterW=w*.34,maxClusterW=w*.94;
  const coarseScales=9;
  const coarseAngles=[-12,-8,-4,0,4,8,12].map(v=>v*Math.PI/180);
  const coarseStep=Math.max(12,Math.round(w*.045));

  for(let si=0;si<coarseScales;si++){
    const clusterW=minClusterW+(maxClusterW-minClusterW)*(si/(coarseScales-1));
    const scale=clusterW/REF.canonicalWidth;
    const halfW=REF.canonicalWidth*scale*.54,halfH=REF.canonicalHeight*scale*.59;
    for(const angle of coarseAngles){
      for(let ty=Math.max(halfH,h*.22);ty<=Math.min(h-halfH,h*.82);ty+=coarseStep){
        for(let tx=Math.max(halfW,w*.18);tx<=Math.min(w-halfW,w*.82);tx+=coarseStep){
          const tr=makeTransform(scale,angle,tx,ty),s=scoreTransform(tr,features,template);
          const centerPenalty=.025*(Math.abs(tx-w/2)/(w/2))+.020*(Math.abs(ty-h/2)/(h/2));
          const adjusted=s.score-centerPenalty;
          if(!best||adjusted>best.adjusted)best={...s,adjusted,transform:tr,pass:1};
        }
      }
    }
  }
  if(!best)throw new Error("No whole-cluster candidate was generated.");

  // PASS 2 — local refinement: much smaller changes around the best coarse candidate.
  let refined=best;
  const base=best.transform;
  const scaleFactors=[.94,.97,.985,1,1.015,1.03,1.06];
  const angleOffsets=[-4,-2,-1,0,1,2,4].map(v=>v*Math.PI/180);
  const fineStep=Math.max(3,Math.round(w*.008));
  for(const sf of scaleFactors)for(const da of angleOffsets){
    for(let dy=-3*fineStep;dy<=3*fineStep;dy+=fineStep){
      for(let dx=-3*fineStep;dx<=3*fineStep;dx+=fineStep){
        const tr=makeTransform(base.scale*sf,base.angle+da,base.tx+dx,base.ty+dy);
        const s=scoreTransform(tr,features,template);
        if(s.score>refined.score)refined={...s,adjusted:s.score,transform:tr,pass:2};
      }
    }
  }

  return refined;
}

function drawOverlay(found){
  overlayCanvas.width=sourceCanvas.width;overlayCanvas.height=sourceCanvas.height;
  const ctx=overlayCanvas.getContext("2d");
  ctx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
  const tr=found.transform;

  // Tolerance band first (green, translucent).
  ctx.lineWidth=Math.max(9,sourceCanvas.width/85);
  ctx.strokeStyle="rgba(75,220,90,.42)";
  ctx.beginPath();
  REF.outerContour.forEach((q,i)=>{
    const p=tr.map(q[0],q[1]);
    if(i===0)ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y);
  });
  const p0=tr.map(REF.outerContour[0][0],REF.outerContour[0][1]);ctx.lineTo(p0.x,p0.y);ctx.stroke();

  // Exact contour (red).
  ctx.lineWidth=Math.max(2.5,sourceCanvas.width/270);
  ctx.strokeStyle="#ff3030";
  ctx.beginPath();
  REF.outerContour.forEach((q,i)=>{
    const p=tr.map(q[0],q[1]);
    if(i===0)ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y);
  });
  ctx.lineTo(p0.x,p0.y);ctx.stroke();

  // Overlap seams (yellow) — strongly weighted.
  ctx.strokeStyle="#ffd52a";
  ctx.lineWidth=Math.max(2.5,sourceCanvas.width/250);
  for(const seam of REF.overlapSeams){
    ctx.beginPath();
    seam.forEach((q,i)=>{
      const p=tr.map(q[0],q[1]);
      if(i===0)ctx.moveTo(p.x,p.y); else ctx.lineTo(p.x,p.y);
    });
    ctx.stroke();
  }

  // Rim support / fixed X/Y/Z crop anchors.
  const colors={X:"#35e5ff",Y:"#ffd84a",Z:"#56e36d"};
  for(const c of REF.circles){
    const p=tr.map(c.cx,c.cy),r=c.rimR*tr.scale;
    ctx.strokeStyle=colors[c.slot];
    ctx.lineWidth=Math.max(2.2,sourceCanvas.width/280);
    ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.stroke();
    const half=c.cropSide*tr.scale/2;
    ctx.strokeRect(p.x-half,p.y-half,half*2,half*2);
    ctx.fillStyle="rgba(0,0,0,.76)";
    ctx.fillRect(p.x-30,p.y-17,60,27);
    ctx.fillStyle="#fff";ctx.font=`bold ${Math.max(13,sourceCanvas.width/46)}px Arial`;
    ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(c.slot,p.x,p.y-3);
  }

  ctx.fillStyle="rgba(0,0,0,.80)";
  ctx.fillRect(8,8,Math.min(560,sourceCanvas.width-16),42);
  ctx.fillStyle="#fff";ctx.textAlign="left";ctx.textBaseline="middle";
  ctx.font=`bold ${Math.max(11,sourceCanvas.width/56)}px Arial`;
  ctx.fillText(
    `cluster ${pct(found.score)} · contour ${pct(found.contour)} · seams ${pct(found.seams)} · body ${pct(found.body)} · rims ${pct(found.rims)}`,
    16,29
  );
}

function buildNormalized(transform){
  normalizedCanvas.width=REF.canonicalWidth;normalizedCanvas.height=REF.canonicalHeight;
  const ctx=normalizedCanvas.getContext("2d");
  const c=Math.cos(transform.angle),s=Math.sin(transform.angle),scale=transform.scale;
  const cx=REF.canonicalWidth/2,cy=REF.canonicalHeight/2;

  // Inverse similarity transform source -> canonical.
  const A=c/scale,C=s/scale,B=-s/scale,D=c/scale;
  const E=cx-A*transform.tx-C*transform.ty;
  const F=cy-B*transform.tx-D*transform.ty;
  ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,normalizedCanvas.width,normalizedCanvas.height);
  ctx.setTransform(A,B,C,D,E,F);ctx.drawImage(sourceCanvas,0,0);
  ctx.setTransform(1,0,0,1,0,0);

  for(const circle of REF.circles){
    const cc=document.getElementById(`crop${circle.slot}`);
    cc.width=circle.cropSide;cc.height=circle.cropSide;
    const cctx=cc.getContext("2d");
    cctx.clearRect(0,0,cc.width,cc.height);
    cctx.drawImage(normalizedCanvas,
      circle.cx-circle.cropSide/2,circle.cy-circle.cropSide/2,circle.cropSide,circle.cropSide,
      0,0,cc.width,cc.height);
  }
}

// ---------------- Emoji recognition ----------------
// Deliberately two-stage: color family first, then only the 3 siblings in that family.
const FAMILY_RANGES=[
  {name:"red",ids:[1,2,3]},
  {name:"orange",ids:[4,5,6]},
  {name:"yellow",ids:[7,8,9]},
  {name:"green",ids:[10,11,12]},
  {name:"blue",ids:[13,14,15]},
  {name:"purple",ids:[16,17,18]}
];

function slotPixelWeight(slot,nx,ny){
  const d=Math.hypot(nx-.5,ny-.5);
  if(d>=.49)return 0;
  let w=d<=.43?1:(.49-d)/.06;
  const masks={X:{start:.54,end:.80},Y:{start:.62,end:.86},Z:{start:1,end:1}};
  const m=masks[slot];
  if(nx>m.start){if(nx>=m.end)return 0;w*=clamp((m.end-nx)/(m.end-m.start),0,1);}
  return w;
}
function descriptor(canvas,slot){
  const size=canvas.width,data=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,size,size).data;
  const hue=new Float32Array(18),lum=new Float32Array(16*16),edge=new Float32Array(16*16),count=new Float32Array(16*16),gray=new Float32Array(size*size);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const w=slotPixelWeight(slot,(x+.5)/size,(y+.5)/size);if(w<=0)continue;
    const i=(y*size+x)*4,R=data[i],G=data[i+1],B=data[i+2],[H,S,V]=rgbToHsv(R,G,B);
    const hw=w*(.2+.8*S)*(.3+.7*V);hue[Math.min(17,Math.floor(H/20))]+=hw;
    const L=.299*R+.587*G+.114*B;gray[y*size+x]=L;
    const gx=Math.min(15,Math.floor(x/size*16)),gy=Math.min(15,Math.floor(y/size*16)),gi=gy*16+gx;
    lum[gi]+=L;count[gi]++;
  }
  for(let y=1;y<size-1;y++)for(let x=1;x<size-1;x++){
    const w=slotPixelWeight(slot,(x+.5)/size,(y+.5)/size);if(w<=0)continue;
    const mag=Math.min(255,Math.hypot(gray[y*size+x+1]-gray[y*size+x-1],gray[(y+1)*size+x]-gray[(y-1)*size+x]));
    const gx=Math.min(15,Math.floor(x/size*16)),gy=Math.min(15,Math.floor(y/size*16));edge[gy*16+gx]+=mag*w;
  }
  let hs=0;for(const v of hue)hs+=v;hs=hs||1;for(let i=0;i<hue.length;i++)hue[i]/=hs;
  for(let i=0;i<lum.length;i++){const n=count[i]||1;lum[i]/=n;edge[i]/=n;}
  function zn(a){const v=[...a],m=v.reduce((p,q)=>p+q,0)/v.length,sd=Math.sqrt(v.reduce((p,q)=>p+(q-m)*(q-m),0)/v.length)||1;for(let i=0;i<a.length;i++)a[i]=(a[i]-m)/sd;}
  zn(lum);zn(edge);return {hue,lum,edge};
}
function histIntersection(a,b){let s=0;for(let i=0;i<a.length;i++)s+=Math.min(a[i],b[i]);return s;}
function cosine(a,b){let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}return aa&&bb?dot/Math.sqrt(aa*bb):0;}

async function ensureEmojiReferences(){
  if(emojiReferences)return emojiReferences;
  const out={X:[],Y:[],Z:[]};
  for(let id=1;id<=18;id++){
    const emoji=`emoji${id}`,img=await loadImage(`images/${emoji}.png?v=01314`);
    const sw=img.naturalWidth||img.width,sh=img.naturalHeight||img.height,side=Math.min(sw,sh)*.76;
    for(const slot of ["X","Y","Z"]){
      const c=document.createElement("canvas");c.width=96;c.height=96;
      c.getContext("2d",{willReadFrequently:true}).drawImage(img,(sw-side)/2,(sh-side)/2,side,side,0,0,96,96);
      out[slot].push({id,emoji,d:descriptor(c,slot)});
    }
  }
  emojiReferences=out;return out;
}
async function runRecognition(){
  const refs=await ensureEmojiReferences(),readings={};
  for(const slot of ["X","Y","Z"]){
    const c=document.getElementById(`crop${slot}`);
    const w=document.createElement("canvas");w.width=96;w.height=96;w.getContext("2d",{willReadFrequently:true}).drawImage(c,0,0,96,96);
    const d=descriptor(w,slot);
    const familyScores=FAMILY_RANGES.map(f=>{
      const members=refs[slot].filter(r=>f.ids.includes(r.id));
      return {f,score:Math.max(...members.map(r=>histIntersection(d.hue,r.d.hue)))};
    }).sort((a,b)=>b.score-a.score);
    const family=familyScores[0].f;
    const ranked=refs[slot].filter(r=>family.ids.includes(r.id)).map(r=>{
      const shape=(cosine(d.edge,r.d.edge)+1)/2,structure=(cosine(d.lum,r.d.lum)+1)/2,color=histIntersection(d.hue,r.d.hue);
      return {id:r.id,emoji:r.emoji,score:shape*.56+structure*.29+color*.15};
    }).sort((a,b)=>b.score-a.score);
    const margin=Math.max(0,ranked[0].score-ranked[1].score);
    const famMargin=Math.max(0,familyScores[0].score-(familyScores[1]?.score||0));
    const confidence=clamp(.18+margin*4.5+famMargin*1.5,0,1);
    const r={...ranked[0],family:family.name,confidence,margin,top3:ranked.slice(0,3)};
    readings[slot]=r;
    const el=document.getElementById(`read${slot}`);
    const top=r.top3.map(q=>`E${q.id} ${q.score.toFixed(3)}`).join(" · ");
    el.className=`reading ${(confidence>=.66&&margin>=.03)?"high":"low"}`;
    el.innerHTML=`E${r.id} · ${r.family} · ${pct(confidence)}<br><span style="font-weight:600">Δ ${margin.toFixed(3)} · ${top}</span>`;
    document.getElementById(`rim${slot}`).textContent="fixed crop from whole-cluster registration";
  }
  document.getElementById("recognitionSummary").textContent=
    `Two-stage recognition: X=E${readings.X.id}, Y=E${readings.Y.id}, Z=E${readings.Z.id}. Color family was chosen first; only 3 sibling shapes were compared second.`;
  return readings;
}

async function analyze(){
  if(!lastImage)return;
  reanalyzeButton.disabled=true;metrics.hidden=true;normalizedCard.hidden=true;
  setStatus("Searching for the complete three-plaque inventory cluster…","working");
  try{
    if(!cvReady){
      if(window.ppaiCvReady&&typeof window.ppaiCvReady.then==="function"){await window.ppaiCvReady;markCvReady();}
    }
    if(!cvReady)throw new Error("OpenCV is not ready yet.");

    drawImageContained(lastImage,sourceCanvas,820);
    overlayCanvas.width=sourceCanvas.width;overlayCanvas.height=sourceCanvas.height;stage.hidden=false;

    const found=await findInventoryCluster(sourceCanvas);
    drawOverlay(found);
    buildNormalized(found.transform);
    const readings=await runRecognition();

    document.getElementById("fitScore").textContent=pct(found.score);
    document.getElementById("overallScore").textContent=pct(found.score*.78+found.rims*.22);
    document.getElementById("scaleScore").textContent=`${found.transform.scale.toFixed(3)}×`;
    document.getElementById("rotationScore").textContent=`${(found.transform.angle*180/Math.PI).toFixed(1)}°`;
    metrics.hidden=false;normalizedCard.hidden=false;

    if(debugEl){
      debugEl.textContent=`outer contour ${pct(found.contour)} · overlap seams ${pct(found.seams)} · plaque body ${pct(found.body)} · rim support ${pct(found.rims)} · outside penalty ${pct(found.outside)}`;
    }

    const locked=found.score>=0.54 && found.contour>=0.44 && found.seams>=0.38;
    const readText=`X=E${readings.X.id}, Y=E${readings.Y.id}, Z=E${readings.Z.id}`;
    if(locked){
      setStatus(`WHOLE INVENTORY CLUSTER LOCKED. Template ${pct(found.score)}. ${readText}. Rims were confirmation only — no 3-circle prerequisite.`,"good");
    }else{
      setStatus(`Whole-cluster candidate found but below lock threshold. Template ${pct(found.score)}, contour ${pct(found.contour)}, seams ${pct(found.seams)}, body ${pct(found.body)}, rims ${pct(found.rims)}. Recognition ran for review: ${readText}.`,"bad");
    }
  }catch(err){
    console.error(err);
    setStatus(`NO CLUSTER MATCH: ${err.message||err}`,"bad");
  }finally{reanalyzeButton.disabled=false;}
}

async function startCamera(){
  if(!navigator.mediaDevices?.getUserMedia){
    setStatus("This browser does not provide an in-page camera. Use Choose Existing Photo.","bad");return;
  }
  try{
    stopCamera();
    setStatus("Opening rear camera…","working");
    stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:"environment"},width:{ideal:1920},height:{ideal:1080}}});
    video.srcObject=stream;cameraWrap.hidden=false;video.hidden=false;
    await video.play();takePhotoButton.disabled=false;stopCameraButton.disabled=false;
    setStatus("Camera ready. Put the three-plaque inventory cluster inside the short viewfinder and tap Take Photo.");
  }catch(err){setStatus(`Camera could not start: ${err.message||err}`,"bad");}
}
function stopCamera(){
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  if(video){try{video.pause();}catch(_){ }video.srcObject=null;}
  takePhotoButton.disabled=true;stopCameraButton.disabled=true;
}
function captureCamera(){
  if(!stream||video.readyState<2){setStatus("Camera is not ready yet.","bad");return;}
  // Preserve the known-good exact visible-viewfinder capture behavior.
  const vw=video.videoWidth,vh=video.videoHeight;
  const boxW=Math.max(1,cameraWrap.clientWidth),boxH=Math.max(1,cameraWrap.clientHeight);
  const coverScale=Math.max(boxW/vw,boxH/vh);
  const visibleW=boxW/coverScale,visibleH=boxH/coverScale;
  const sx=(vw-visibleW)/2,sy=(vh-visibleH)/2;
  const targetW=Math.min(1600,Math.max(900,Math.round(visibleW))),targetH=Math.round(targetW*(boxH/boxW));
  const c=document.createElement("canvas");c.width=targetW;c.height=targetH;
  c.getContext("2d").drawImage(video,sx,sy,visibleW,visibleH,0,0,targetW,targetH);
  const img=new Image();
  img.onload=()=>{lastImage=img;stopCamera();cameraWrap.hidden=true;setStatus("Captured exactly the visible viewfinder. Searching for the whole inventory cluster…","working");analyze();};
  img.src=c.toDataURL("image/jpeg",.95);
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
