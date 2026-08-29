(function(){
"use strict";

const VERSION="0.13.18";
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

function setStatus(text,kind=""){statusEl.textContent=text;statusEl.className=kind;}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pct(v){return `${Math.round(clamp(v,0,1)*100)}%`;}

function cvHasRequiredApis(){
  const cv=window.cv;
  const required=["Mat","imread","imshow","cvtColor","getStructuringElement","morphologyEx"];
  return !!cv && required.every(name=>typeof cv[name]!=="undefined");
}
function markCvReady(){
  cvReady=cvHasRequiredApis();
  if(cvReady){
    opencvStatus.textContent=`OpenCV ready — v${VERSION} mask + dark-hole triplet detector available.`;
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
    const rMean=(x.r+y.r+z.r)/3;
    const rSpread=(Math.max(x.r,y.r,z.r)-Math.min(x.r,y.r,z.r))/Math.max(1,rMean);
    const ySpread=(Math.max(x.cy,y.cy,z.cy)-Math.min(x.cy,y.cy,z.cy))/Math.max(1,rMean*2);
    const d1=y.cx-x.cx,d2=z.cx-y.cx,dMean=(d1+d2)/2;
    if(d1<rMean*.75||d2<rMean*.75||d1>rMean*3.0||d2>rMean*3.0)continue;
    const spacingBalance=1-clamp(Math.abs(d1-d2)/Math.max(1,dMean),0,1);
    const sizeScore=1-clamp(rSpread/.48,0,1);
    const alignScore=1-clamp(ySpread/.72,0,1);
    const center=(x.cx+z.cx)/2,centerScore=1-clamp(Math.abs(center-w/2)/(w*.44),0,1);
    const verticalScore=1-clamp(Math.abs((x.cy+y.cy+z.cy)/3-h*.39)/(h*.42),0,1);
    const score=alignScore*.32+sizeScore*.24+spacingBalance*.22+centerScore*.14+verticalScore*.08;
    if(!best||score>best.score)best={slots:{X:x,Y:y,Z:z},score,alignScore,sizeScore,spacingBalance,centerScore,rMean,d1,d2};
  }
  return best;
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
  setStatus("Building the B&W plaque mask, then looking specifically for three enclosed dark emoji windows…","working");
  try{
    if(!cvReady&&window.ppaiCvReady&&typeof window.ppaiCvReady.then==="function"){await window.ppaiCvReady;markCvReady();}
    if(!cvReady)throw new Error("OpenCV is not ready.");
    drawImageContained(lastImage,sourceCanvas,900);buildRawMask();cleanMask();
    const comps=findDarkComponents();const triplet=chooseTriplet(comps,cleanedCanvas.width,cleanedCanvas.height);drawMaskDetection(comps,triplet);
    if(!triplet){candidateInfo.innerHTML=`<strong>No valid three-hole X/Y/Z geometry.</strong><br>${comps.length} enclosed dark component(s) passed basic shape filtering.`;if(debugEl)debugEl.textContent="Mask formation succeeded, but three similarly sized, nearly aligned enclosed dark windows were not found.";setStatus("NO GEOMETRY LOCK. The mask did not produce a valid three-window inventory triplet.","bad");return;}
    candidateInfo.innerHTML=`<strong>Dark-window geometry: ${pct(triplet.score)}</strong><br>alignment ${pct(triplet.alignScore)} · size match ${pct(triplet.sizeScore)} · spacing ${pct(triplet.spacingBalance)} · centrality ${pct(triplet.centerScore)}.`;
    if(debugEl)debugEl.textContent=`Enclosed dark components: ${comps.length}. X/Y/Y spacing ${triplet.d1.toFixed(1)} / ${triplet.d2.toFixed(1)} px; average dark-window radius ${triplet.rMean.toFixed(1)} px.`;
    if(triplet.score<.66){setStatus(`THREE DARK WINDOWS FOUND, but geometry is only ${pct(triplet.score)}. Crops shown for review; recognition withheld.`,`bad`);cropSlots(triplet);return;}
    cropSlots(triplet);const readings=await runRecognition();
    setStatus(`MASK + X/Y/Z GEOMETRY LOCKED (${pct(triplet.score)}). Recognition: X=E${readings.X.id}, Y=E${readings.Y.id}, Z=E${readings.Z.id}.`,`good`);
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