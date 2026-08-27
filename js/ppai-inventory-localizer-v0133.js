(function(){
"use strict";

const REF=window.PPAI_INVENTORY_REFERENCE_V0133;
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
let lastImage=null;
let templateEdgePoints=null;

function setStatus(text,kind=""){
  statusEl.textContent=text;
  statusEl.className=kind;
}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function pct(v){return `${Math.round(clamp(v,0,1)*100)}%`;}

window.addEventListener("ppai-opencv-ready",()=>{
  opencvStatus.textContent="OpenCV ready — v0.13.3 silhouette localizer available.";
  opencvStatus.style.color="#238636";
});
window.addEventListener("ppai-opencv-error",e=>{
  opencvStatus.textContent=`OpenCV error: ${e.detail?.message||"failed to initialize"}`;
  opencvStatus.style.color="#b42318";
});

function loadImage(url){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>resolve(img);
    img.onerror=()=>reject(new Error(`Could not load ${url}`));
    img.src=url;
  });
}

function drawImageContained(img,canvas,maxWidth=720){
  const sw=img.naturalWidth||img.width, sh=img.naturalHeight||img.height;
  const scale=Math.min(1,maxWidth/sw);
  canvas.width=Math.max(1,Math.round(sw*scale));
  canvas.height=Math.max(1,Math.round(sh*scale));
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
}

function makeLocalizationCanvas(){
  const maxW=360;
  const scale=Math.min(1,maxW/sourceCanvas.width);
  const c=document.createElement("canvas");
  c.width=Math.max(1,Math.round(sourceCanvas.width*scale));
  c.height=Math.max(1,Math.round(sourceCanvas.height*scale));
  c.getContext("2d",{willReadFrequently:true}).drawImage(sourceCanvas,0,0,c.width,c.height);
  return c;
}

function cannyData(canvas){
  const cv=window.cv;
  const src=cv.imread(canvas);
  const gray=new cv.Mat(), blur=new cv.Mat(), edges=new cv.Mat();
  try{
    cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray,blur,new cv.Size(5,5),0,0,cv.BORDER_DEFAULT);
    cv.Canny(blur,edges,60,140);
    return {width:edges.cols,height:edges.rows,data:new Uint8Array(edges.data)};
  } finally {
    src.delete();gray.delete();blur.delete();edges.delete();
  }
}

function dilateBinary(edge,w,h,radius=2){
  const out=new Uint8Array(w*h);
  for(let y=0;y<h;y++){
    for(let x=0;x<w;x++){
      if(!edge[y*w+x])continue;
      for(let dy=-radius;dy<=radius;dy++){
        const yy=y+dy;if(yy<0||yy>=h)continue;
        const row=yy*w;
        for(let dx=-radius;dx<=radius;dx++){
          const xx=x+dx;if(xx>=0&&xx<w)out[row+xx]=1;
        }
      }
    }
  }
  return out;
}

async function buildTemplateEdgePoints(){
  if(templateEdgePoints)return templateEdgePoints;
  const [img,maskImg]=await Promise.all([
    loadImage("images/inventory_reference_v0133.png?v=0133"),
    loadImage("images/inventory_reference_mask_v0133.png?v=0133")
  ]);

  const tc=document.createElement("canvas");
  tc.width=REF.canonicalWidth;tc.height=REF.canonicalHeight;
  tc.getContext("2d",{willReadFrequently:true}).drawImage(img,0,0,tc.width,tc.height);

  const mc=document.createElement("canvas");
  mc.width=REF.canonicalWidth;mc.height=REF.canonicalHeight;
  mc.getContext("2d",{willReadFrequently:true}).drawImage(maskImg,0,0,mc.width,mc.height);

  const edge=cannyData(tc);
  const maskData=mc.getContext("2d",{willReadFrequently:true}).getImageData(0,0,mc.width,mc.height).data;
  const pts=[];
  for(let y=1;y<edge.height-1;y++){
    for(let x=1;x<edge.width-1;x++){
      const i=y*edge.width+x;
      if(edge.data[i] && maskData[i*4]>127)pts.push([x-REF.templateCenter[0],y-REF.templateCenter[1]]);
    }
  }

  // Deterministic even sampling: enough structural detail without making mobile search slow.
  const target=520;
  const sampled=[];
  if(pts.length<=target)sampled.push(...pts);
  else{
    for(let i=0;i<target;i++){
      sampled.push(pts[Math.floor(i*(pts.length-1)/(target-1))]);
    }
  }
  templateEdgePoints=sampled;
  return sampled;
}

function transformedOffsets(points,scale,angleRad){
  const c=Math.cos(angleRad),s=Math.sin(angleRad);
  const out=new Array(points.length);
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(let i=0;i<points.length;i++){
    const x=points[i][0],y=points[i][1];
    const dx=Math.round((x*c-y*s)*scale);
    const dy=Math.round((x*s+y*c)*scale);
    out[i]=[dx,dy];
    if(dx<minX)minX=dx;if(dx>maxX)maxX=dx;
    if(dy<minY)minY=dy;if(dy>maxY)maxY=dy;
  }
  return {offsets:out,minX,maxX,minY,maxY};
}

function scoreAt(edge,w,cx,cy,offsets,bestSoFar=0){
  let hits=0;
  const n=offsets.length;
  for(let i=0;i<n;i++){
    const p=offsets[i];
    if(edge[(cy+p[1])*w+(cx+p[0])])hits++;
    // Cheap pruning late in the loop.
    if(i>260 && (hits+(n-i-1))/n < bestSoFar-0.025)return hits/n;
  }
  return hits/n;
}

function searchTemplate(edge,w,h,points){
  let best={score:-1,cx:0,cy:0,ratio:0,angle:0};
  const ratios=[];
  for(let r=0.35;r<=0.97+1e-6;r+=0.04)ratios.push(r);
  const angles=[-8,-4,0,4,8];

  for(const ratio of ratios){
    const scale=ratio*w/REF.canonicalWidth;
    for(const deg of angles){
      const tr=transformedOffsets(points,scale,deg*Math.PI/180);
      const step=8;
      for(let cy=-tr.minY;cy<h-tr.maxY;cy+=step){
        for(let cx=-tr.minX;cx<w-tr.maxX;cx+=step){
          const score=scoreAt(edge,w,cx,cy,tr.offsets,best.score);
          if(score>best.score)best={score,cx,cy,ratio,angle:deg};
        }
      }
    }
  }

  // Fine search around the best coarse candidate.
  let fine={...best};
  for(let ratio=Math.max(0.30,best.ratio-0.05);ratio<=Math.min(1.05,best.ratio+0.05)+1e-6;ratio+=0.01){
    const scale=ratio*w/REF.canonicalWidth;
    for(let deg=best.angle-3;deg<=best.angle+3;deg+=1){
      const tr=transformedOffsets(points,scale,deg*Math.PI/180);
      for(let cy=Math.max(-tr.minY,best.cy-14);cy<=Math.min(h-tr.maxY-1,best.cy+14);cy+=2){
        for(let cx=Math.max(-tr.minX,best.cx-14);cx<=Math.min(w-tr.maxX-1,best.cx+14);cx+=2){
          const score=scoreAt(edge,w,cx,cy,tr.offsets,fine.score);
          if(score>fine.score)fine={score,cx,cy,ratio,angle:deg};
        }
      }
    }
  }
  return fine;
}

function mapCanonical(match,locW,x,y){
  const scale=match.ratio*locW/REF.canonicalWidth;
  const a=match.angle*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
  const dx=x-REF.templateCenter[0],dy=y-REF.templateCenter[1];
  return {
    x:match.cx+(dx*c-dy*s)*scale,
    y:match.cy+(dx*s+dy*c)*scale
  };
}

function rimSupport(edge,w,h,match,locW,circle){
  const scale=match.ratio*locW/REF.canonicalWidth;
  const a=match.angle*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
  let hit=0,n=0;
  for(let k=0;k<56;k++){
    const t=k/56*Math.PI*2;
    const tx=circle.cx+Math.cos(t)*circle.rimR;
    const ty=circle.cy+Math.sin(t)*circle.rimR;
    const dx=tx-REF.templateCenter[0],dy=ty-REF.templateCenter[1];
    const x=Math.round(match.cx+(dx*c-dy*s)*scale);
    const y=Math.round(match.cy+(dx*s+dy*c)*scale);
    if(x>=0&&x<w&&y>=0&&y<h){n++;if(edge[y*w+x])hit++;}
  }
  return n?hit/n:0;
}

function drawOverlay(match,locW,locH,rims){
  overlayCanvas.width=sourceCanvas.width;
  overlayCanvas.height=sourceCanvas.height;
  const ctx=overlayCanvas.getContext("2d");
  ctx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
  const sx=sourceCanvas.width/locW,sy=sourceCanvas.height/locH;

  function P(x,y){
    const p=mapCanonical(match,locW,x,y);
    return {x:p.x*sx,y:p.y*sy};
  }

  // Detected inventory outline.
  ctx.lineWidth=Math.max(3,sourceCanvas.width/240);
  ctx.strokeStyle="#ff2b2b";
  ctx.beginPath();
  REF.outline.forEach((q,i)=>{
    const p=P(q[0],q[1]);
    if(i===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);
  });
  ctx.closePath();ctx.stroke();

  // Fixed X/Y/Z crop boxes.
  REF.circles.forEach((circle,i)=>{
    const half=circle.cropSide/2;
    const corners=[
      P(circle.cx-half,circle.cy-half),
      P(circle.cx+half,circle.cy-half),
      P(circle.cx+half,circle.cy+half),
      P(circle.cx-half,circle.cy+half)
    ];
    ctx.strokeStyle="#35e5ff";
    ctx.lineWidth=Math.max(2,sourceCanvas.width/300);
    ctx.beginPath();ctx.moveTo(corners[0].x,corners[0].y);
    corners.slice(1).forEach(p=>ctx.lineTo(p.x,p.y));
    ctx.closePath();ctx.stroke();

    const center=P(circle.cx,circle.cy);
    ctx.fillStyle="rgba(0,0,0,.72)";
    ctx.fillRect(center.x-34,center.y-18,68,28);
    ctx.fillStyle="#fff";ctx.font=`bold ${Math.max(14,sourceCanvas.width/38)}px Arial`;
    ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText(`${circle.slot} ${Math.round(rims[i]*100)}%`,center.x,center.y-4);
  });
}

function buildNormalized(match,locW){
  const scaleLoc=match.ratio*locW/REF.canonicalWidth;
  const sourcePerLoc=sourceCanvas.width/locW;
  const scale=scaleLoc*sourcePerLoc;
  const cx=match.cx*sourcePerLoc;
  const cy=match.cy*(sourceCanvas.height/(sourceCanvas.width/locW));
  const a=match.angle*Math.PI/180,c=Math.cos(a),s=Math.sin(a);

  normalizedCanvas.width=REF.canonicalWidth;
  normalizedCanvas.height=REF.canonicalHeight;
  const ctx=normalizedCanvas.getContext("2d");
  ctx.save();
  ctx.clearRect(0,0,normalizedCanvas.width,normalizedCanvas.height);

  // Map source pixels back into canonical reference coordinates.
  const A=c/scale, C=s/scale, B=-s/scale, D=c/scale;
  const E=REF.templateCenter[0]-A*cx-C*cy;
  const F=REF.templateCenter[1]-B*cx-D*cy;
  ctx.setTransform(A,B,C,D,E,F);
  ctx.drawImage(sourceCanvas,0,0);
  ctx.restore();

  for(const circle of REF.circles){
    const cc=document.getElementById(`crop${circle.slot}`);
    cc.width=circle.cropSide;cc.height=circle.cropSide;
    const cctx=cc.getContext("2d");
    cctx.clearRect(0,0,cc.width,cc.height);
    cctx.drawImage(
      normalizedCanvas,
      circle.cx-circle.cropSide/2,circle.cy-circle.cropSide/2,circle.cropSide,circle.cropSide,
      0,0,cc.width,cc.height
    );
  }
}

async function analyze(){
  if(!lastImage)return;
  reanalyzeButton.disabled=true;
  metrics.hidden=true;normalizedCard.hidden=true;
  setStatus("Analyzing permanent three-plaque silhouette…","working");

  try{
    await window.ppaiCvReady;
    const points=await buildTemplateEdgePoints();
    drawImageContained(lastImage,sourceCanvas,720);
    overlayCanvas.width=sourceCanvas.width;overlayCanvas.height=sourceCanvas.height;
    stage.hidden=false;

    const locCanvas=makeLocalizationCanvas();
    const rawEdge=cannyData(locCanvas);
    const edge=dilateBinary(rawEdge.data,rawEdge.width,rawEdge.height,2);

    // Yield once so Safari paints the "working" state before the CPU search.
    await new Promise(r=>setTimeout(r,30));
    const match=searchTemplate(edge,rawEdge.width,rawEdge.height,points);
    const rims=REF.circles.map(c=>rimSupport(edge,rawEdge.width,rawEdge.height,match,rawEdge.width,c));

    const rimAverage=(rims[0]+rims[1]+rims[2])/3;
    const zSupport=rims[2];
    const confidence=clamp(
      ((match.score-0.48)/0.48)*0.78 +
      rimAverage*0.14 +
      zSupport*0.08,
      0,1
    );

    drawOverlay(match,rawEdge.width,rawEdge.height,rims);
    buildNormalized(match,rawEdge.width);

    document.getElementById("fitScore").textContent=pct(match.score);
    document.getElementById("overallScore").textContent=pct(confidence);
    document.getElementById("scaleScore").textContent=`${Math.round(match.ratio*100)}% frame width`;
    document.getElementById("rotationScore").textContent=`${match.angle.toFixed(1)}°`;
    document.getElementById("rimX").textContent=`rim support ${pct(rims[0])}`;
    document.getElementById("rimY").textContent=`rim support ${pct(rims[1])}`;
    document.getElementById("rimZ").textContent=`rim support ${pct(rims[2])}`;
    metrics.hidden=false;normalizedCard.hidden=false;

    const locked=match.score>=0.68 && zSupport>=0.42 && rimAverage>=0.34;
    if(locked){
      setStatus(
        `INVENTORY LOCKED. Template fit ${pct(match.score)}; rim confirmation X ${pct(rims[0])}, Y ${pct(rims[1])}, Z ${pct(rims[2])}. Fixed X/Y/Z crops are shown below.`,
        "good"
      );
    }else{
      setStatus(
        `Inventory candidate found, but not locked. Template fit ${pct(match.score)}; rim confirmation X ${pct(rims[0])}, Y ${pct(rims[1])}, Z ${pct(rims[2])}. Retake with the full three-plaque cluster visible.`,
        "bad"
      );
    }
  }catch(err){
    console.error(err);
    setStatus(`Localization failed: ${err.message||err}`,"bad");
  }finally{
    reanalyzeButton.disabled=false;
  }
}

photoInput.addEventListener("change",()=>{
  const file=photoInput.files&&photoInput.files[0];
  if(!file)return;
  const url=URL.createObjectURL(file);
  const img=new Image();
  img.onload=()=>{
    URL.revokeObjectURL(url);
    lastImage=img;
    analyze();
  };
  img.onerror=()=>{
    URL.revokeObjectURL(url);
    setStatus("Could not read that image.","bad");
  };
  img.src=url;
});
reanalyzeButton.addEventListener("click",analyze);
}());
