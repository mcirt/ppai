(function(){
  "use strict";

  function initInventoryScanner(){
  const byId=id=>document.getElementById(id);
  const openButton=byId("scanInventoryButton");
  const dialog=byId("ppaiInventoryDialog");
  const video=byId("ppaiInventoryVideo");
  const progress=byId("ppaiInventoryProgress");
  const status=byId("ppaiInventoryStatus");
  const acceptButton=byId("ppaiInventoryAccept");
  const resetButton=byId("ppaiInventoryResetStability");
  const closeButton=byId("ppaiInventoryClose");
  const workCanvas=byId("ppaiInventoryWorkCanvas");
  const results={
    X:byId("ppaiInventoryResultX"),
    Y:byId("ppaiInventoryResultY"),
    Z:byId("ppaiInventoryResultZ")
  };

  // The white UI guides remain generous for aiming. Recognition itself uses
  // only the smaller inner colored disk (recognitionScale).
  const SLOT_CONFIG={
    X:{x:0.36,y:0.53,r:0.095,recognitionScale:0.70,softOcclusionStart:0.50,softOcclusionEnd:0.73},
    Y:{x:0.50,y:0.53,r:0.095,recognitionScale:0.70,softOcclusionStart:0.60,softOcclusionEnd:0.82},
    Z:{x:0.64,y:0.53,r:0.095,recognitionScale:0.70,softOcclusionStart:1.00,softOcclusionEnd:1.00}
  };

  const guideEls={
    X:byId("ppaiInventoryGuideX"),
    Y:byId("ppaiInventoryGuideY"),
    Z:byId("ppaiInventoryGuideZ")
  };

  // X/Y are mechanically spaced from Z in the real-game fan. Z is the only
  // fully exposed card, so each frame refines Z and derives X/Y from it.
  const SLOT_STEP_X=0.14;
  let anchoredSlots={
    X:{x:SLOT_CONFIG.X.x,y:SLOT_CONFIG.X.y},
    Y:{x:SLOT_CONFIG.Y.x,y:SLOT_CONFIG.Y.y},
    Z:{x:SLOT_CONFIG.Z.x,y:SLOT_CONFIG.Z.y}
  };

  const FAMILY_RANGES=[
    {name:"red",ids:[1,2,3]},
    {name:"orange",ids:[4,5,6]},
    {name:"yellow",ids:[7,8,9]},
    {name:"green",ids:[10,11,12]},
    {name:"blue",ids:[13,14,15]},
    {name:"purple",ids:[16,17,18]}
  ];

  let stream=null;
  let timer=null;
  let references=null;
  let nextStart=29;
  let processing=false;
  let frameCounter=0;
  const anchorCanvas=document.createElement("canvas");
  anchorCanvas.width=120; anchorCanvas.height=120;

  const histories={X:[],Y:[],Z:[]};
  const current={X:null,Y:null,Z:null};

  function setStatus(text){ if(status)status.textContent=text; }

  function resetStability(){
    anchoredSlots={
      X:{x:SLOT_CONFIG.X.x,y:SLOT_CONFIG.X.y},
      Y:{x:SLOT_CONFIG.Y.x,y:SLOT_CONFIG.Y.y},
      Z:{x:SLOT_CONFIG.Z.x,y:SLOT_CONFIG.Z.y}
    };
    updateGuidePositions(false);
    for(const slot of ["X","Y","Z"]){
      histories[slot]=[];
      current[slot]=null;
      renderSlot(slot,null);
    }
    acceptButton.disabled=true;
  }

  function updateProgress(){
    const state=window.ppaiGetInventoryScanState?.();
    if(!state)return false;

    if(!state.pyramidComplete){
      progress.textContent="Fill Pyramid Tiles 1–28 first";
      setStatus("The live inventory scanner starts only after Tiles 1–28 are filled.");
      return false;
    }
    if(state.complete){
      progress.textContent="Inventory complete — Tiles 29–52 filled";
      setStatus("All 52 tiles are filled.");
      acceptButton.disabled=true;
      return false;
    }

    nextStart=state.nextTile;
    const valid=[29,32,35,38,41,44,47,50];
    if(!valid.includes(nextStart)){
      progress.textContent=`Next empty tile: ${nextStart}`;
      setStatus("The next empty tile is not at an inventory 3-card boundary. Correct the board first.");
      return false;
    }

    progress.textContent=`Next capture: Tiles ${nextStart}–${nextStart+2}  (X=${nextStart}, Y=${nextStart+1}, Z=${nextStart+2})`;
    acceptButton.textContent=`Accept ${nextStart}–${nextStart+2}`;
    return true;
  }

  function stopCamera(){
    if(timer){clearInterval(timer);timer=null;}
    if(stream){
      stream.getTracks().forEach(t=>t.stop());
      stream=null;
    }
    try{video.pause();}catch(_){}
    if(video)video.srcObject=null;
    processing=false;
  }

  async function startCamera(){
    if(!updateProgress())return;
    resetStability();

    if(!navigator.mediaDevices?.getUserMedia){
      setStatus("This browser does not provide an in-page camera.");
      return;
    }

    try{
      stream=await navigator.mediaDevices.getUserMedia({
        audio:false,
        video:{
          facingMode:{ideal:"environment"},
          width:{ideal:1920},
          height:{ideal:1080}
        }
      });
      video.srcObject=stream;
      await video.play();
      setStatus("Live recognition running. Align Z approximately; Z anchors the X/Y/Z geometry automatically.");
      timer=setInterval(processFrame,220); // ~4.5 recognition checks/sec
    }catch(error){
      console.error(error);
      setStatus(`Camera could not start: ${error.message||error}`);
    }
  }

  function loadImage(url){
    return new Promise((resolve,reject)=>{
      const img=new Image();
      img.onload=()=>resolve(img);
      img.onerror=()=>reject(new Error(`Could not load ${url}`));
      img.src=url;
    });
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

  function slotPixelWeight(slot,nx,ny){
    // Use an inner circular recognition ROI. A soft outer feather reduces small
    // framing errors without letting card-body pixels dominate.
    const dx=nx-0.5,dy=ny-0.5;
    const d=Math.hypot(dx,dy);
    if(d>=0.49)return 0;
    let weight=d<=0.43?1:(0.49-d)/0.06;

    // X and Y are covered from the right. Instead of a hard vertical cutoff,
    // taper those pixels down gradually so the visible left-side shape remains
    // useful while the covering card contributes almost nothing.
    const cfg=SLOT_CONFIG[slot];
    if(nx>cfg.softOcclusionStart){
      if(nx>=cfg.softOcclusionEnd)return 0;
      const t=(cfg.softOcclusionEnd-nx)/(cfg.softOcclusionEnd-cfg.softOcclusionStart);
      weight*=Math.max(0,Math.min(1,t));
    }
    return weight;
  }

  function extractDescriptor(canvas,slot){
    const size=canvas.width;
    const data=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,size,size).data;

    const hue=new Float32Array(18);
    const sat=new Float32Array(4);
    const val=new Float32Array(4);
    const grid=16;
    const lum=new Float32Array(grid*grid);
    const edge=new Float32Array(grid*grid);
    const count=new Float32Array(grid*grid);
    let visiblePixels=0;

    // First pass: luminance/color.
    const gray=new Float32Array(size*size);
    for(let y=0;y<size;y++){
      for(let x=0;x<size;x++){
        const nx=(x+0.5)/size,ny=(y+0.5)/size;
        const maskWeight=slotPixelWeight(slot,nx,ny);
        if(maskWeight<=0)continue;

        const i=(y*size+x)*4;
        const R=data[i],G=data[i+1],B=data[i+2],A=(data[i+3]/255)*maskWeight;
        if(A<0.03)continue;

        const [H,S,V]=rgbToHsv(R,G,B);
        const weight=A*(0.20+0.80*S)*(0.30+0.70*V);
        hue[Math.min(17,Math.floor(H/20))]+=weight;
        sat[Math.min(3,Math.floor(S*4))]+=A;
        val[Math.min(3,Math.floor(V*4))]+=A;

        const L=0.299*R+0.587*G+0.114*B;
        gray[y*size+x]=L;
        const gx=Math.min(grid-1,Math.floor(x/size*grid));
        const gy=Math.min(grid-1,Math.floor(y/size*grid));
        const gi=gy*grid+gx;
        lum[gi]+=L;count[gi]+=1;
        visiblePixels++;
      }
    }

    // Sobel-ish edge magnitude, valuable for distinguishing the 3 shapes
    // within one color family.
    for(let y=1;y<size-1;y++){
      for(let x=1;x<size-1;x++){
        const nx=(x+0.5)/size,ny=(y+0.5)/size;
        const maskWeight=slotPixelWeight(slot,nx,ny);
        if(maskWeight<=0)continue;
        const gxv=gray[y*size+x+1]-gray[y*size+x-1];
        const gyv=gray[(y+1)*size+x]-gray[(y-1)*size+x];
        const mag=Math.min(255,Math.hypot(gxv,gyv));
        const gx=Math.min(grid-1,Math.floor(x/size*grid));
        const gy=Math.min(grid-1,Math.floor(y/size*grid));
        edge[gy*grid+gx]+=mag*maskWeight;
      }
    }

    function normalizeHist(a){
      const s=a.reduce((p,v)=>p+v,0)||1;
      for(let i=0;i<a.length;i++)a[i]/=s;
    }
    normalizeHist(hue);normalizeHist(sat);normalizeHist(val);

    for(let i=0;i<lum.length;i++){
      const n=count[i]||1;
      lum[i]/=n;
      edge[i]/=n;
    }

    // Normalize structure maps.
    function zNormalize(a){
      const vals=[...a];
      const mean=vals.reduce((p,v)=>p+v,0)/vals.length;
      const sd=Math.sqrt(vals.reduce((p,v)=>p+(v-mean)*(v-mean),0)/vals.length)||1;
      for(let i=0;i<a.length;i++)a[i]=(a[i]-mean)/sd;
    }
    zNormalize(lum);zNormalize(edge);

    return {hue,sat,val,lum,edge,visiblePixels};
  }

  function histIntersection(a,b){
    let s=0;for(let i=0;i<a.length;i++)s+=Math.min(a[i],b[i]);return s;
  }
  function cosine(a,b){
    let dot=0,aa=0,bb=0;
    for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
    return aa&&bb?dot/Math.sqrt(aa*bb):0;
  }

  async function ensureReferences(){
    if(references)return references;
    const out={X:[],Y:[],Z:[]};

    for(let id=1;id<=18;id++){
      const emoji=`emoji${id}`;
      const img=await loadImage(`images/${emoji}.png`);

      for(const slot of ["X","Y","Z"]){
        const c=document.createElement("canvas");
        c.width=72;c.height=72;
        const ctx=c.getContext("2d",{willReadFrequently:true});
        const sw=img.naturalWidth||img.width,sh=img.naturalHeight||img.height;
        const fullSide=Math.min(sw,sh);
        const side=fullSide*0.76;
        ctx.drawImage(img,(sw-side)/2,(sh-side)/2,side,side,0,0,72,72);
        out[slot].push({id,emoji,descriptor:extractDescriptor(c,slot)});
      }
    }

    references=out;
    return references;
  }

  function familyScore(desc,refs){
    // Average color match for the three variants in a color family.
    const scores=FAMILY_RANGES.map(f=>{
      const members=refs.filter(r=>f.ids.includes(r.id));
      let best=0;
      for(const r of members){
        const d=r.descriptor;
        const s=histIntersection(desc.hue,d.hue)*0.78 +
                histIntersection(desc.sat,d.sat)*0.12 +
                histIntersection(desc.val,d.val)*0.10;
        best=Math.max(best,s);
      }
      return {family:f,score:best};
    });
    scores.sort((a,b)=>b.score-a.score);
    return scores;
  }

  function classify(desc,slot,refs){
    if(desc.visiblePixels<360)return null;

    const families=familyScore(desc,refs);
    const family=families[0].family;
    const candidates=refs.filter(r=>family.ids.includes(r.id));

    const ranked=candidates.map(r=>{
      const d=r.descriptor;
      const color=histIntersection(desc.hue,d.hue);
      const shape=(cosine(desc.edge,d.edge)+1)/2;
      const structure=(cosine(desc.lum,d.lum)+1)/2;

      // Once the color family is selected, shape carries most of the decision.
      const score=color*0.16 + shape*0.56 + structure*0.28;
      return {emoji:r.emoji,id:r.id,score,color,shape,structure};
    }).sort((a,b)=>b.score-a.score);

    const best=ranked[0],second=ranked[1];
    const margin=Math.max(0,best.score-second.score);
    const familyMargin=Math.max(0,families[0].score-(families[1]?.score||0));

    // This number is intentionally conservative. Raw similarity alone cannot
    // produce a high confidence; separation from the runner-up is required.
    const confidence=Math.max(0,Math.min(1,
      0.20 + margin*5.0 + familyMargin*1.7 + Math.max(0,best.shape-0.60)*0.55
    ));

    return {
      emoji:best.emoji,
      id:best.id,
      confidence,
      margin,
      familyMargin,
      family:family.name,
      score:best.score,
      secondEmoji:second.emoji,
      secondScore:second.score
    };
  }

  function meanSaturationPatch(data,w,h,cx,cy,r){
    let sum=0,n=0;
    const x0=Math.max(0,Math.floor(cx-r)),x1=Math.min(w-1,Math.ceil(cx+r));
    const y0=Math.max(0,Math.floor(cy-r)),y1=Math.min(h-1,Math.ceil(cy+r));
    for(let y=y0;y<=y1;y+=2){
      for(let x=x0;x<=x1;x+=2){
        const dx=x-cx,dy=y-cy;
        if(dx*dx+dy*dy>r*r)continue;
        const i=(y*w+x)*4;
        const R=data[i],G=data[i+1],B=data[i+2];
        const max=Math.max(R,G,B),min=Math.min(R,G,B);
        const sat=max?((max-min)/max):0;
        const lum=(R+G+B)/765;
        // Emoji disks are saturated and not near-white card material.
        sum+=sat*(1-Math.max(0,lum-0.82)*4);
        n++;
      }
    }
    return n?sum/n:0;
  }

  function refineZAnchor(){
    const vw=video.videoWidth,vh=video.videoHeight;
    if(!vw||!vh)return;

    const nominalX=SLOT_CONFIG.Z.x*vw;
    const nominalY=SLOT_CONFIG.Z.y*vh;
    const guideR=SLOT_CONFIG.Z.r*vw;

    // Read one modest patch around the nominal Z guide, then search locally for
    // the most saturated circular disk. Z is fully visible, unlike X and Y.
    const patchR=guideR*1.65;
    const sx=Math.max(0,nominalX-patchR),sy=Math.max(0,nominalY-patchR);
    const sw=Math.min(vw-sx,patchR*2),sh=Math.min(vh-sy,patchR*2);

    const c=anchorCanvas,ctx=c.getContext("2d",{willReadFrequently:true});
    ctx.clearRect(0,0,c.width,c.height);
    ctx.drawImage(video,sx,sy,sw,sh,0,0,c.width,c.height);
    const img=ctx.getImageData(0,0,c.width,c.height);
    const scaleX=sw/c.width,scaleY=sh/c.height;

    let best={score:-1,x:c.width/2,y:c.height/2};
    const search=22;
    const diskR=Math.max(10,(guideR*0.60)/scaleX);

    for(let dy=-search;dy<=search;dy+=4){
      for(let dx=-search;dx<=search;dx+=4){
        const cx=c.width/2+dx,cy=c.height/2+dy;
        const score=meanSaturationPatch(img.data,c.width,c.height,cx,cy,diskR);
        if(score>best.score)best={score,x:cx,y:cy};
      }
    }

    const refinedX=sx+best.x*scaleX;
    const refinedY=sy+best.y*scaleY;
    const zNormX=refinedX/vw,zNormY=refinedY/vh;

    // Limit movement so a random colorful object cannot hijack the anchor.
    const maxDx=0.045,maxDy=0.040;
    const clampedZ={
      x:Math.max(SLOT_CONFIG.Z.x-maxDx,Math.min(SLOT_CONFIG.Z.x+maxDx,zNormX)),
      y:Math.max(SLOT_CONFIG.Z.y-maxDy,Math.min(SLOT_CONFIG.Z.y+maxDy,zNormY))
    };

    // Smooth over time to avoid jitter in all three derived ROIs.
    const alpha=0.34;
    anchoredSlots.Z.x=anchoredSlots.Z.x*(1-alpha)+clampedZ.x*alpha;
    anchoredSlots.Z.y=anchoredSlots.Z.y*(1-alpha)+clampedZ.y*alpha;
    anchoredSlots.Y.x=anchoredSlots.Z.x-SLOT_STEP_X;
    anchoredSlots.Y.y=anchoredSlots.Z.y;
    anchoredSlots.X.x=anchoredSlots.Z.x-SLOT_STEP_X*2;
    anchoredSlots.X.y=anchoredSlots.Z.y;

    updateGuidePositions(best.score>0.18);
  }

  function updateGuidePositions(anchorLocked){
    for(const slot of ["X","Y","Z"]){
      const el=guideEls[slot];
      if(!el)continue;
      el.style.left=`${anchoredSlots[slot].x*100}%`;
      el.style.top=`${anchoredSlots[slot].y*100}%`;
      el.classList.toggle("anchor-locked",slot==="Z"&&anchorLocked);
    }
  }

  function cropSlot(slot){
    const cfg=SLOT_CONFIG[slot];
    const vw=video.videoWidth,vh=video.videoHeight;
    const guideR=cfg.r*vw;
    const r=guideR*cfg.recognitionScale;
    const cx=anchoredSlots[slot].x*vw;
    const cy=anchoredSlots[slot].y*vh;

    const c=workCanvas;
    c.width=72;c.height=72;
    const ctx=c.getContext("2d",{willReadFrequently:true});
    ctx.clearRect(0,0,72,72);
    ctx.drawImage(video,cx-r,cy-r,r*2,r*2,0,0,72,72);
    return c;
  }

  function updateHistory(slot,prediction){
    const h=histories[slot];
    h.push(prediction||null);
    if(h.length>8)h.shift();

    const last4=h.slice(-4);
    const sameFour=last4.length===4 &&
      last4.every(Boolean) &&
      last4.every(p=>p.emoji===last4[0].emoji);

    // Temporal repetition alone is not enough. Every contributing frame must
    // clearly beat the second-best shape and the second-best color family.
    const marginFloor=slot==="X"?0.030:slot==="Y"?0.038:0.045;
    const strongEnough=sameFour &&
      last4.every(p=>p.margin>=marginFloor) &&
      last4.every(p=>p.familyMargin>=0.020) &&
      last4.every(p=>p.confidence>=0.60);

    const avgMargin=last4.filter(Boolean).length
      ? last4.filter(Boolean).reduce((s,p)=>s+p.margin,0)/last4.filter(Boolean).length
      : 0;

    current[slot]=prediction?{...prediction,stable:strongEnough,avgMargin}:null;
  }

  function renderSlot(slot,p){
    const el=results[slot];
    if(!el)return;

    if(!p){
      el.className="inventory-result";
      el.innerHTML=`<div class="slot">${slot}</div><div class="guess">Align tile</div><div class="status">waiting</div>`;
      return;
    }

    el.className="inventory-result "+(p.stable?"stable":"uncertain");
    el.innerHTML=
      `<div class="slot">${slot}</div>`+
      `<img src="images/${p.emoji}.png" alt="${p.emoji}">`+
      `<div class="guess">${p.emoji.replace("emoji","E")} · ${p.family}</div>`+
      `<div class="status">${p.stable?"STABLE":`checking · ${Math.round(p.confidence*100)}%`}</div>`+
      `<div class="margin">runner-up ${p.secondEmoji.replace("emoji","E")} · Δ ${p.margin.toFixed(3)}</div>`;
  }

  async function processFrame(){
    if(processing || !stream || video.readyState<2)return;
    processing=true;

    try{
      const refs=await ensureReferences();
      refineZAnchor();
      for(const slot of ["X","Y","Z"]){
        const c=cropSlot(slot);
        const desc=extractDescriptor(c,slot);
        const p=classify(desc,slot,refs[slot]);
        updateHistory(slot,p);
        renderSlot(slot,current[slot]);
      }

      frameCounter++;
      const allStable=["X","Y","Z"].every(slot=>current[slot]?.stable);
      acceptButton.disabled=!allStable;

      if(allStable){
        setStatus(`Stable: X=${current.X.emoji}, Y=${current.Y.emoji}, Z=${current.Z.emoji}. Ready to accept Tiles ${nextStart}–${nextStart+2}.`);
      }else{
        setStatus("Live recognition running. Z is anchoring the triplet; STABLE now requires 4 matching frames plus clear runner-up separation.");
      }
    }catch(error){
      console.error(error);
      setStatus(`Recognition error: ${error.message||error}`);
    }finally{
      processing=false;
    }
  }

  function acceptTriplet(){
    if(!["X","Y","Z"].every(slot=>current[slot]?.stable))return;

    const values=[current.X.emoji,current.Y.emoji,current.Z.emoji];
    const response=window.ppaiApplyInventoryTriplet?.(values);
    if(!response?.ok){
      setStatus(response?.message||"Could not apply inventory triplet.");
      return;
    }

    if(response.complete){
      progress.textContent=`Accepted Tiles ${response.start}–${response.end}. Inventory complete.`;
      setStatus("All Tiles 29–52 have been entered.");
      resetStability();
      return;
    }

    nextStart=response.nextTile;
    progress.textContent=`Accepted ${response.start}–${response.end}. Press Draw in the real game. Next: Tiles ${nextStart}–${nextStart+2}.`;
    acceptButton.textContent=`Accept ${nextStart}–${nextStart+2}`;
    resetStability();
    setStatus("Waiting for the next three real-game inventory cards. Keep the camera open and press Draw in the game.");
  }

  async function openScanner(){
    dialog.hidden=false;
    if(!updateProgress())return;
    await startCamera();
  }

  function closeScanner(){
    stopCamera();
    dialog.hidden=true;
  }

  if (!openButton || !dialog || !video) {
    console.error("PPAI inventory scanner: required DOM elements are missing.");
    return;
  }

  // Use a direct onclick assignment as well as an exported launcher. This keeps
  // the camera launch attached to the actual user tap on mobile Safari.
  window.ppaiOpenInventoryScanner = openScanner;
  openButton.onclick = function(event){
    event.preventDefault();
    openScanner();
  };
  acceptButton?.addEventListener("click",acceptTriplet);
  resetButton?.addEventListener("click",()=>{
    resetStability();
    setStatus("Guesses reset. Hold the camera steady over X / Y / Z.");
  });
  closeButton?.addEventListener("click",closeScanner);

  document.addEventListener("visibilitychange",()=>{if(document.hidden)stopCamera();});
  window.addEventListener("pagehide",stopCamera);
  }

  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", initInventoryScanner, {once:true});
  else initInventoryScanner();
}());
