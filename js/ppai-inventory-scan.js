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

  const SLOT_CONFIG={
    X:{x:0.36,y:0.53,r:0.095,maskRight:0.39},
    Y:{x:0.50,y:0.53,r:0.095,maskRight:0.30},
    Z:{x:0.64,y:0.53,r:0.095,maskRight:0.00}
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

  const histories={X:[],Y:[],Z:[]};
  const current={X:null,Y:null,Z:null};

  function setStatus(text){ if(status)status.textContent=text; }

  function resetStability(){
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
      setStatus("Live recognition running. Align the three emoji circles with X / Y / Z.");
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

  function slotPixelVisible(slot,nx,ny){
    // Circle mask.
    const dx=nx-0.5,dy=ny-0.5;
    if(Math.hypot(dx,dy)>0.47)return false;

    // X/Y in the real game are partially covered by the card to their right.
    // Ignore the right wedge instead of treating occlusion as part of the emoji.
    const blocked=SLOT_CONFIG[slot].maskRight;
    if(blocked>0){
      const cutoff=1-blocked;
      if(nx>cutoff)return false;
    }
    return true;
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
        if(!slotPixelVisible(slot,nx,ny))continue;

        const i=(y*size+x)*4;
        const R=data[i],G=data[i+1],B=data[i+2],A=data[i+3]/255;
        if(A<0.08)continue;

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
        if(!slotPixelVisible(slot,nx,ny))continue;
        const gxv=gray[y*size+x+1]-gray[y*size+x-1];
        const gyv=gray[(y+1)*size+x]-gray[(y-1)*size+x];
        const mag=Math.min(255,Math.hypot(gxv,gyv));
        const gx=Math.min(grid-1,Math.floor(x/size*grid));
        const gy=Math.min(grid-1,Math.floor(y/size*grid));
        edge[gy*grid+gx]+=mag;
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
        const side=Math.min(sw,sh);
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
    if(desc.visiblePixels<500)return null;

    const families=familyScore(desc,refs);
    const family=families[0].family;
    const candidates=refs.filter(r=>family.ids.includes(r.id));

    const ranked=candidates.map(r=>{
      const d=r.descriptor;
      const color=histIntersection(desc.hue,d.hue);
      const shape=(cosine(desc.edge,d.edge)+1)/2;
      const structure=(cosine(desc.lum,d.lum)+1)/2;
      const score=color*0.25 + shape*0.48 + structure*0.27;
      return {emoji:r.emoji,id:r.id,score,color,shape,structure};
    }).sort((a,b)=>b.score-a.score);

    const best=ranked[0],second=ranked[1];
    const margin=Math.max(0,best.score-second.score);
    const familyMargin=Math.max(0,families[0].score-(families[1]?.score||0));

    // Confidence is driven mostly by best-vs-second separation, not raw similarity.
    const confidence=Math.max(0,Math.min(1,
      0.38 + margin*3.6 + familyMargin*1.25 + Math.max(0,best.shape-0.55)*0.5
    ));

    return {
      emoji:best.emoji,
      id:best.id,
      confidence,
      margin,
      family:family.name,
      score:best.score
    };
  }

  function cropSlot(slot){
    const cfg=SLOT_CONFIG[slot];
    const vw=video.videoWidth,vh=video.videoHeight;
    const r=cfg.r*vw;
    const cx=cfg.x*vw,cy=cfg.y*vh;

    const c=workCanvas;
    c.width=72;c.height=72;
    const ctx=c.getContext("2d",{willReadFrequently:true});
    ctx.clearRect(0,0,72,72);
    ctx.drawImage(video,cx-r,cy-r,r*2,r*2,0,0,72,72);
    return c;
  }

  function updateHistory(slot,prediction){
    const h=histories[slot];
    if(!prediction){
      h.push(null);
    }else{
      h.push(prediction);
    }
    if(h.length>6)h.shift();

    const valid=h.filter(Boolean);
    const last3=valid.slice(-3);
    const stable=last3.length===3 &&
      last3.every(p=>p.emoji===last3[0].emoji) &&
      last3.every(p=>p.confidence>=0.56);

    current[slot]=prediction?{...prediction,stable}:null;
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
      `<div class="status">${p.stable?"STABLE":`checking · ${Math.round(p.confidence*100)}%`}</div>`;
  }

  async function processFrame(){
    if(processing || !stream || video.readyState<2)return;
    processing=true;

    try{
      const refs=await ensureReferences();
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
        setStatus("Live recognition running. Hold the camera steady until X, Y and Z all show STABLE.");
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
