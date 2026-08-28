(function(){
  "use strict";

  const byId=id=>document.getElementById(id);
  const openButton=byId("scanInventoryButton");
  const dialog=byId("ppaiInventoryDialog");
  const video=byId("ppaiInventoryVideo");
  const photoCanvas=byId("ppaiInventoryPhotoCanvas");
  const overlayCanvas=byId("ppaiInventoryOverlayCanvas");
  const progress=byId("ppaiInventoryProgress");
  const status=byId("ppaiInventoryStatus");
  const captureButton=byId("ppaiInventoryCapture");
  const retakeButton=byId("ppaiInventoryRetake");
  const acceptButton=byId("ppaiInventoryAccept");
  const closeButton=byId("ppaiInventoryClose");
  const resultsWrap=byId("ppaiInventoryResults");
  const results={
    X:byId("ppaiInventoryResultX"),
    Y:byId("ppaiInventoryResultY"),
    Z:byId("ppaiInventoryResultZ")
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
  let references=null;
  let nextStart=29;
  let lastReading=null;
  let cvReady=false;

  function setStatus(text){ if(status)status.textContent=text; }

  function updateProgress(){
    const state=window.ppaiGetInventoryScanState?.();
    if(!state)return false;

    if(!state.pyramidComplete){
      progress.textContent="Fill Pyramid Tiles 1–28 first";
      setStatus("The inventory scanner starts only after Tiles 1–28 are filled.");
      return false;
    }
    if(state.complete){
      progress.textContent="Inventory complete — Tiles 29–52 filled";
      setStatus("All 52 tiles are filled.");
      return false;
    }

    nextStart=state.nextTile;
    const valid=[29,32,35,38,41,44,47,50];
    if(!valid.includes(nextStart)){
      progress.textContent=`Next empty tile: ${nextStart}`;
      setStatus("The next empty tile is not at a 3-tile inventory boundary. Correct the board first.");
      return false;
    }

    progress.textContent=`Next photo: Tiles ${nextStart}–${nextStart+2}  (X=${nextStart}, Y=${nextStart+1}, Z=${nextStart+2})`;
    acceptButton.textContent=`Accept ${nextStart}–${nextStart+2}`;
    return true;
  }

  function stopCamera(){
    if(stream){
      stream.getTracks().forEach(t=>t.stop());
      stream=null;
    }
    try{video.pause();}catch(_){}
    if(video)video.srcObject=null;
  }

  async function startCamera(){
    if(!updateProgress())return;
    lastReading=null;
    acceptButton.disabled=true;
    retakeButton.disabled=true;
    resultsWrap.hidden=true;
    photoCanvas.hidden=true;
    overlayCanvas.hidden=true;
    video.hidden=false;

    if(!navigator.mediaDevices?.getUserMedia){
      setStatus("This browser does not provide an in-page camera.");
      return;
    }

    try{
      setStatus("Opening rear camera…");
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
      captureButton.disabled=false;
      setStatus("Camera ready. Frame the three inventory cards and tap Capture 3 Inventory Tiles.");
    }catch(error){
      console.error(error);
      setStatus(`Camera could not start: ${error.message||error}`);
    }
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
    const dx=nx-0.5,dy=ny-0.5,d=Math.hypot(dx,dy);
    if(d>=0.49)return 0;
    let weight=d<=0.43?1:(0.49-d)/0.06;

    // X and Y are partially covered by the next card to the right.
    const masks={
      X:{start:0.54,end:0.80},
      Y:{start:0.62,end:0.86},
      Z:{start:1,end:1}
    };
    const m=masks[slot];
    if(nx>m.start){
      if(nx>=m.end)return 0;
      weight*=Math.max(0,Math.min(1,(m.end-nx)/(m.end-m.start)));
    }
    return weight;
  }

  function extractDescriptor(canvas,slot){
    const size=canvas.width;
    const data=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,size,size).data;
    const hue=new Float32Array(18),sat=new Float32Array(4),val=new Float32Array(4);
    const grid=16,lum=new Float32Array(grid*grid),edge=new Float32Array(grid*grid),count=new Float32Array(grid*grid);
    const gray=new Float32Array(size*size);
    let visiblePixels=0;

    for(let y=0;y<size;y++){
      for(let x=0;x<size;x++){
        const nx=(x+0.5)/size,ny=(y+0.5)/size;
        const mask=slotPixelWeight(slot,nx,ny);
        if(mask<=0)continue;
        const i=(y*size+x)*4;
        const A=(data[i+3]/255)*mask;
        if(A<0.03)continue;
        const R=data[i],G=data[i+1],B=data[i+2];
        const [H,S,V]=rgbToHsv(R,G,B);
        const w=A*(0.20+0.80*S)*(0.30+0.70*V);
        hue[Math.min(17,Math.floor(H/20))]+=w;
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

    for(let y=1;y<size-1;y++){
      for(let x=1;x<size-1;x++){
        const nx=(x+0.5)/size,ny=(y+0.5)/size;
        const mask=slotPixelWeight(slot,nx,ny);
        if(mask<=0)continue;
        const gxv=gray[y*size+x+1]-gray[y*size+x-1];
        const gyv=gray[(y+1)*size+x]-gray[(y-1)*size+x];
        const mag=Math.min(255,Math.hypot(gxv,gyv));
        const gx=Math.min(grid-1,Math.floor(x/size*grid));
        const gy=Math.min(grid-1,Math.floor(y/size*grid));
        edge[gy*grid+gx]+=mag*mask;
      }
    }

    function normHist(a){
      const s=a.reduce((p,v)=>p+v,0)||1;
      for(let i=0;i<a.length;i++)a[i]/=s;
    }
    normHist(hue);normHist(sat);normHist(val);

    for(let i=0;i<lum.length;i++){
      const n=count[i]||1;
      lum[i]/=n; edge[i]/=n;
    }

    function zNorm(a){
      const vals=[...a];
      const mean=vals.reduce((p,v)=>p+v,0)/vals.length;
      const sd=Math.sqrt(vals.reduce((p,v)=>p+(v-mean)*(v-mean),0)/vals.length)||1;
      for(let i=0;i<a.length;i++)a[i]=(a[i]-mean)/sd;
    }
    zNorm(lum);zNorm(edge);

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

  function loadImage(url){
    return new Promise((resolve,reject)=>{
      const img=new Image();
      img.onload=()=>resolve(img);
      img.onerror=()=>reject(new Error(`Could not load ${url}`));
      img.src=url;
    });
  }

  async function ensureReferences(){
    if(references)return references;
    const out={X:[],Y:[],Z:[]};

    for(let id=1;id<=18;id++){
      const emoji=`emoji${id}`;
      const img=await loadImage(`images/${emoji}.png`);
      const sw=img.naturalWidth||img.width,sh=img.naturalHeight||img.height;
      const fullSide=Math.min(sw,sh);
      const side=fullSide*0.76;

      for(const slot of ["X","Y","Z"]){
        const c=document.createElement("canvas");
        c.width=72;c.height=72;
        const ctx=c.getContext("2d",{willReadFrequently:true});
        ctx.drawImage(img,(sw-side)/2,(sh-side)/2,side,side,0,0,72,72);
        out[slot].push({id,emoji,descriptor:extractDescriptor(c,slot)});
      }
    }
    references=out;
    return references;
  }

  function familyScore(desc,refs){
    return FAMILY_RANGES.map(f=>{
      const members=refs.filter(r=>f.ids.includes(r.id));
      let best=0;
      for(const r of members){
        const d=r.descriptor;
        best=Math.max(best,
          histIntersection(desc.hue,d.hue)*0.78+
          histIntersection(desc.sat,d.sat)*0.12+
          histIntersection(desc.val,d.val)*0.10
        );
      }
      return {family:f,score:best};
    }).sort((a,b)=>b.score-a.score);
  }

  function classify(desc,slot,refs){
    const families=familyScore(desc,refs);
    const family=families[0].family;
    const candidates=refs.filter(r=>family.ids.includes(r.id));

    const ranked=candidates.map(r=>{
      const d=r.descriptor;
      const color=histIntersection(desc.hue,d.hue);
      const shape=(cosine(desc.edge,d.edge)+1)/2;
      const structure=(cosine(desc.lum,d.lum)+1)/2;
      return {
        emoji:r.emoji,id:r.id,
        score:color*0.16+shape*0.56+structure*0.28,
        color,shape,structure
      };
    }).sort((a,b)=>b.score-a.score);

    const best=ranked[0],second=ranked[1];
    const margin=Math.max(0,best.score-second.score);
    const familyMargin=Math.max(0,families[0].score-(families[1]?.score||0));
    const confidence=Math.max(0,Math.min(1,
      0.20+margin*5.0+familyMargin*1.7+Math.max(0,best.shape-0.60)*0.55
    ));

    return {
      emoji:best.emoji,id:best.id,family:family.name,
      confidence,margin,familyMargin,
      secondEmoji:second.emoji,secondScore:second.score
    };
  }

  function orderTriplet(circles){
    return [...circles].sort((a,b)=>a.x-b.x);
  }


  function localEdgeDensity(binaryData,w,h,cx,cy,r){
    const x0=Math.max(0,Math.floor(cx-r)),x1=Math.min(w-1,Math.ceil(cx+r));
    const y0=Math.max(0,Math.floor(cy-r)),y1=Math.min(h-1,Math.ceil(cy+r));
    let hit=0,n=0;
    for(let y=y0;y<=y1;y+=2){
      const off=y*w;
      for(let x=x0;x<=x1;x+=2){
        const dx=x-cx,dy=y-cy;
        if(dx*dx+dy*dy>r*r)continue;
        if(binaryData[off+x])hit++;
        n++;
      }
    }
    return n?hit/n:0;
  }

  function candidateStrength(c){
    return (c.circularity||0)*0.65 + Math.min(1,(c.r||1)/45)*0.35;
  }

  function inferTripletFromAnchor(anchor,candidates,w,h){
    // Real inventory fan geometry:
    // Z is the most exposed rightmost tile. Y and X sit at near-constant horizontal
    // offsets to its left and approximately the same vertical centerline.
    //
    // Search a modest spacing range because camera scale changes from photo to photo.
    const stepCandidates=[];
    const baseR=Math.max(8,anchor.r||18);

    for(let step=baseR*1.45;step<=baseR*2.35;step+=baseR*0.10){
      const inferred=[
        {x:anchor.x-step*2,y:anchor.y,r:baseR,inferred:true},
        {x:anchor.x-step,y:anchor.y,r:baseR,inferred:true},
        {x:anchor.x,y:anchor.y,r:baseR,inferred:false}
      ];

      // Reject if the inferred cluster falls substantially outside the image.
      if(inferred[0].x-baseR<0 || inferred[2].x+baseR>w)continue;

      let score=0;

      // Reward agreement with any actual contour candidates near inferred X/Y/Z.
      for(const p of inferred){
        let best=0;
        for(const c of candidates){
          const dist=Math.hypot(c.x-p.x,c.y-p.y);
          const tol=Math.max(baseR,c.r)*1.15;
          if(dist<=tol){
            const radiusAgreement=1-Math.min(1,Math.abs(c.r-baseR)/Math.max(baseR,c.r,1));
            const proximity=1-Math.min(1,dist/tol);
            best=Math.max(best,proximity*0.65+radiusAgreement*0.35);
          }
        }
        score+=best;
      }

      // Centering prior: typical user framing puts the triplet around the middle.
      const clusterCx=(inferred[0].x+inferred[2].x)/2;
      score-=Math.abs(clusterCx-w/2)/w*0.55;
      score-=Math.abs(anchor.y-h*0.46)/h*0.35;

      stepCandidates.push({circles:inferred,score});
    }

    stepCandidates.sort((a,b)=>b.score-a.score);
    return stepCandidates[0]||null;
  }

  function chooseGeometryFallback(candidates,w,h){
    if(!candidates.length)return null;

    // Prefer a strong right-side candidate as Z, because Z is normally the fully
    // visible card. If that is unavailable, fall back to the strongest candidate.
    const sorted=[...candidates].sort((a,b)=>{
      const ar=(a.x/w)*0.55 + candidateStrength(a)*0.45;
      const br=(b.x/w)*0.55 + candidateStrength(b)*0.45;
      return br-ar;
    });

    const anchorPool=sorted.slice(0,Math.min(6,sorted.length));
    let best=null;

    for(const anchor of anchorPool){
      const fit=inferTripletFromAnchor(anchor,candidates,w,h);
      if(!fit)continue;

      // Bonus when the chosen anchor really is on the right side of the image.
      const zBonus=(anchor.x/w)*0.25;
      const score=fit.score+zBonus+candidateStrength(anchor)*0.35;
      if(!best||score>best.score)best={circles:fit.circles,score,anchor};
    }

    return best;
  }

  function chooseBestTriplet(candidates,w,h){
    if(candidates.length<3)return null;

    let best=null;
    for(let i=0;i<candidates.length-2;i++){
      for(let j=i+1;j<candidates.length-1;j++){
        for(let k=j+1;k<candidates.length;k++){
          const g=orderTriplet([candidates[i],candidates[j],candidates[k]]);
          const rs=g.map(c=>c.r);
          const rMean=rs.reduce((a,b)=>a+b,0)/3;
          const rVar=rs.reduce((s,r)=>s+Math.abs(r-rMean),0)/(3*rMean+1e-6);
          if(rVar>0.28)continue;

          const yMean=g.reduce((s,c)=>s+c.y,0)/3;
          const ySpread=Math.max(...g.map(c=>Math.abs(c.y-yMean)))/(rMean+1e-6);
          if(ySpread>1.6)continue;

          const gap1=g[1].x-g[0].x;
          const gap2=g[2].x-g[1].x;
          const gapBalance=Math.abs(gap1-gap2)/Math.max(gap1,gap2,1);

          const span=g[2].x-g[0].x;
          if(span<rMean*2.1 || span>rMean*6.5)continue;

          // Prefer triplets near the image center and lower-middle area, which is
          // where a user naturally frames the inventory.
          const centerPenalty=Math.abs(((g[0].x+g[2].x)/2)-w/2)/w;
          const yPenalty=Math.abs(yMean-h*0.48)/h;

          const score=
            rVar*3.0+
            ySpread*1.4+
            gapBalance*1.8+
            centerPenalty*0.7+
            yPenalty*0.8;

          if(!best || score<best.score)best={circles:g,score};
        }
      }
    }
    return best;
  }

  function detectInventoryCircles(sourceCanvas){
    if(!window.cv)throw new Error("OpenCV is not ready.");

    const cv=window.cv;
    let src=null,small=null,gray=null,blurred=null,edges=null,closed=null,kernel=null,contours=null,hierarchy=null;
    try{
      src=cv.imread(sourceCanvas);
      const targetWidth=Math.min(720,src.cols);
      const scale=targetWidth/src.cols;
      const targetHeight=Math.max(1,Math.round(src.rows*scale));

      small=new cv.Mat();
      cv.resize(src,small,new cv.Size(targetWidth,targetHeight),0,0,cv.INTER_AREA);

      gray=new cv.Mat();
      cv.cvtColor(small,gray,cv.COLOR_RGBA2GRAY);

      blurred=new cv.Mat();
      cv.GaussianBlur(gray,blurred,new cv.Size(7,7),0);

      edges=new cv.Mat();
      cv.Canny(blurred,edges,45,135);

      closed=new cv.Mat();
      kernel=cv.getStructuringElement(cv.MORPH_RECT,new cv.Size(5,5));
      cv.morphologyEx(edges,closed,cv.MORPH_CLOSE,kernel);

      contours=new cv.MatVector();
      hierarchy=new cv.Mat();
      cv.findContours(closed,contours,hierarchy,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE);

      const minBox=targetWidth*0.055;
      const maxBox=targetWidth*0.28;
      const candidates=[];

      for(let i=0;i<contours.size();i++){
        const contour=contours.get(i);
        try{
          const rect=cv.boundingRect(contour);
          if(rect.width<minBox||rect.height<minBox||rect.width>maxBox||rect.height>maxBox)continue;

          const aspect=rect.width/Math.max(1,rect.height);
          if(aspect<0.65||aspect>1.45)continue;

          const area=Math.abs(cv.contourArea(contour));
          const perimeter=cv.arcLength(contour,true);
          if(perimeter<=0)continue;
          const circularity=4*Math.PI*area/(perimeter*perimeter);
          if(circularity<0.18)continue;

          const cx=rect.x+rect.width/2;
          const cy=rect.y+rect.height/2;

          // Ignore extreme top/bottom UI regions.
          if(cy<targetHeight*0.12||cy>targetHeight*0.82)continue;

          candidates.push({
            x:cx,y:cy,r:(rect.width+rect.height)/4,
            circularity
          });
        }finally{
          contour.delete();
        }
      }

      // Deduplicate nested contour rings around the same emblem.
      candidates.sort((a,b)=>b.circularity-a.circularity);
      const deduped=[];
      for(const c of candidates){
        const duplicate=deduped.some(d=>Math.hypot(c.x-d.x,c.y-d.y)<Math.min(c.r,d.r)*0.72);
        if(!duplicate)deduped.push(c);
      }

      let chosen=chooseBestTriplet(deduped,targetWidth,targetHeight);
      let mode="direct-3-circle";

      if(!chosen){
        // New v0.13.2 behavior:
        // do NOT fail merely because X/Y are partially occluded and only one or
        // two circles survive contour filtering. Use the fixed inventory geometry.
        chosen=chooseGeometryFallback(deduped,targetWidth,targetHeight);
        mode="geometry-fallback";
      }

      if(!chosen){
        throw new Error(`Could not establish the inventory cluster geometry (${deduped.length} circle candidates found).`);
      }

      const sx=sourceCanvas.width/targetWidth;
      const sy=sourceCanvas.height/targetHeight;
      const mapped=chosen.circles.map(c=>({
        x:c.x*sx,
        y:c.y*sy,
        r:c.r*(sx+sy)/2,
        inferred:!!c.inferred
      }));
      mapped.detectionMode=mode;
      mapped.candidateCount=deduped.length;
      return mapped;
    }finally{
      [hierarchy,contours,kernel,closed,edges,blurred,gray,small,src].forEach(m=>{
        if(m&&typeof m.delete==="function")m.delete();
      });
    }
  }

  function cropCircle(sourceCanvas,circle,slot){
    const c=document.createElement("canvas");
    c.width=72;c.height=72;
    const ctx=c.getContext("2d",{willReadFrequently:true});
    const cropR=circle.r*0.78;
    ctx.drawImage(
      sourceCanvas,
      circle.x-cropR,circle.y-cropR,cropR*2,cropR*2,
      0,0,72,72
    );
    return c;
  }

  function renderDetection(circles,readings){
    overlayCanvas.width=photoCanvas.width;
    overlayCanvas.height=photoCanvas.height;
    const ctx=overlayCanvas.getContext("2d");
    ctx.drawImage(photoCanvas,0,0);

    const labels=["X","Y","Z"];
    ctx.save();
    ctx.lineWidth=Math.max(3,overlayCanvas.width/250);
    ctx.font=`bold ${Math.max(20,overlayCanvas.width/30)}px Arial`;
    ctx.textAlign="center";
    ctx.textBaseline="middle";

    circles.forEach((c,i)=>{
      ctx.beginPath();
      ctx.arc(c.x,c.y,c.r*0.88,0,Math.PI*2);
      ctx.strokeStyle=c.inferred?"#ffd24a":"#39e37a";
      ctx.stroke();

      const badgeR=Math.max(16,overlayCanvas.width/45);
      ctx.beginPath();
      ctx.arc(c.x,c.y-c.r*1.25,badgeR,0,Math.PI*2);
      ctx.fillStyle="rgba(0,0,0,.78)";
      ctx.fill();
      ctx.fillStyle="#fff";
      ctx.fillText(labels[i],c.x,c.y-c.r*1.25);
    });
    ctx.restore();

    overlayCanvas.hidden=false;
    photoCanvas.hidden=true;
  }

  function renderResults(readings){
    resultsWrap.hidden=false;
    for(const slot of ["X","Y","Z"]){
      const r=readings[slot];
      const el=results[slot];
      const high=r.confidence>=0.72&&r.margin>=0.04;
      el.className=`inventory-result ${high?"high":"low"}`;
      el.innerHTML=
        `<div class="slot">${slot}</div>`+
        `<img src="images/${r.emoji}.png" alt="${r.emoji}">`+
        `<div class="guess">${r.emoji.replace("emoji","E")} · ${r.family}</div>`+
        `<div class="status">${Math.round(r.confidence*100)}% ${high?"":"⚠️"}</div>`+
        `<div class="runner">runner-up ${r.secondEmoji.replace("emoji","E")} · Δ ${r.margin.toFixed(3)}</div>`;
    }
  }

  async function captureAndRecognize(){
    if(!stream||video.readyState<2){
      setStatus("Camera is not ready yet.");
      return;
    }

    captureButton.disabled=true;
    acceptButton.disabled=true;
    setStatus("Captured. OpenCV is finding the three inventory circles…");

    const maxWidth=1400;
    const scale=Math.min(1,maxWidth/video.videoWidth);
    photoCanvas.width=Math.max(1,Math.round(video.videoWidth*scale));
    photoCanvas.height=Math.max(1,Math.round(video.videoHeight*scale));
    photoCanvas.getContext("2d",{alpha:false}).drawImage(video,0,0,photoCanvas.width,photoCanvas.height);

    stopCamera();
    video.hidden=true;
    photoCanvas.hidden=false;
    retakeButton.disabled=false;

    try{
      const circles=detectInventoryCircles(photoCanvas);
      const refs=await ensureReferences();
      const slots=["X","Y","Z"];
      const readings={};

      for(let i=0;i<3;i++){
        const slot=slots[i];
        const crop=cropCircle(photoCanvas,circles[i],slot);
        const desc=extractDescriptor(crop,slot);
        readings[slot]=classify(desc,slot,refs[slot]);
      }

      lastReading=readings;
      renderDetection(circles,readings);
      renderResults(readings);

      const low=slots.filter(s=>readings[s].confidence<0.72||readings[s].margin<0.04);
      const mode=circles.detectionMode||"direct-3-circle";
      const modeText=mode==="geometry-fallback"
        ? `Geometry fallback used (${circles.candidateCount||0} contour candidate(s)); X/Y/Z positions were inferred from the inventory fan.`
        : "All three inventory circles were detected directly.";
      setStatus(
        low.length
          ? `${modeText} ${low.length} tile(s) are low-confidence; review them. You may still accept if the displayed emojis are correct.`
          : `${modeText} Review X/Y/Z, then accept.`
      );

      // Human review is the gate now. Accept is always enabled after a reading.
      acceptButton.disabled=false;
    }catch(error){
      console.error(error);
      setStatus(`Inventory detection failed: ${error.message||error}. Keep the three-card cluster visible and reasonably centered, then retake.`);
      lastReading=null;
      acceptButton.disabled=true;
    }finally{
      captureButton.disabled=false;
    }
  }

  function acceptTriplet(){
    if(!lastReading)return;

    const values=[lastReading.X.emoji,lastReading.Y.emoji,lastReading.Z.emoji];
    const response=window.ppaiApplyInventoryTriplet?.(values);
    if(!response?.ok){
      setStatus(response?.message||"Could not apply inventory triplet.");
      return;
    }

    if(response.complete){
      progress.textContent=`Accepted Tiles ${response.start}–${response.end}. Inventory complete.`;
      setStatus("All Tiles 29–52 have been entered.");
      acceptButton.disabled=true;
      return;
    }

    nextStart=response.nextTile;
    progress.textContent=`Accepted ${response.start}–${response.end}. Next photo: Tiles ${nextStart}–${nextStart+2}.`;
    acceptButton.textContent=`Accept ${nextStart}–${nextStart+2}`;
    lastReading=null;
    acceptButton.disabled=true;
    resultsWrap.hidden=true;
    retakeButton.disabled=true;
    overlayCanvas.hidden=true;
    startCamera();
  }

  async function retake(){
    lastReading=null;
    acceptButton.disabled=true;
    resultsWrap.hidden=true;
    overlayCanvas.hidden=true;
    photoCanvas.hidden=true;
    await startCamera();
  }

  async function openScanner(){
    dialog.hidden=false;
    await startCamera();
  }

  function closeScanner(){
    stopCamera();
    dialog.hidden=true;
  }

  function markCvReady(){
    const cv=window.cv;
    const required=[
      "Mat","MatVector","imread","resize","cvtColor","GaussianBlur","Canny",
      "getStructuringElement","morphologyEx","findContours","boundingRect",
      "contourArea","arcLength"
    ];
    const missing=required.filter(name=>typeof cv?.[name]==="undefined");
    if(missing.length){
      cvReady=false;
      setStatus(`OpenCV loaded, but inventory detection is missing: ${missing.join(", ")}`);
      return;
    }
    cvReady=true;
  }

  window.addEventListener("ppai-opencv-ready",markCvReady);
  if(window.ppaiCvReady)window.ppaiCvReady.then(markCvReady).catch(()=>{});

  openButton?.addEventListener("click",openScanner);
  captureButton?.addEventListener("click",captureAndRecognize);
  retakeButton?.addEventListener("click",retake);
  acceptButton?.addEventListener("click",acceptTriplet);
  closeButton?.addEventListener("click",closeScanner);

  document.addEventListener("visibilitychange",()=>{if(document.hidden)stopCamera();});
  window.addEventListener("pagehide",stopCamera);
}());
