import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname=path.dirname(fileURLToPath(import.meta.url));
const PAIRS={emoji1:'emoji2',emoji2:'emoji1',emoji4:'emoji5',emoji5:'emoji4',emoji7:'emoji8',emoji8:'emoji7',emoji10:'emoji11',emoji11:'emoji10',emoji13:'emoji14',emoji14:'emoji13',emoji16:'emoji17',emoji17:'emoji16'};
const UNMATCHED=new Set(['emoji3','emoji6','emoji9','emoji12','emoji15','emoji18']);
const BLOCKERS={1:[2,3],2:[4,5],3:[5,6],4:[7,8],5:[8,9],6:[9,10],7:[11,12],8:[12,13],9:[13,14],10:[14,15],11:[16,17],12:[17,18],13:[18,19],14:[19,20],15:[20,21],16:[22,23],17:[23,24],18:[24,25],19:[25,26],20:[26,27],21:[27,28]};
const COMPLETE_BONUS=1760;
class Engine{
 constructor(tileData){this.tileData={};for(let i=1;i<=52;i++)this.tileData[i]=tileData[String(i)]??tileData[i]??null;this.stockQueue=[];for(let i=29;i<=52;i++)if(this.tileData[i])this.stockQueue.push(i);this.inPlay=[];this.hold=null;this.score=0;this.streak=0;this.resets=0;this.gameOver=false;this.completed=false}
 z(){return [...this.inPlay].sort((a,b)=>b-a)[0]??null}
 pyramidPlayable(id){return (BLOCKERS[id]||[]).every(b=>this.tileData[b]===null)}
 playable(id){if(id==='h')return !!this.hold;id=Number(id);if(id>=29)return id===this.z()&&!!this.tileData[id];return !!this.tileData[id]&&this.pyramidPlayable(id)}
 emoji(id){return id==='h'?this.hold?.emoji:this.tileData[Number(id)]}
 pair(a,b){return PAIRS[this.emoji(a)]===this.emoji(b)}
 award(){this.streak++;this.score+=Math.min(this.streak,5)*50}
 remove(id){if(id==='h'){this.hold=null;return}id=Number(id);this.tileData[id]=null;this.inPlay=this.inPlay.filter(x=>x!==id);this.stockQueue=this.stockQueue.filter(x=>x!==id)}
 complete(){if(this.completed)return;this.completed=true;this.gameOver=true;this.score+=COMPLETE_BONUS}
 apply(d){const t=d.type;
  if(t==='DRAW'){const drawn=this.stockQueue.splice(0,Math.min(3,this.stockQueue.length));this.inPlay.push(...drawn);this.streak=0;return}
  if(t==='RESET'){this.stockQueue=[...this.inPlay].sort((a,b)=>a-b);this.inPlay=[];this.streak=0;this.score=Math.max(0,this.score-50);this.resets++;return}
  if(t==='HOLD'){const id=Number(d.id);if(!this.playable(id)||this.hold||UNMATCHED.has(this.emoji(id)))throw new Error(`Illegal HOLD ${id}`);const emoji=this.emoji(id);this.remove(id);this.hold={emoji,sourceId:id};if(id===1)this.complete();return}
  if(t==='CLEAR'){const id=d.id==='h'?'h':Number(d.id);if(!this.playable(id)||!UNMATCHED.has(this.emoji(id)))throw new Error(`Illegal CLEAR ${id}`);const source=id==='h'?this.hold?.sourceId:id;this.remove(id);this.award();if(Number(source)===1)this.complete();return}
  if(t==='MATCH'){const a=Number(d.a),b=Number(d.b);if(!this.playable(a)||!this.playable(b)||!this.pair(a,b))throw new Error(`Illegal MATCH ${a},${b}`);this.remove(a);this.remove(b);this.award();if(a===1||b===1)this.complete();return}
  if(t==='MATCH_HOLD'){const o=Number(d.o);if(!this.playable('h')||!this.playable(o)||!this.pair('h',o))throw new Error(`Illegal MATCH_HOLD h=${d.h},o=${o}`);const held=this.hold?.sourceId; if(Number(d.h)!==Number(held))throw new Error(`Held id mismatch expected ${held}, got ${d.h}`);this.remove('h');this.remove(o);this.award();if(Number(held)===1||o===1)this.complete();return}
  throw new Error(`Unknown action ${t}`)
 }
}
const files=fs.readdirSync(path.join(__dirname,'fixtures')).filter(f=>f.endsWith('.json')).sort();
let failed=0;
for(const f of files){const raw=JSON.parse(fs.readFileSync(path.join(__dirname,'fixtures',f),'utf8'));const init=raw.events.find(e=>e.action==='Game Initialized')?.details?.initialTileData;const eng=new Engine(init);let n=0;try{for(const e of raw.events){if(e.action!=='ACT')continue;const d=e.details;if(d.type==='FINAL_SCORE')continue;eng.apply(d);n++}const logged=raw.events.find(e=>e.action==='ACT'&&e.details?.type==='FINAL_SCORE')?.details?.finalScore ?? raw.meta?.finalScore;const ok=eng.score===logged&&eng.gameOver;console.log(`${ok?'PASS':'FAIL'} ${f}: moves=${n} score=${eng.score} logged=${logged} resets=${eng.resets} complete=${eng.gameOver}`);if(!ok)failed++}catch(err){console.error(`FAIL ${f}: ${err.message}`);failed++}}
if(failed)process.exit(1);
