"""Isolated HTML player. Untrusted transcript strings only enter DOM via textContent.

For video, captions are drawn over the bottom of the picture, like broadcast subtitles, and a
fullscreen button takes the picture and its captions fullscreen together (the browser's own video
fullscreen would show the bare video). Audio-only media keeps a caption box below the controls.
"""
import base64
import html
import json

from .loudness import reference_level, size_scale
from .schema import emotion_label

STYLE = """
body{margin:0;font:16px system-ui;color:#172b4d;background:#fff}
audio{width:100%;margin:24px 0}
.stage{position:relative;margin:0 auto;max-width:100%;container-type:inline-size;
background:#111827;border-radius:12px;overflow:hidden}
.stage video{display:block;width:100%;max-height:340px;background:#111827}
.stage:fullscreen{width:100vw!important;height:100vh;border-radius:0;display:flex;align-items:center}
.stage:fullscreen video{width:100%;height:100%;max-height:none;object-fit:contain}
.overlay{position:absolute;left:50%;bottom:max(13%,62px);transform:translateX(-50%);width:max-content;max-width:88%;
display:flex;flex-direction:column;align-items:center;gap:.25em;pointer-events:none;
font-size:clamp(14px,3.2cqw,46px);line-height:1.45;text-align:center}
.overlay[hidden]{display:none}
.overlay .cue{background:rgba(0,0,0,.78);color:#fff;padding:.18em .55em .22em;border-radius:.35em}
.overlay .speaker{display:inline-block;font-size:.6em;font-weight:700;color:#fff;padding:.12em .55em;
border-radius:999px;margin-right:.45em;vertical-align:.18em;letter-spacing:.01em}
.box{min-height:110px;padding:20px;background:#eef3f8;border-radius:12px;font-size:21px;line-height:1.6;margin-top:14px}
.box .speaker{font-size:14px;font-weight:700;display:block}
.active{background:#fde68a;color:#172b4d;border-radius:3px}
.toolbar{display:flex;gap:6px;justify-content:center;margin-top:8px}
button{border:1px solid #ccd6e2;border-radius:8px;background:#fff;color:#172b4d;
padding:7px 10px;cursor:pointer;margin:5px 6px 0 0}
.toolbar button{margin:0}
button[aria-pressed="false"]{color:#64748b;text-decoration:line-through}
button:focus-visible{outline:3px solid #0f766e}
"""

SCRIPT = """
const data=PAYLOAD;
const media=document.getElementById('media'), box=document.getElementById('captions');
const stage=document.getElementById('stage'), overlay=!!stage;
const timed=(w,t)=>w.start!==null && t>=w.start && t<w.end;
let shown=null, frame=null, ccOn=true;
function render(){
 const t=media.currentTime;
 const active=ccOn?data.captions.filter(c=>t>=c.start && t<c.end):[];
 // Rebuild only when the active cue or word changes; playback calls this on every frame.
 const key=ccOn+'/'+active.map(c=>data.captions.indexOf(c)+':'+c.words.findIndex(w=>timed(w,t))).join('|');
 if(key===shown)return;
 shown=key;
 box.replaceChildren();
 box.hidden=overlay && !active.length;
 if(!active.length){if(!overlay)box.textContent=ccOn?'Captions appear here during speech.':'Captions are off.';return;}
 for(const c of active){
  const line=document.createElement('div'), label=document.createElement('span');
  line.className='cue';label.className='speaker';
  if(overlay)label.style.background=c.color;else label.style.color=c.color;
  label.textContent=c.speaker+(c.emotion?' · '+c.emotion:'');line.append(label);
  if(c.words.length){
   for(const w of c.words){
    const span=document.createElement('span');span.textContent=w.text+' ';
    if(w.scale!==1)span.style.fontSize=w.scale+'em';
    if(w.level_db!==null)span.title=w.level_db.toFixed(1)+' dBFS';
    if(timed(w,t))span.className='active';line.append(span);
   }
  }else{const span=document.createElement('span');span.textContent=c.text;line.append(span);}
  box.append(line);
 }
}
// timeupdate fires only about every 250 ms, which skips short words; follow frames while playing.
function tick(){render();frame=media.paused||media.ended?null:requestAnimationFrame(tick);}
media.addEventListener('play',()=>{if(frame===null)frame=requestAnimationFrame(tick);});
media.addEventListener('timeupdate',render);media.addEventListener('seeked',render);
const cc=document.getElementById('cc');
function setCaptions(on){ccOn=on;cc.setAttribute('aria-pressed',String(on));
 cc.textContent=on?'CC  Captions on':'CC  Captions off';render();}
cc.onclick=()=>setCaptions(!ccOn);
if(overlay){
 // Size the stage to the picture so captions sit on the video, not on letterbox bars.
 const fit=()=>{
  const ratio=media.videoWidth/media.videoHeight;
  if(document.fullscreenElement){
   // Keep captions on the picture, above the controls, when the screen letterboxes the video.
   stage.style.width='';
   const w=stage.clientWidth, h=stage.clientHeight, pic=ratio?Math.min(h,w/ratio):h;
   box.style.bottom=((h-pic)/2+Math.max(0.13*pic,62))+'px';
   return;
  }
  box.style.bottom='';
  stage.style.width=ratio?Math.min(document.body.clientWidth,340*ratio)+'px':'';
 };
 media.addEventListener('loadedmetadata',fit);window.addEventListener('resize',fit);
 document.addEventListener('fullscreenchange',fit);
 const full=()=>document.fullscreenElement?document.exitFullscreen():stage.requestFullscreen();
 document.getElementById('fs').onclick=full;
 media.addEventListener('dblclick',e=>{e.preventDefault();full();});
 document.addEventListener('keydown',e=>{
  if(e.target.tagName==='BUTTON' && e.key===' ')return;
  if(e.key==='f'||e.key==='F')full();
  if(e.key==='c'||e.key==='C')setCaptions(!ccOn);
 });
}
const seek=document.getElementById('seek');
for(const c of data.captions){
 const button=document.createElement('button');
 button.textContent=c.start.toFixed(1)+'s · '+c.speaker;
 button.onclick=()=>{media.currentTime=c.start;render();};seek.append(button);
}
render();
"""


def player_html(media: bytes, mime: str, project, emotions=False, loudness=True):
    reference = reference_level(project) if loudness else None
    payload = json.dumps({
        "captions": [
            {"start": c.start, "end": c.end, "speaker": project.speakers[c.speaker],
             "color": project.colors()[c.speaker], "text": c.text,
             "emotion": emotion_label(c.emotion) if emotions and c.emotion else None,
             "words": [{**w.__dict__, "scale": size_scale(w.level_db, reference)} for w in c.words]
             if c.alignment_valid and c.text == c.original_text else []}
            for c in project.captions
        ]
    }).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    source = f'src="data:{html.escape(mime, quote=True)};base64,{base64.b64encode(media).decode("ascii")}"'
    region = 'role="region" aria-label="Synchronized captions"'
    if mime.startswith("video/"):
        body = (f'<div id="stage" class="stage"><video id="media" controls controlsList="nofullscreen" '
                f'preload="metadata" aria-label="Recording playback" {source}></video>'
                f'<div id="captions" class="overlay" {region}></div></div>'
                '<div class="toolbar"><button id="cc" aria-pressed="true">CC  Captions on</button>'
                '<button id="fs" title="Shortcut: F">⛶  Fullscreen with captions</button></div>')
    else:
        body = (f'<audio id="media" controls preload="metadata" aria-label="Recording playback" {source}></audio>'
                f'<div id="captions" class="box" {region}></div>'
                '<div class="toolbar"><button id="cc" aria-pressed="true">CC  Captions on</button></div>')
    return ('<!doctype html><html><head><meta charset="utf-8"><style>' + STYLE + '</style></head><body>'
            + body + '<div id="seek" aria-label="Jump to caption"></div><script>'
            + SCRIPT.replace("PAYLOAD", payload) + '</script></body></html>')
