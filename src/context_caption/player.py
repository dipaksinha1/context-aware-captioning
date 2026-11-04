"""Isolated HTML player. Untrusted transcript strings only enter DOM via textContent."""
import base64
import html
import json

from .loudness import reference_level, size_scale
from .schema import emotion_label


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
    encoded = base64.b64encode(media).decode("ascii")
    tag = "video" if mime.startswith("video/") else "audio"
    return """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font:16px system-ui;color:#172b4d;background:#fff}
video{width:100%;max-height:340px;background:#111827;border-radius:12px}
audio{width:100%;margin:24px 0}
#captions{min-height:110px;padding:20px;background:#eef3f8;border-radius:12px;
font-size:21px;line-height:1.6;margin-top:14px}
.speaker{font-size:14px;font-weight:700;display:block}
.active{background:#fde68a;color:#172b4d;border-radius:3px}
button{border:1px solid #ccd6e2;border-radius:8px;background:#fff;color:#172b4d;
padding:7px 10px;cursor:pointer;margin:5px 6px 0 0}
button:focus-visible{outline:3px solid #0f766e}
</style></head><body>""" + (
        f'<{tag} id="media" controls preload="metadata" aria-label="Recording playback" '
        f'src="data:{html.escape(mime, quote=True)};base64,{encoded}"></{tag}>'
    ) + """<div id="captions" role="region" aria-label="Synchronized captions"></div>
<div id="seek" aria-label="Jump to caption"></div><script>
const data=PAYLOAD;
const media=document.getElementById('media'), box=document.getElementById('captions');
const timed=(w,t)=>w.start!==null && t>=w.start && t<w.end;
let shown=null, frame=null;
function render(){
 const t=media.currentTime;
 const active=data.captions.filter(c=>t>=c.start && t<c.end);
 // Rebuild only when the active cue or word changes; playback calls this on every frame.
 const key=active.map(c=>data.captions.indexOf(c)+':'+c.words.findIndex(w=>timed(w,t))).join('|');
 if(key===shown)return;
 shown=key;
 box.replaceChildren();
 if(!active.length){box.textContent='Captions appear here during speech.';return;}
 for(const c of active){
  const line=document.createElement('div'), label=document.createElement('span');
  label.className='speaker';label.style.color=c.color;
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
const seek=document.getElementById('seek');
for(const c of data.captions){
 const button=document.createElement('button');
 button.textContent=c.start.toFixed(1)+'s · '+c.speaker;
 button.onclick=()=>{media.currentTime=c.start;render();};seek.append(button);
}
render();
</script></body></html>""".replace("PAYLOAD", payload)
