"""Isolated HTML player. Untrusted transcript strings only enter DOM via textContent."""
import base64
import html
import json

from .sounds import format_sound


def player_html(media: bytes, mime: str, project, emotions=False, sounds=True):
    payload = json.dumps({
        "sounds": [{"start": s.start, "end": s.end, "text": format_sound(s)}
                   for s in project.sounds] if sounds else [],
        "captions": [
            {"start": c.start, "end": c.end, "speaker": project.speakers[c.speaker],
             "color": project.colors()[c.speaker], "text": c.text,
             "emotion": c.emotion if emotions else None,
             "words": [w.__dict__ for w in c.words] if c.alignment_valid and c.text == c.original_text else []}
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
.sound{font-size:18px;color:#475569}
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
function render(){
 const t=media.currentTime;
 box.replaceChildren();
 const active=data.captions.filter(c=>t>=c.start && t<c.end);
 const heard=data.sounds.filter(s=>t>=s.start && t<s.end);
 if(!active.length && !heard.length){
  box.textContent='Captions appear here during speech and sounds.';return;}
 for(const c of active){
  const line=document.createElement('div'), label=document.createElement('span');
  label.className='speaker';label.style.color=c.color;
  label.textContent=c.speaker+(c.emotion?' · '+c.emotion:'');line.append(label);
  if(c.words.length){
   for(const w of c.words){
    const span=document.createElement('span');span.textContent=w.text+' ';
    if(w.start!==null && t>=w.start && t<w.end)span.className='active';line.append(span);
   }
  }else{const span=document.createElement('span');span.textContent=c.text;line.append(span);}
  box.append(line);
 }
 for(const s of heard){  // below speech, so a sound starting mid-sentence does not move the words
  const line=document.createElement('div');line.className='sound';
  line.textContent=s.text;box.append(line);
 }
}
media.addEventListener('timeupdate',render);media.addEventListener('seeked',render);
const seek=document.getElementById('seek');
const cues=[...data.captions,...data.sounds].sort((a,b)=>a.start-b.start);
for(const c of cues){
 const button=document.createElement('button');
 button.textContent=c.start.toFixed(1)+'s · '+(c.speaker||c.text);
 button.onclick=()=>{media.currentTime=c.start;render();};seek.append(button);
}
render();
</script></body></html>""".replace("PAYLOAD", payload)
