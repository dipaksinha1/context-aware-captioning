// Record an end-to-end walkthrough of the app on the synthetic three-speaker test video.
//
// Requires: the app running with the ML extra and model access, e.g.
//   HF_TOKEN=... streamlit run app.py --server.port 8599 --server.fileWatcherType none
// plus Node.js Playwright with Chromium, FFmpeg, and `context-caption-eval` on PATH (or CAPTION_EVAL).
//   node scripts/record_app_demo.js [--url http://localhost:8599] [--out outputs/demo]
// Produces <out>/app_demo.mp4: browser recording, processing segment sped up, with the conversation
// audio muxed in during synchronized playback (browser recordings carry no sound).
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const REPO = path.resolve(__dirname, '..');
const URL = arg('--url', 'http://localhost:8599');
const OUT = path.resolve(arg('--out', path.join(REPO, 'outputs/demo')));
const MEDIA = path.join(REPO, 'examples/test_conversation.mp4');
const REFERENCE = path.join(REPO, 'examples/test_conversation_reference.json');
const SPEEDUP = 8;
const W = 1280, H = 800;

const banner = (page, text) => page.evaluate(t => {
  let el = document.getElementById('demo-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-banner';
    el.style.cssText = 'position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:99999;' +
      'background:rgba(15,23,42,.92);color:#fff;font:600 17px/1.35 system-ui,sans-serif;padding:12px 22px;' +
      'border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.25);max-width:1100px;text-align:center';
    document.body.appendChild(el);
  }
  el.textContent = t;
}, text);

async function moveTo(page, locator) {
  const box = await locator.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 18 });
  await page.waitForTimeout(250);
  return box;
}
async function click(page, locator) {
  await locator.scrollIntoViewIfNeeded();
  await moveTo(page, locator);
  await locator.click();
}
async function typeInto(page, locator, text) {
  await click(page, locator);
  await page.keyboard.press('Control+A');
  await page.keyboard.type(text, { delay: 70 });
}
async function scrollTo(locator) {
  await locator.evaluate(el => el.scrollIntoView({ block: 'start', behavior: 'smooth' }));
}

(async () => {
  fs.mkdirSync(path.join(OUT, 'raw'), { recursive: true });
  const evalBin = process.env.CAPTION_EVAL || 'context-caption-eval';
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: W, height: H },
    recordVideo: { dir: path.join(OUT, 'raw'), size: { width: W, height: H } },
  });
  // Visible cursor in the top-level page (headless recordings have none).
  await context.addInitScript(() => {
    if (window.top !== window) return;
    window.addEventListener('DOMContentLoaded', () => {
      const dot = document.createElement('div');
      dot.style.cssText = 'position:fixed;z-index:100000;width:18px;height:18px;margin:-9px 0 0 -9px;' +
        'border-radius:50%;background:rgba(220,38,38,.55);border:2px solid #fff;pointer-events:none;' +
        'left:-50px;top:-50px;transition:transform .1s';
      document.body.appendChild(dot);
      document.addEventListener('mousemove', e => { dot.style.left = e.clientX + 'px'; dot.style.top = e.clientY + 'px'; }, true);
      document.addEventListener('mousedown', () => { dot.style.transform = 'scale(.6)'; }, true);
      document.addEventListener('mouseup', () => { dot.style.transform = ''; }, true);
    });
  });
  const t0 = Date.now();
  const page = await context.newPage();
  const at = () => (Date.now() - t0) / 1000;
  const events = {};

  await page.goto(URL);
  await page.getByText('Every voice. In context.').waitFor({ timeout: 60000 });
  await page.waitForTimeout(500);
  await banner(page, 'Context Caption: speaker-aware captions for multi-speaker media');
  await page.waitForTimeout(3500);

  // 1. Settings.
  await banner(page, '1 · Settings: WhisperX base on CPU, 3 known speakers, optional text emotion');
  const sidebar = page.locator('[data-testid="stSidebar"]');
  const count = sidebar.getByLabel('Known speaker count (0 = detect)');
  await typeInto(page, count, '3'); await count.press('Enter');
  await page.waitForTimeout(800);
  await click(page, sidebar.getByText('Estimate English text emotion'));
  await page.waitForTimeout(1500);

  // 2. Upload and consent.
  await banner(page, '2 · Upload a recording (41 s, three synthetic voices) and confirm consent');
  for (let attempt = 0; ; attempt++) {
    await page.locator('[data-testid="stFileUploaderDropzoneInput"]').first().setInputFiles(MEDIA);
    try {
      await page.locator('[data-testid="stFileUploaderDropzone"]').first().getByText('1.8MB').waitFor({ timeout: 10000 });
      break;
    } catch (e) { if (attempt >= 3) throw e; }
  }
  await page.waitForTimeout(1200);
  const consentText = page.getByText('I have permission to process the voices in this recording.');
  const consent = page.getByRole('checkbox', { name: 'I have permission to process the voices in this recording.' });
  for (let i = 0; i < 5 && !(await consent.isChecked()); i++) { await click(page, consentText); await page.waitForTimeout(1500); }
  const generate = page.getByRole('button', { name: 'Generate captions' });
  await page.waitForFunction(() => {
    const b = [...document.querySelectorAll('button')].find(x => x.innerText.includes('Generate captions'));
    return b && !b.disabled;
  }, null, { timeout: 15000 });

  // 3. Real processing (sped up in post).
  await banner(page, `3 · Real models on CPU: transcription → word alignment → speaker diarization → emotion (shown ${SPEEDUP}× faster)`);
  await moveTo(page, generate);
  events.procStart = at();
  await generate.click();
  await page.locator('[data-testid="stMetric"]').first().waitFor({ timeout: 600000 });
  events.procEnd = at();
  const metrics = await page.locator('[data-testid="stMetricValue"]').allInnerTexts();
  await banner(page, `Done: ${metrics[0]} caption cues, ${metrics[1]} speaker labels, language ${metrics[2]}`);
  await page.waitForTimeout(3000);

  // 4. Name the voices.
  await click(page, page.getByRole('tab', { name: 'Review & edit' }));
  await banner(page, '4 · Name the voices (diarization gives anonymous labels)');
  await page.waitForTimeout(1000);
  for (const [k, v] of Object.entries({ SPEAKER_02: 'Maya', SPEAKER_00: 'Raj', SPEAKER_01: 'Alba' })) {
    await typeInto(page, page.getByLabel(k, { exact: true }), v);
    await page.waitForTimeout(300);
  }
  await click(page, page.getByRole('button', { name: 'Save speaker names' }));
  await page.getByText('Speaker names saved.').waitFor({ timeout: 15000 });
  await page.waitForTimeout(1500);

  // 5. Fix the transcription error.
  await banner(page, '5 · Correct a transcription error: “Dude,” → “Good.” (original text is kept in the project)');
  const grid = page.locator('[data-testid="stDataFrame"]').first();
  await scrollTo(grid);
  await page.waitForTimeout(2000);
  const rows = JSON.parse(fs.readFileSync(path.join(REPO, 'examples/test_conversation_output.json'))).captions;
  const idx = rows.findIndex(c => c.text === 'Dude,');
  // The canvas grid mirrors each cell in a hidden element (glide-cell-<col>-<row>) that takes focus;
  // move one row at a time and confirm focus arrived, since key presses can be dropped during redraws.
  const focused = () => page.evaluate(() => document.activeElement && document.activeElement.id);
  const target = `glide-cell-4-${idx}`;
  for (let attempt = 0; attempt < 3 && !(await focused() || '').startsWith('glide-cell-4-'); attempt++) {
    const box = await grid.boundingBox();
    const first = [box.x + box.width * 0.7, box.y + 35 + 17];
    await page.mouse.move(...first, { steps: 18 });
    await page.mouse.click(...first);
    await page.waitForTimeout(1500);
  }
  for (let guard = 0; (await focused()) !== target; guard++) {
    if (guard > idx * 4) throw new Error(`Could not reach ${target}; focus is on ${await focused()}`);
    const row = Number(((await focused()) || 'glide-cell-4-0').split('-').pop());
    await page.keyboard.press(row < idx ? 'ArrowDown' : 'ArrowUp');
    await page.waitForTimeout(220);
  }
  await page.waitForTimeout(500);
  await page.keyboard.press('Enter');
  const editor = page.locator('textarea.gdg-input');
  await editor.waitFor({ timeout: 5000 });
  if ((await editor.inputValue()) !== 'Dude,') throw new Error(`Editor opened on "${await editor.inputValue()}"`);
  await page.keyboard.press('Control+A');
  await page.keyboard.type('Good.', { delay: 90 });
  await page.keyboard.press('Enter');
  await page.waitForTimeout(2500);
  await click(page, page.getByRole('button', { name: 'Save caption edits' }));
  await page.waitForTimeout(3000);
  const savedTexts = await page.locator('[data-testid="stText"]').allInnerTexts();
  if (savedTexts[idx] !== 'Good.' || savedTexts.includes('Dude,') || !savedTexts.includes('Thanks for joining the Accessibility Review.')) {
    throw new Error('Caption edit did not land on the intended row: ' + savedTexts.join(' | '));
  }

  // 6. Synchronized playback with audio (muxed in post).
  await page.evaluate(() => window.scrollTo(0, 0));
  await click(page, page.getByRole('tab', { name: 'Playback' }));
  await page.waitForTimeout(1000);
  await click(page, page.getByText('Show available emotion cues'));
  await page.waitForTimeout(2000);
  const iframe = page.locator('iframe').first();
  await scrollTo(iframe);
  await page.waitForTimeout(1500);
  await banner(page, '6 · Synchronized playback: named, colored speakers with word-by-word highlighting (sound on)');
  const media = page.frameLocator('iframe').first().locator('#media');
  events.captionBox = await page.frameLocator('iframe').first().locator('#captions').boundingBox();
  await page.mouse.move(W - 40, 120, { steps: 10 });
  await media.evaluate(m => new Promise(r => m.readyState >= 1 ? r() : m.addEventListener('loadedmetadata', r)));
  await media.evaluate(m => { m.currentTime = 0; m.muted = true; return m.play(); });
  await page.waitForTimeout(1000);
  const sample = await media.evaluate(m => m.currentTime);
  events.playZero = at() - sample;  // recording time at media position 0
  const mediaDuration = await media.evaluate(m => m.duration);
  await page.waitForTimeout(8000);
  await banner(page, 'Text emotion cues are optional, uncertain estimates; transcript and speakers come first');
  await media.evaluate(m => new Promise(r => m.ended ? r() : m.addEventListener('ended', r)));
  events.playEnd = at();
  await page.waitForTimeout(800);

  // 7. Seek and search.
  await banner(page, '7 · Jump to any caption, or search the transcript');
  const seek = page.frameLocator('iframe').first().locator('#seek button');
  for (const n of [7, 14]) {
    const b = seek.nth(n);
    const bb = await b.boundingBox();
    await page.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2, { steps: 18 });
    await b.click();
    await page.waitForTimeout(1500);
  }
  const search = page.getByLabel('Search transcript');
  await typeInto(page, search, 'Toronto'); await search.press('Enter');
  await page.waitForTimeout(1200);
  await scrollTo(search);
  await page.waitForTimeout(2500);
  await search.fill(''); await search.press('Enter');
  await page.waitForTimeout(800);

  // 8. Export.
  await page.evaluate(() => window.scrollTo(0, 0));
  await click(page, page.getByRole('tab', { name: 'Export' }));
  await banner(page, '8 · Export speaker-named SRT, color-styled VTT, or a JSON project; emotion cues (with emoji) are opt-in');
  await page.waitForTimeout(1200);
  await click(page, page.getByText('Include available emotion cues in subtitles'));
  await page.waitForTimeout(2000);
  const dl = async name => {
    const btn = page.getByRole('button', { name });
    await moveTo(page, btn);
    const [d] = await Promise.all([page.waitForEvent('download'), btn.click()]);
    const p = path.join(OUT, d.suggestedFilename()); await d.saveAs(p); return p;
  };
  const srtPath = await dl('Download SRT');
  await dl('Download VTT');
  await dl('Save project JSON');
  await page.waitForTimeout(1500);

  // 9. Results: exported SRT, then scores from context-caption-eval on the unedited model output.
  const srt = fs.readFileSync(srtPath, 'utf8').trim().split(/\n\n/).slice(0, 6).join('\n\n');
  const score = JSON.parse(execFileSync(evalBin, ['--reference', REFERENCE, '--hypothesis',
    path.join(REPO, 'examples/test_conversation_output.json'), '--json']));
  const pct = v => (v * 100).toFixed(1) + '%';
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const map = Object.entries(score.speakers.mapping).map(([h, r]) => `${h} → ${r}`).join(', ');
  await page.setContent(`<!doctype html><html><body style="margin:0;background:#f8fafc;font-family:system-ui,sans-serif;color:#0f172a">
  <div style="display:grid;grid-template-columns:1fr 1.15fr;gap:28px;padding:40px 48px">
   <div><div style="font-size:13px;letter-spacing:.12em;color:#64748b">EXPORTED SUBTITLES (SRT)</div>
    <pre style="background:#0f172a;color:#e2e8f0;padding:20px;border-radius:12px;font-size:14px;line-height:1.45;white-space:pre-wrap;margin-top:10px">${esc(srt)}</pre></div>
   <div><div style="font-size:13px;letter-spacing:.12em;color:#64748b">RESULTS VS. THE SCRIPT · context-caption-eval</div>
    <h1 style="font-size:30px;margin:10px 0 18px">41 s, 3 synthetic voices, WhisperX base on CPU</h1>
    <table style="border-collapse:collapse;width:100%;font-size:18px">
     ${[['Word error rate', pct(score.words.wer), `${score.words.substitutions} sub, ${score.words.deletions} del, ${score.words.insertions} ins / ${score.words.reference_words} words`],
        ['Diarization error rate', pct(score.speakers.der), `collar ${score.speakers.collar} s, 3 speakers given`],
        ['Words with correct speaker', pct(score.speakers.word_speaker_accuracy), `${score.speakers.matched_words} matched words`],
        ['Processing time', `${(events.procEnd - events.procStart).toFixed(0)} s`, `${mediaDuration.toFixed(0)} s of audio, 4 vCPU`]]
       .map(([k, v, d]) => `<tr style="border-bottom:1px solid #e2e8f0"><td style="padding:14px 0">${k}<div style="font-size:13px;color:#64748b">${d}</div></td><td style="text-align:right;font-weight:700;font-size:28px;color:#0f766e">${v}</td></tr>`).join('')}
    </table>
    <p style="font-size:15px;color:#334155;line-height:1.5;margin-top:18px">Speaker mapping: ${esc(map)}.<br>
     Main errors: quick replies attached to the previous speaker; “Good.” heard as “Dude,”.
     With speaker count on auto-detect, DER rose to 38.1% (two similar voices merged).<br>
     Synthetic speech: a pipeline check, not an accuracy claim for real conversation.</p></div>
  </div></body></html>`);
  await page.mouse.move(W - 40, H - 40);
  await banner(page, '9 · Results: transcription is strong; speaker turns need review and correction');
  await page.waitForTimeout(9000);
  events.end = at();

  const video = page.video();
  await context.close();
  await browser.close();
  const raw = await video.path();
  const shown = JSON.parse(fs.readFileSync(path.join(OUT, 'caption-project.json'), 'utf8'));
  events.sync = calibrate(raw, events.captionBox, events.playZero, shown);
  fs.writeFileSync(path.join(OUT, 'events.json'), JSON.stringify({ ...events, metrics, score }, null, 2));

  // Post-process: speed up processing, mux conversation audio at playback, encode H.264/AAC.
  const a = events.procStart + 2, b = events.procEnd - 1;
  const saved = (b - a) * (1 - 1 / SPEEDUP);
  const delayMs = Math.max(0, Math.round((events.sync.playZero - saved) * 1000));
  const out = path.join(OUT, 'app_demo.mp4');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', raw, '-i', MEDIA, '-filter_complex',
    `[0:v]trim=0:${a},setpts=PTS-STARTPTS[v1];[0:v]trim=${a}:${b},setpts=(PTS-STARTPTS)/${SPEEDUP}[v2];` +
    `[0:v]trim=start=${b},setpts=PTS-STARTPTS[v3];[v1][v2][v3]concat=n=3:v=1:a=0,fps=25,format=yuv420p[v];` +
    `[1:a]aresample=44100,adelay=${delayMs}:all=1[a]`,
    '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-preset', 'medium', '-crf', '26',
    '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', out]);
  console.log(`Wrote ${out}; audio offset ${(delayMs / 1000).toFixed(2)} s; events`, events);
})().catch(e => { console.error(e); process.exit(1); });

// Browser recordings have no audio and their clock is offset from Node's. Find the recording time
// of media position 0 from the frames: detect when the yellow word highlight moves, then pick the
// offset that best lines those moments up with the word start times shown in the player.
function calibrate(raw, box, approx, project) {
  const fps = 25;
  const even = v => Math.max(2, Math.round(v) & ~1);
  const [x, y, w, h] = [Math.round(box.x), Math.round(box.y), even(box.width), even(box.height)];
  const [ow, oh] = [even(w / 2), even(h / 2)];
  const start = Math.max(0, approx - 2);
  const buf = execFileSync('ffmpeg', ['-v', 'error', '-ss', String(start), '-i', raw, '-t', '48', '-vf',
    `fps=${fps},crop=${w}:${h}:${x}:${y},scale=${ow}:${oh}`, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
  { maxBuffer: 1 << 30 });
  const size = ow * oh * 3;
  const onsets = [];
  let previous = null;
  for (let f = 0; f < Math.floor(buf.length / size); f++) {
    let sx = 0, sy = 0, n = 0;
    for (let p = 0; p < ow * oh; p++) {
      const i = f * size + p * 3;
      // Highlight is #fde68a; VP8 shifts colors slightly.
      if (buf[i] > 225 && buf[i + 1] > 195 && buf[i + 1] < 250 && buf[i + 2] > 85 && buf[i + 2] < 190) {
        sx += p % ow; sy += Math.floor(p / ow); n++;
      }
    }
    const spot = n > 15 ? [sx / n, sy / n] : null;
    if (spot && (!previous || Math.hypot(spot[0] - previous[0], spot[1] - previous[1]) > 3)) onsets.push(start + f / fps);
    previous = spot;
  }
  const starts = project.captions
    .filter(c => c.alignment_valid && c.text === c.original_text)
    .flatMap(c => c.words.filter(wd => wd.start !== null).map(wd => wd.start));
  let best = approx, bestCost = Infinity;
  for (let d = approx - 1.5; d <= approx + 1.5; d += 0.005) {
    let cost = 0;
    for (const v of onsets) cost += Math.min(0.2, ...starts.map(s => Math.abs(v - d - s)));
    if (cost < bestCost) { bestCost = cost; best = d; }
  }
  return { playZero: best, estimate: approx, onsets: onsets.length, words: starts.length,
           meanResidual: onsets.length ? bestCost / onsets.length : null };
}
