const $ = id => document.getElementById(id);
let models = [], pose = defaultPose(), playing = true, ready = false, online = false;
let busy = false, dirty = false, refreshing = false, revision = 0, retryTimer = 0;
let time = 0, lastTick = performance.now(), lastStart = -Infinity, lastStats = 0;
let currentImage = '', lastFrame = null, frameTimes = [], pointer = null;
const MAX_FPS = 30;
function defaultPose() { return {yaw:0, pitch:0, zoom:1, panx:0, pany:0}; }
function selected() { return models.find(m => m.id === $('model').value); }
function reference() { return selected()?.cameras[Number($('camera').value) || 0]; }
function cycleSeconds() { const range = selected()?.time_range || [0,1]; return range[1]-range[0] > 1 ? range[1]-range[0] : 8; }
function setStatus(text) { if ($('status').textContent !== text) $('status').textContent = text; }
function showTime() { $('time').value = time; $('time-value').textContent = time.toFixed(3); }
function setTime(value) { const [lo, hi] = selected()?.time_range || [0,1]; time = Math.max(lo, Math.min(hi, value)); showTime(); }
function showPlayback() {
  $('play').textContent = playing ? 'Ⅱ 暂停' : '▶ 播放';
  $('play').setAttribute('aria-label', playing ? '暂停播放' : '开始播放');
  $('play').setAttribute('aria-pressed', String(playing));
  $('play-state').textContent = !online ? '等待连接' : !ready ? '正在加载首帧…' : playing ? '正在播放 · 可拖动画面' : '时间已暂停 · 仍可旋转';
}
function setPlaying(value) { playing = value; lastTick = performance.now(); frameTimes = []; showPlayback(); requestFrame(); }
function align() { pose = defaultPose(); setTime(reference()?.time || 0); }
function leaveComparison() { if ($('mode').value === 'compare') { $('mode').value = 'color'; revision++; $('comparison-note').style.display = 'none'; } }
function requestFrame() { dirty = true; }
function retryLater() { clearTimeout(retryTimer); retryTimer = setTimeout(refresh, 5000); }
function failure(error) {
  online = false; ready = false; dirty = false; frameTimes = [];
  const message = error.message || String(error);
  setStatus(message); $('play-state').textContent = '连接中断，正在尝试恢复'; $('fps').textContent = '—';
  if (!currentImage) $('empty').textContent = message;
  retryLater();
}
async function refresh() {
  if (refreshing) return;
  refreshing = true; clearTimeout(retryTimer); $('refresh').disabled = true;
  try {
    setStatus('正在读取服务器模型…');
    const response = await fetch('/api/models', {signal:AbortSignal.timeout(60000)});
    const data = await response.json();
    if (!response.ok) throw Error(data.error);
    if (!Array.isArray(data) || !data.length) throw Error('尚无可加载的模型，稍后自动重试。');
    const old = $('model').value;
    models = data.sort((a,b) => Number(b.role === 'auditfix_full') - Number(a.role === 'auditfix_full'));
    $('model').replaceChildren(...models.map(m => new Option(m.label, m.id)));
    if (models.some(m => m.id === old)) $('model').value = old;
    online = true;
    modelChanged();
  } catch (error) { failure(error); }
  finally { refreshing = false; $('refresh').disabled = false; }
}
function modelChanged() {
  const model = selected(); if (!model) return;
  revision++; ready = false; frameTimes = [];
  $('checkpoint-view').replaceChildren(...(model.checkpoint_views || [{id:'best',label:'当前参考存档'}]).map(v => new Option(v.label,v.id)));
  $('camera').replaceChildren(...model.cameras.map(c => new Option(`${c.id+1} · ${c.name} · t=${c.time.toFixed(3)}`, c.id)));
  const [lo,hi] = model.time_range;
  $('time').min = lo; $('time').max = hi; $('time-min').textContent = lo; $('time-max').textContent = hi;
  $('cycle').textContent = `1× 时每轮 ${cycleSeconds()} 秒`;
  align(); showPlayback(); setStatus('正在加载模型…'); requestFrame();
}
async function draw(now) {
  if (!online || !selected() || busy || !dirty || document.hidden || now-lastStart < 1000/MAX_FPS) return;
  busy = true; dirty = false; lastStart = now;
  const sentRevision = revision;
  const params = {experiment:$('model').value, camera:Number($('camera').value), time,
    width:Number($('resolution').value), mode:$('mode').value, checkpoint_view:$('checkpoint-view').value, ...pose};
  try {
    const response = await fetch('/api/render', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(params), signal:AbortSignal.timeout(60000)});
    const data = await response.json();
    if (!response.ok) throw Error(data.error);
    if (sentRevision !== revision) return;
    const image = new Image(); image.src = 'data:image/jpeg;base64,' + data.image;
    await image.decode();
    if (sentRevision !== revision) return;
    currentImage = image.src; lastFrame = {...params, time:data.time};
    $('frame').src = currentImage; $('frame').style.display = 'block'; $('empty').style.display = 'none';
    $('total').textContent = data.points.toLocaleString(); $('active').textContent = data.active_time_points.toLocaleString();
    $('latency').textContent = data.render_ms.toFixed(1);
    const completedSteps = selected()?.training_complete ? `已训练 ${selected().steps.toLocaleString()} 步 · ` : selected()?.preview_iteration ? '训练中预览 · ' : '';
    $('iteration').textContent = `${completedSteps}展示第 ${data.iteration.toLocaleString()} 步 · t ${data.time.toFixed(3)}`;
    $('comparison-note').style.display = params.mode === 'compare' ? 'block' : 'none';
    const finished = performance.now(); frameTimes.push(finished);
    if (finished-lastStats > 700) {
      $('transport-note').textContent = `SSH 直连 · 本帧含传输与显示 ${(finished-now).toFixed(0)} ms`;
      lastStats = finished;
    }
    ready = true; showPlayback(); setStatus(params.mode === 'compare' ? '对照已对齐原图的时间与相机。点击播放可回到自由观察。' : '已连接 · 时间播放与视角操作可同时进行');
  } catch (error) { if (sentRevision === revision) failure(error); }
  finally { busy = false; }
}
function tick(now) {
  const dt = Math.min((now-lastTick)/1000, 0.1); lastTick = now;
  if (!document.hidden && online && ready && playing && selected()) {
    const [lo, hi] = selected().time_range, span = hi-lo;
    time = span > 0 ? lo + ((time-lo + dt*Number($('speed').value)*span/cycleSeconds()) % span) : lo;
    showTime(); requestFrame();
  }
  frameTimes = frameTimes.filter(t => now-t < 2000);
  $('fps').textContent = frameTimes.length > 1 ? ((frameTimes.length-1)*1000/(now-frameTimes[0])).toFixed(1) : ready ? '0.0' : '—';
  void draw(now);
  requestAnimationFrame(tick);
}
$('model').onchange = modelChanged;
$('checkpoint-view').onchange = () => { revision++; ready = false; frameTimes = []; showPlayback(); requestFrame(); };
$('camera').onchange = () => { revision++; ready = false; align(); showPlayback(); requestFrame(); };
$('mode').onchange = () => {
  revision++;
  if ($('mode').value === 'compare') { setPlaying(false); align(); }
  requestFrame();
};
$('resolution').onchange = () => { revision++; requestFrame(); };
$('time').oninput = () => { leaveComparison(); revision++; setTime(Number($('time').value)); lastTick = performance.now(); requestFrame(); };
$('reset').onclick = () => { leaveComparison(); pose = defaultPose(); requestFrame(); };
$('refresh').onclick = refresh;
$('play').onclick = () => { leaveComparison(); setPlaying(!playing); };
$('download').onclick = () => {
  if (!currentImage || !lastFrame) return;
  const link = document.createElement('a'); link.href = currentImage;
  link.download = `${lastFrame.experiment}_t${lastFrame.time.toFixed(3)}.jpg`; link.click();
};
$('fullscreen').onclick = async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('viewer-surface').requestFullscreen(); }
  catch { setStatus('当前嵌入窗口不支持全屏，可在浏览器中打开 http://127.0.0.1:8765。'); }
};
document.addEventListener('fullscreenchange', () => { $('fullscreen').textContent = document.fullscreenElement ? '退出全屏' : '全屏'; });
document.addEventListener('visibilitychange', () => { lastTick = performance.now(); frameTimes = []; if (!document.hidden) requestFrame(); });
const stage = $('stage');
stage.oncontextmenu = event => event.preventDefault();
stage.onpointerdown = event => {
  if (pointer || ![0,2].includes(event.button)) return;
  pointer = {id:event.pointerId, x:event.clientX, y:event.clientY, button:event.button};
  stage.setPointerCapture(event.pointerId); stage.focus({preventScroll:true});
};
stage.onpointermove = event => {
  if (!pointer || pointer.id !== event.pointerId) return;
  const dx = event.clientX-pointer.x, dy = event.clientY-pointer.y;
  pointer.x = event.clientX; pointer.y = event.clientY; leaveComparison();
  if (pointer.button === 2 || event.shiftKey) { pose.panx -= dx*.004; pose.pany += dy*.004; }
  else { pose.yaw -= dx*.008; pose.pitch = Math.max(-1.45, Math.min(1.45, pose.pitch-dy*.008)); }
  // Do not touch the playback clock here: motion and orbit are independent.
  requestFrame();
};
function releasePointer(event) { if (pointer?.id === event.pointerId) pointer = null; }
stage.onpointerup = releasePointer; stage.onpointercancel = releasePointer; stage.onlostpointercapture = releasePointer;
stage.addEventListener('wheel', event => {
  event.preventDefault(); leaveComparison(); pose.zoom = Math.min(4, Math.max(.25, pose.zoom*Math.exp(event.deltaY*.001))); requestFrame();
}, {passive:false});
stage.onkeydown = event => { if (event.code === 'Space') { event.preventDefault(); $('play').click(); } };
showPlayback(); void refresh(); requestAnimationFrame(tick);

// Training work happens on the server and survives closing this page.
let previousCompleted = null;
async function pollTraining() {
  try {
    const response = await fetch('/api/progress', {signal:AbortSignal.timeout(60000)});
    if (!response.ok) throw Error('Progress unavailable');
    const rows = await response.json();
    $('training-progress').hidden = !rows.length;
    const stages = {starting:'启动中', preparing:'准备完整数据', recovering:'正在恢复训练', migrating:'迁移到空闲显卡', training:'训练中', waiting:'长时间未更新，需检查', trained:'训练结束，等待评估', evaluating:'评估与导出', complete:'已完成', failed:'发生错误，需检查日志'};
    const prepared = {colmap_features:'提取特征',colmap_matching:'匹配特征',colmap_triangulation:'三角化',colmap_undistort:'图像校正',colmap_dense_stereo:'稠密立体重建',colmap_fusion:'融合点云',waiting_for_frame_extraction:'提取视频帧',complete:'等待训练资源'};
    $('training-rows').replaceChildren(...rows.map(row => {
      const line = document.createElement('div');
      const label = row.label || row.id.replace('_paper_full','').replace('_auditfix_full','').replaceAll('_',' ');
      const detail = row.stage === 'preparing' ? (prepared[row.preparation] || '准备数据') : `${row.step.toLocaleString()} / ${row.target_steps.toLocaleString()} 步`;
      line.textContent = `${label}：${stages[row.stage] || row.stage} · ${detail}`;
      if (row.gpu != null && !['complete','preparing'].includes(row.stage)) line.textContent += ` · GPU${row.gpu}`;
      if (row.gpu_migration?.stage === 'waiting_for_checkpoint') line.textContent += ` · 已安排转至 GPU${row.gpu_migration.to_gpu}（等待第 ${row.gpu_migration.earliest_iteration.toLocaleString()} 步起的可续训存档）`;
      else if (row.gpu_migration?.stage === 'waiting_for_gpu0') line.textContent += ` · 等待 GPU${row.gpu_migration.to_gpu} 可用`;
      if (row.stage === 'failed') line.style.color = '#ffadad';
      return line;
    }));
    const completed = rows.filter(row => row.stage === 'complete').map(row => row.id).sort().join(',');
    if (previousCompleted !== null && completed !== previousCompleted) void refresh();
    previousCompleted = completed;
  } catch { /* Connection errors are explained by the main viewer status. */ }
  finally { setTimeout(pollTraining, 15000); }
}
void pollTraining();
