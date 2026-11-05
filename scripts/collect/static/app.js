// 수집 도구 프런트. 상태 클래스는 ok / warn / bad 만 쓴다.
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

async function api(url, body) {
  const opt = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({ok: false, error: `HTTP ${r.status}`}));
  return j;
}

// ─── 페이지 안 확인 단계 (confirm() 금지) ───
let confirmFn = null;
function askConfirm(msg, fn) { $('confirmMsg').textContent = msg; confirmFn = fn; $('confirm').hidden = false; }
$('confirmYes').onclick = () => { $('confirm').hidden = true; const f = confirmFn; confirmFn = null; if (f) f(); };
$('confirmNo').onclick = () => { $('confirm').hidden = true; confirmFn = null; };

// ─── 탭 ───
function showTab(rec) {
  $('viewRec').hidden = !rec; $('viewMng').hidden = rec;
  $('tabRec').classList.toggle('active', rec); $('tabMng').classList.toggle('active', !rec);
  // 관리 화면에서는 MJPEG 연결을 끊는다
  $('cam1').src = rec ? '/stream/camera1' : ''; $('cam3').src = rec ? '/stream/camera3' : '';
  if (!rec) loadSessions();
  else if (player) player.playing = false;
}
$('tabRec').onclick = () => showTab(true);
$('tabMng').onclick = () => showTab(false);

// ─── 녹화 화면 ───
const FLOW = {idle: -1, align: 0, ready: 1, recording: 2, result: 3, saving: 3};
let devBuilt = false;

function buildDevs(joints, tol, tolGripper) {
  const scale = Math.max(tol, tolGripper) * 3;  // 트랙 반폭 = 허용치 3배
  $('devs').innerHTML = joints.map((j, i) => `
    <div class="dev"><span>${esc(j)}</span>
      <div class="track"><div class="tol" style="left:${50 - (j === 'gripper' ? tolGripper : tol) / scale * 50}%;width:${(j === 'gripper' ? tolGripper : tol) / scale * 100}%"></div>
        <div class="mid"></div><div class="mark" id="mark${i}" style="left:50%"></div></div>
      <span class="val" id="val${i}">-</span></div>`).join('');
  $('tolText').textContent = `${tol} (팔), ±${tolGripper} (그리퍼)`;
  devBuilt = true;
}

function renderStatus(s) {
  if (!devBuilt) buildDevs(s.joints, s.tol, s.tol_gripper);
  const scale = Math.max(s.tol, s.tol_gripper) * 3;
  document.querySelectorAll('#flow li').forEach(li => {
    li.classList.toggle('active', FLOW[s.phase] === FLOW[li.dataset.phase]);
  });
  $('teleopBanner').hidden = s.teleop;
  $('errorBanner').hidden = !s.error; $('errorBanner').textContent = s.error || '';
  $('sessionInfo').textContent = s.session ? `${s.session} (이번 실행 저장 ${s.saved_count})` : '세션 없음';
  $('btnStart').hidden = !!s.session; $('btnEnd').hidden = !s.session;
  $('product').disabled = !!s.session;
  $('btnEnd').disabled = ['recording', 'result', 'saving'].includes(s.phase);

  $('holdBar').style.width = `${s.hold * 100}%`;
  const rec = s.phase === 'recording';
  const pct = rec ? Math.min(1, s.elapsed / s.duration) : (FLOW[s.phase] >= 3 ? 1 : 0);
  $('progBar').style.width = `${pct * 100}%`;
  $('progBar').className = 'fill ' + (rec ? 'warn' : '');
  $('targetLine').style.left = `${s.target / s.duration * 100}%`;
  $('progText').textContent = rec ? `${s.elapsed.toFixed(1)} / ${s.duration}초 (목표 복귀 ${s.target}초)` : '';

  for (let i = 0; i < s.joints.length; i++) {
    const tol = s.joints[i] === 'gripper' ? s.tol_gripper : s.tol;
    const d = s.dev ? s.dev[i] : null, m = $('mark' + i);
    if (d === null) { m.style.left = '50%'; m.className = 'mark'; $('val' + i).textContent = '-'; continue; }
    const c = Math.max(-scale, Math.min(scale, d));
    m.style.left = `${50 + c / scale * 50}%`;
    m.className = 'mark ' + (Math.abs(d) <= tol ? 'ok' : Math.abs(d) <= 2 * tol ? 'warn' : 'bad');
    $('val' + i).textContent = (d > 0 ? '+' : '') + d;
  }
  for (const [k, id] of [['camera1', 'age1'], ['camera3', 'age3']]) {
    const a = s.ages[k], el = $(id);
    el.textContent = a === null ? '(수신 없음)' : `${a.toFixed(1)}초 전`;
    el.className = a === null || a > s.max_gap ? 'bad-t' : 'muted';
  }

  const res = s.result;
  $('resultCard').hidden = !res;
  if (res) {
    $('verdict').textContent = res.verdict; $('verdict').className = 'badge ' + res.verdict;
    const f = v => v === null || v === undefined ? '-' : v;
    $('resultList').innerHTML = [
      ['길이', f(res.duration_sec) + '초'],
      ['샘플 수', `상태 ${res.counts.state} / cam1 ${res.counts.camera1} / cam3 ${res.counts.camera3}`],
      ['복귀 시각', f(res.return_time_sec) + (res.return_time_sec === null ? '' : '초')],
      ['속도 편차', f(res.speed_dev_sec)],
      ['시작 편차(raw)', f(res.start_dev_raw && res.start_dev_raw.join(', '))],
      ['끝 편차(팔 5축)', f(res.end_dev_raw && res.end_dev_raw.join(', '))],
      ['그리퍼 닫힘', res.gripper_closed === null ? '-' : (res.gripper_closed ? '예' : '아니오')],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');
    $('warnList').innerHTML = res.warnings.map(w => `<li class="${res.verdict}-t">${esc(w)}</li>`).join('');
  }
  $('btnSave').disabled = s.phase !== 'result'; $('btnDiscard').disabled = s.phase !== 'result';
}

$('btnStart').onclick = async () => {
  const j = await api('/api/session', {product: $('product').value});
  if (!j.ok) return $('errorBanner').textContent = j.error, $('errorBanner').hidden = false;
  $('sessionWarn').hidden = !j.warnings.length; $('sessionWarn').textContent = j.warnings.join(' / ');
  loadCams();
};
$('btnEnd').onclick = async () => { await api('/api/session/end', {}); $('sessionWarn').hidden = true; };

// ─── 카메라 설정 ───
const CAM_LABEL = {camera1: 'camera1 (사이드캠)', camera3: 'camera3 (탑뷰)'};
const CTRL_LABEL = {auto_exposure: '자동 노출', exposure_time_absolute: '노출 시간', gain: '게인', brightness: '밝기',
  white_balance_automatic: '자동 화이트밸런스', white_balance_temperature: '화이트밸런스 온도(K)',
  exposure_dynamic_framerate: '어두우면 fps 낮춤'};
const AE_MANUAL = 1, AE_AUTO = 3;  // v4l2 auto_exposure 메뉴: 1 수동, 3 조리개 우선(자동)
const isOn = (k, v) => k === 'auto_exposure' ? v !== AE_MANUAL : !!v;

function renderCams(cams) {
  let allMatch = true, anyProfile = false;
  $('camPanels').innerHTML = Object.entries(cams).map(([cam, c]) => {
    if (c.error) { allMatch = false; return `<div class="campanel"><h3>${CAM_LABEL[cam]}</h3><p class="bad-t">${esc(c.error)}</p></div>`; }
    const prof = c.profile || {};
    if (c.profile) anyProfile = true; else allMatch = false;
    const rows = Object.keys(CTRL_LABEL).filter(k => c.ctrls[k]).map(k => {
      const x = c.ctrls[k], p = prof[k];
      const diff = p !== undefined && p !== x.value;
      if (diff) allMatch = false;
      const input = (x.type === 'int')
        ? `<input type="number" data-cam="${cam}" data-k="${k}" min="${x.min}" max="${x.max}" step="${x.step || 1}" value="${x.value}">`
        : `<input type="checkbox" data-cam="${cam}" data-k="${k}" ${isOn(k, x.value) ? 'checked' : ''}>`;
      const show = v => v === undefined ? '-' : (x.type === 'int' ? v : (isOn(k, v) ? '켬' : '끔'));
      return `<tr><td>${CTRL_LABEL[k]}${x.type === 'int' ? ` <span class="muted small">${x.min}~${x.max}</span>` : ''}</td>
        <td>${input}</td><td class="${diff ? 'warn-t' : ''}">${show(x.value)}</td><td>${show(p)}</td></tr>`;
    }).join('');
    return `<div class="campanel"><h3>${CAM_LABEL[cam]} <span class="muted small">${esc(c.device)}</span></h3>
      <table><tr><th>항목</th><th>바꿀 값</th><th>현재</th><th>프로필</th></tr>${rows}</table>
      <button data-apply="${cam}">적용</button></div>`;
  }).join('');
  const b = $('camProfileBadge');
  b.className = 'badge ' + (!anyProfile ? 'warn' : allMatch ? 'ok' : 'bad');
  b.textContent = !anyProfile ? '프로필 없음' : allMatch ? '프로필 적용됨' : '현재 값 ≠ 프로필';
  document.querySelectorAll('[data-apply]').forEach(btn => btn.onclick = () => applyCam(btn.dataset.apply));
}

function camMsg(text, cls) { $('camMsg').textContent = text; $('camMsg').className = cls ? cls + '-t' : 'muted'; }

async function loadCams() {
  const j = await api('/api/camera');
  if (j.ok) renderCams(j.cameras); else camMsg(j.error, 'bad');
}

async function camAction(url, body, okText) {
  const j = await api(url, body);
  if (j.cameras) renderCams(j.cameras);
  camMsg(j.ok ? okText : `실패: ${j.error}`, j.ok ? 'ok' : 'bad');
}

function applyCam(cam) {
  const v = {};
  document.querySelectorAll(`[data-cam="${cam}"]`).forEach(el => {
    const k = el.dataset.k;
    if (el.type === 'checkbox') v[k] = k === 'auto_exposure' ? (el.checked ? AE_AUTO : AE_MANUAL) : (el.checked ? 1 : 0);
    else v[k] = Number(el.value);
  });
  // 자동 모드가 무시하는 수동 값은 보내지 않는다(장치가 거부함)
  if (v.auto_exposure !== undefined && v.auto_exposure !== AE_MANUAL) delete v.exposure_time_absolute;
  if (v.white_balance_automatic) delete v.white_balance_temperature;
  camAction('/api/camera/set', {camera: cam, ctrls: v}, `${cam} 적용됨 — 영상 확인 후 프로필 저장`);
}
$('btnCamSave').onclick = () => camAction('/api/camera/profile/save', {}, '프로필 저장됨 (다음 세션·로봇 기동부터 이 값)');
$('btnCamApply').onclick = () => camAction('/api/camera/profile/apply', {}, '프로필 다시 적용됨');
$('btnCamReload').onclick = () => { camMsg(''); loadCams(); };
loadCams();
$('btnSave').onclick = async () => {
  const j = await api('/api/episode/save', {});
  if (!j.ok) $('errorBanner').textContent = '저장 실패: ' + j.error, $('errorBanner').hidden = false;
};
$('btnDiscard').onclick = () => askConfirm('이 에피소드를 버립니다.', () => api('/api/episode/discard', {}));

if (typeof io !== 'undefined') io().on('status', renderStatus);
else setInterval(async () => renderStatus(await api('/api/status')), 500);  // CDN 접속 불가 시 폴링

// ─── 데이터 관리 화면 ───
let convReady = false, sessionData = [], selSessions = new Set(), selEps = new Set(), datasetRoot = '', currentSid = null;

async function loadSessions() {
  const j = await api('/api/sessions');
  sessionData = j.sessions; datasetRoot = j.dataset_root;
  selSessions = new Set([...selSessions].filter(id => sessionData.some(s => s.id === id)));
  selEps = new Set([...selEps].filter(k => sessionData.some(s => s.episodes.some(e => k === `${s.id}:${e.index}`))));
  convReady = j.convert_ready; currentSid = j.current;
  if (player && !findEp(player.sid, player.ep)) closePlayer();  // 삭제·재정렬로 사라진 에피소드
  renderSessions(currentSid);
  pollConvert(true);
}

function maxAbs(a) { return a ? Math.max(...a.map(Math.abs)) : '-'; }

function renderSessions(current) {
  if (!sessionData.length) { $('sessions').innerHTML = '<p class="muted">세션이 없습니다.</p>'; return; }
  $('sessions').innerHTML = sessionData.map(s => `
    <div class="sess">
      <div class="head">
        <label><input type="checkbox" data-sess="${esc(s.id)}" ${selSessions.has(s.id) ? 'checked' : ''}> <b>${esc(s.id)}</b></label>
        <span>품종 ${esc(s.product)}</span><span class="muted">${s.episodes.length}개${s.id === current ? ' · 진행 중' : ''}</span>
        <span class="muted">사용 ${s.episodes.filter(e => e.review === 'keep').length} · 안 씀 ${s.episodes.filter(e => e.review === 'discard').length} · 미정 ${s.episodes.filter(e => !e.review).length}</span>
        ${s.merged_from.length ? `<span class="muted">합침: ${esc(s.merged_from.join(', '))}</span>` : ''}
        ${s.error ? `<span class="bad-t">${esc(s.error)}</span>` : ''}
      </div>
      ${s.episodes.length ? `<div class="eps"><table>
        <tr><th></th><th>#</th><th>길이</th><th>복귀</th><th>시작 편차</th><th>끝 편차</th><th>판정</th><th>검수</th><th>경고</th><th></th></tr>
        ${s.episodes.map(e => `<tr class="${player && player.sid === s.id && player.ep === e.index ? 'playing' : ''}">
          <td><input type="checkbox" data-ep="${esc(s.id)}:${e.index}" ${selEps.has(s.id + ':' + e.index) ? 'checked' : ''}></td>
          <td>${e.index}</td><td>${e.duration_sec ?? '-'}</td><td>${e.return_time_sec ?? '-'}</td>
          <td>${maxAbs(e.start_dev_raw)}</td><td>${maxAbs(e.end_dev_raw)}</td>
          <td><span class="badge ${esc(e.verdict)}">${esc(e.verdict)}</span></td>
          <td><span class="badge ${reviewCls(e.review)}">${REVIEW_LABEL[e.review] || '미정'}</span></td>
          <td>${esc((e.warnings || []).join('; '))}</td>
          <td><button class="secondary" data-play="${esc(s.id)}:${e.index}">재생</button></td></tr>`).join('')}
      </table></div>` : ''}
    </div>`).join('');
}

$('sessions').onchange = ev => {
  const t = ev.target;
  if (t.dataset.sess) { t.checked ? selSessions.add(t.dataset.sess) : selSessions.delete(t.dataset.sess); setConvDefault(); }
  if (t.dataset.ep) t.checked ? selEps.add(t.dataset.ep) : selEps.delete(t.dataset.ep);
};
$('sessions').onclick = ev => {
  const key = ev.target.dataset.play;
  if (!key) return;
  const i = key.lastIndexOf(':');
  openPlayer(key.slice(0, i), +key.slice(i + 1));
};

// ─── 재생·검수 ───
// 두 카메라를 녹화 수신 시각(stamps) 기준으로 실제 속도로 재생한다. 시각 t 에는 t 이하 마지막 프레임을 보여준다
// (변환기의 격자 맞춤과 같은 규칙이라 보이는 것이 학습에 들어갈 장면이다).
const REVIEW_LABEL = {keep: '사용', discard: '안 씀'};
const reviewCls = r => r === 'keep' ? 'ok' : r === 'discard' ? 'bad' : 'none';
const CAMS = [['camera1', 'pcam1'], ['camera3', 'pcam3']];
let player = null;

function findEp(sid, ep) { const s = sessionData.find(x => x.id === sid); return s && s.episodes.find(e => e.index === ep); }
function epOrder() { return sessionData.flatMap(s => s.episodes.map(e => [s.id, e.index])); }

async function openPlayer(sid, ep) {
  const j = await api(`/api/episode/${encodeURIComponent(sid)}/${ep}/stamps`);
  if (!j.ok) return mngMsg(`재생 실패: ${j.error}`, 'bad');
  const base = `/api/frame/${encodeURIComponent(sid)}/${ep}`;
  const urls = {};
  for (const [c] of CAMS) urls[c] = j.stamps[c].map((_, k) => { const u = `${base}/${c}/${k}`; new Image().src = u; return u; });
  const t0 = Math.min(0, ...CAMS.map(([c]) => j.stamps[c][0] ?? 0));
  player = {sid, ep, stamps: j.stamps, urls, t0, t: t0, dur: j.duration, target: j.target, playing: true, last: null, shown: {}};
  $('playerCard').hidden = false;
  $('playerTitle').textContent = `${sid} / ep_${String(ep).padStart(4, '0')}`;
  $('ptarget').style.left = `${(j.target - t0) / (j.duration - t0) * 100}%`;
  renderSessions(currentSid); renderReview();
  $('playerCard').scrollIntoView({block: 'nearest'});
  requestAnimationFrame(tick(player));
}

function closePlayer() { player = null; $('playerCard').hidden = true; renderSessions(currentSid); }

function lastAtOrBefore(ts, t) {
  let lo = 0, hi = ts.length - 1, ans = 0;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (ts[m] <= t) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}

function tick(p) {
  return now => {
    if (p !== player) return;  // 다른 에피소드를 열었거나 닫힘
    if (p.playing && p.last !== null) p.t = Math.min(p.dur, p.t + (now - p.last) / 1000);
    if (p.t >= p.dur) p.playing = false;
    p.last = p.playing ? now : null;
    for (const [c, id] of CAMS) {
      if (!p.urls[c].length) continue;
      const k = lastAtOrBefore(p.stamps[c], p.t);
      if (p.shown[c] !== k) { p.shown[c] = k; $(id).src = p.urls[c][k]; }
    }
    $('pfill').style.width = `${(p.t - p.t0) / (p.dur - p.t0) * 100}%`;
    const order = epOrder(), pos = order.findIndex(([s, e]) => s === p.sid && e === p.ep);
    $('playerPos').textContent = `${Math.max(0, p.t).toFixed(2)} / ${p.dur}초 · ${pos + 1} / ${order.length}`;
    $('pPlay').textContent = p.playing ? '정지' : '재생';
    requestAnimationFrame(tick(p));
  };
}

function togglePlay() {
  if (!player) return;
  if (!player.playing && player.t >= player.dur) player.t = player.t0;
  player.playing = !player.playing;
}
function restart() { if (player) { player.t = player.t0; player.playing = true; } }

function renderReview() {
  const e = player && findEp(player.sid, player.ep);
  const r = e && e.review;
  $('playerReview').className = 'badge ' + reviewCls(r);
  $('playerReview').textContent = REVIEW_LABEL[r] || '미정';
}

function step(d) {
  if (!player) return;
  const order = epOrder(), i = order.findIndex(([s, e]) => s === player.sid && e === player.ep) + d;
  if (i < 0 || i >= order.length) return mngMsg(d > 0 ? '마지막 에피소드' : '첫 에피소드', 'warn');
  openPlayer(...order[i]);
}

async function setReview(review, advance) {
  if (!player) return;
  const {sid, ep} = player;
  const j = await api('/api/review', {session: sid, episode: ep, review});
  if (!j.ok) return mngMsg(`검수 기록 실패: ${j.error}`, 'bad');
  const e = findEp(sid, ep);
  if (e) { if (review) e.review = review; else delete e.review; }
  mngMsg(`ep_${String(ep).padStart(4, '0')} → ${REVIEW_LABEL[review] || '미정'}`, review === 'discard' ? 'warn' : 'ok');
  renderReview(); renderSessions(currentSid);
  if (advance) step(1);
}

$('pPlay').onclick = togglePlay;
$('pKeep').onclick = () => setReview('keep', true);
$('pDrop').onclick = () => setReview('discard', true);
$('pClear').onclick = () => setReview(null, false);
$('pClose').onclick = closePlayer;
$('pbar').onclick = ev => {
  if (!player) return;
  const r = $('pbar').getBoundingClientRect();
  player.t = player.t0 + Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * (player.dur - player.t0);
};

// 한글 입력 상태에서도 동작하도록 e.key 대신 물리 키(e.code)로 판정
document.addEventListener('keydown', ev => {
  if (!player || $('viewMng').hidden || !$('confirm').hidden) return;
  if (ev.target.matches('input, textarea') || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const act = {Space: togglePlay, KeyR: restart, ArrowLeft: () => step(-1), ArrowRight: () => step(1),
               KeyY: () => setReview('keep', true), KeyN: () => setReview('discard', true),
               KeyU: () => setReview(null, false), Escape: closePlayer}[ev.code];
  if (!act) return;
  ev.preventDefault();
  act();
});
$('btnReload').onclick = loadSessions;

function setConvDefault() {
  const ids = [...selSessions];
  if (ids.length) $('convOut').value = `${datasetRoot}/${ids.length === 1 ? ids[0] : 'combined'}`;
}
function mngMsg(text, cls) { $('mngMsg').textContent = text; $('mngMsg').className = cls ? cls + '-t' : 'muted'; }

$('btnDelete').onclick = () => {
  const by = {};
  [...selEps].forEach(k => { const i = k.lastIndexOf(':'); (by[k.slice(0, i)] ||= []).push(+k.slice(i + 1)); });
  const n = [...selEps].length;
  if (!n) return mngMsg('삭제할 에피소드를 선택하세요', 'warn');
  const ren = $('renumber').checked;
  askConfirm(`에피소드 ${n}개를 삭제합니다${ren ? ' (번호 재정렬)' : ''}. 되돌릴 수 없습니다.`, async () => {
    for (const [sid, eps] of Object.entries(by)) {
      const j = await api('/api/delete', {session: sid, episodes: eps, renumber: ren});
      if (!j.ok) return mngMsg(`삭제 실패(${sid}): ${j.error}`, 'bad');
    }
    selEps.clear(); mngMsg(`${n}개 삭제됨`, 'ok'); loadSessions();
  });
};

$('btnMerge').onclick = () => {
  const ids = [...selSessions].sort();  // 이름(=시각) 오름차순으로 합쳐 번호가 시간순이 되게
  if (ids.length < 1) return mngMsg('합칠 세션을 선택하세요', 'warn');
  const move = $('mergeMove').checked;
  askConfirm(`세션 ${ids.length}개를 새 세션으로 ${move ? '이동' : '복사'}해 합칩니다.`, async () => {
    const j = await api('/api/merge', {sessions: ids, out_name: $('mergeName').value, move});
    if (!j.ok) return mngMsg(`합치기 거부: ${j.error}`, 'bad');
    selSessions.clear(); mngMsg(`${j.session} 생성 (에피소드 ${j.episodes}개)`, 'ok'); loadSessions();
  });
};

// ─── 변환 ───
let convTimer = null;
$('btnConvert').onclick = () => {
  const ids = [...selSessions].sort();
  if (!ids.length) return mngMsg('변환할 세션을 선택하세요', 'warn');
  if (!$('convOut').value) setConvDefault();
  askConfirm(`세션 ${ids.length}개를 ${$('convOut').value} 로 변환합니다.`, async () => {
    const j = await api('/api/convert', {sessions: ids, out: $('convOut').value});
    if (!j.ok) return mngMsg(j.missing ? '변환 도구 미설치' : `변환 시작 실패: ${j.error}`, 'bad');
    mngMsg('', ''); pollConvert(true);
  });
};
async function pollConvert(start) {
  const j = await api('/api/convert');
  if (!j.ok) return;
  const log = $('convLog'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = j.lines.join('\n');
  if (atEnd) log.scrollTop = log.scrollHeight;
  $('convState').textContent = j.running ? '변환 중…' : (convReady ? '' : '변환 도구 미설치');
  $('btnConvert').disabled = j.running || !convReady;
  const done = !j.running && j.rc !== null;
  $('convResult').hidden = !done;
  if (done) {
    // 변환기는 마지막 줄에 품질 점검 요약 JSON 을 출력한다(없으면 로그 끝부분을 그대로 보여준다)
    let sum = null;
    try { sum = JSON.parse(j.lines[j.lines.length - 1]); } catch (e) { /* 요약 없음 */ }
    const bad = j.rc !== 0 || (sum && (!sum.length_ok || sum.tasks !== 1));
    const warn = !bad && sum && sum.warn_episodes > 0;
    const q = j.lines.lastIndexOf('품질 점검');
    const block = j.lines.slice(q >= 0 ? q : -8).filter(l => !l.startsWith('{')).join('\n');
    $('convResult').className = 'banner ' + (bad ? 'bad' : warn ? 'warn' : 'ok');
    $('convResult').innerHTML = `<b>${j.rc === 0 ? '변환 완료' : `변환 실패 (종료 코드 ${j.rc})`}</b><pre style="margin:6px 0 0;white-space:pre-wrap">${esc(block)}</pre>`;
  }
  clearTimeout(convTimer);
  if (j.running) convTimer = setTimeout(pollConvert, 1000);
}

showTab(true);
