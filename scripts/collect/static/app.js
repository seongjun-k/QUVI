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
}
$('tabRec').onclick = () => showTab(true);
$('tabMng').onclick = () => showTab(false);

// ─── 녹화 화면 ───
const FLOW = {idle: -1, align: 0, ready: 1, recording: 2, result: 3, saving: 3};
let devBuilt = false;

function buildDevs(joints, tol) {
  const scale = tol * 3;  // 트랙 반폭 = 허용치 3배
  $('devs').innerHTML = joints.map((j, i) => `
    <div class="dev"><span>${esc(j)}</span>
      <div class="track"><div class="tol" style="left:${50 - tol / scale * 50}%;width:${tol / scale * 100}%"></div>
        <div class="mid"></div><div class="mark" id="mark${i}" style="left:50%"></div></div>
      <span class="val" id="val${i}">-</span></div>`).join('');
  $('tolText').textContent = tol;
  devBuilt = true;
}

function renderStatus(s) {
  if (!devBuilt) buildDevs(s.joints, s.tol);
  const scale = s.tol * 3;
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
    const d = s.dev ? s.dev[i] : null, m = $('mark' + i);
    if (d === null) { m.style.left = '50%'; m.className = 'mark'; $('val' + i).textContent = '-'; continue; }
    const c = Math.max(-scale, Math.min(scale, d));
    m.style.left = `${50 + c / scale * 50}%`;
    m.className = 'mark ' + (Math.abs(d) <= s.tol ? 'ok' : Math.abs(d) <= 2 * s.tol ? 'warn' : 'bad');
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
  if (!j.ok) $('errorBanner').textContent = j.error, $('errorBanner').hidden = false;
};
$('btnEnd').onclick = async () => { await api('/api/session/end', {}); };
$('btnSave').onclick = async () => {
  const j = await api('/api/episode/save', {});
  if (!j.ok) $('errorBanner').textContent = '저장 실패: ' + j.error, $('errorBanner').hidden = false;
};
$('btnDiscard').onclick = () => askConfirm('이 에피소드를 버립니다.', () => api('/api/episode/discard', {}));

if (typeof io !== 'undefined') io().on('status', renderStatus);
else setInterval(async () => renderStatus(await api('/api/status')), 500);  // CDN 접속 불가 시 폴링

// ─── 데이터 관리 화면 ───
let convReady = false, sessionData = [], selSessions = new Set(), selEps = new Set(), datasetRoot = '';

async function loadSessions() {
  const j = await api('/api/sessions');
  sessionData = j.sessions; datasetRoot = j.dataset_root;
  selSessions = new Set([...selSessions].filter(id => sessionData.some(s => s.id === id)));
  selEps = new Set([...selEps].filter(k => sessionData.some(s => s.episodes.some(e => k === `${s.id}:${e.index}`))));
  convReady = j.convert_ready;
  renderSessions(j.current);
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
        ${s.merged_from.length ? `<span class="muted">합침: ${esc(s.merged_from.join(', '))}</span>` : ''}
        ${s.error ? `<span class="bad-t">${esc(s.error)}</span>` : ''}
      </div>
      ${s.episodes.length ? `<div class="eps"><table>
        <tr><th></th><th>#</th><th>길이</th><th>복귀</th><th>시작 편차</th><th>끝 편차</th><th>판정</th><th>경고</th><th></th></tr>
        ${s.episodes.map(e => `<tr>
          <td><input type="checkbox" data-ep="${esc(s.id)}:${e.index}" ${selEps.has(s.id + ':' + e.index) ? 'checked' : ''}></td>
          <td>${e.index}</td><td>${e.duration_sec ?? '-'}</td><td>${e.return_time_sec ?? '-'}</td>
          <td>${maxAbs(e.start_dev_raw)}</td><td>${maxAbs(e.end_dev_raw)}</td>
          <td><span class="badge ${esc(e.verdict)}">${esc(e.verdict)}</span></td>
          <td>${esc((e.warnings || []).join('; '))}</td>
          <td><button class="secondary" data-prev="${esc(s.id)}:${e.index}">미리보기</button></td></tr>`).join('')}
      </table></div>` : ''}
    </div>`).join('');
}

$('sessions').onchange = ev => {
  const t = ev.target;
  if (t.dataset.sess) { t.checked ? selSessions.add(t.dataset.sess) : selSessions.delete(t.dataset.sess); setConvDefault(); }
  if (t.dataset.ep) t.checked ? selEps.add(t.dataset.ep) : selEps.delete(t.dataset.ep);
};
$('sessions').onclick = ev => {
  const key = ev.target.dataset.prev;
  if (!key) return;
  const [sid, ep] = [key.slice(0, key.lastIndexOf(':')), key.slice(key.lastIndexOf(':') + 1)];
  $('previewCard').hidden = false; $('previewTitle').textContent = `${sid} / ep_${ep.padStart(4, '0')}`;
  $('preview').innerHTML = ['camera1', 'camera3'].flatMap(c => ['first', 'mid', 'last'].map(p =>
    `<figure style="margin:0"><img src="/api/frame/${encodeURIComponent(sid)}/${ep}/${c}/${p}"><figcaption>${c} · ${{first: '처음', mid: '중간', last: '끝'}[p]}</figcaption></figure>`)).join('');
};
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
