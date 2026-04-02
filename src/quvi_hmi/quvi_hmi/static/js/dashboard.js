/* ═══════════════════════════════════════════
   QUVI HMI Dashboard — Client-Side Logic
   WebSocket 실시간 업데이트 + UI 렌더링
   ═══════════════════════════════════════════ */

// ─── WebSocket 연결 ───
const socket = io();

socket.on('connect', () => {
    console.log('[QUVI] WebSocket 연결됨');
});

socket.on('disconnect', () => {
    console.log('[QUVI] WebSocket 연결 끊김');
});

// ─── 실시간 업데이트 수신 ───
socket.on('status_update', (data) => {
    updateStatus(data.status);
    updateStats(data.stats);
    if (data.latest_inspection) {
        updateLatestInspection(data.latest_inspection);
    }
});

// ─── 시스템 상태 업데이트 ───
function updateStatus(status) {
    const stateEl = document.getElementById('systemState');
    const stateText = document.getElementById('stateText');
    const errorMsg = document.getElementById('errorMsg');

    const state = status.current_state || 'IDLE';
    stateText.textContent = state;

    // 상태별 스타일
    stateEl.className = 'status-indicator';
    if (state === 'IDLE') stateEl.classList.add('idle');
    else if (state === 'ERROR') stateEl.classList.add('error');
    else if (state === 'DONE') stateEl.classList.add('done');
    else stateEl.classList.add('running');

    // 에러 메시지
    if (status.error_message) {
        errorMsg.textContent = status.error_message;
        errorMsg.style.display = 'block';
    } else {
        errorMsg.style.display = 'none';
    }

    // 통계 카드 (status에서 직접 받는 값)
    document.getElementById('statTotal').textContent = status.total_objects;
    document.getElementById('statProcessed').textContent = status.processed_count;

    // 서브시스템
    setSubsystem('subYolo', status.yolo_ready);
    setSubsystem('subGrasp', status.grasp_ready);
    setSubsystem('subInspect', status.inspect_ready);
    setSubsystem('subMotor', status.motor_ready);
}

function setSubsystem(id, online) {
    const el = document.getElementById(id);
    el.className = 'subsystem-dot ' + (online ? 'online' : 'offline');
}

// ─── 통계 업데이트 ───
function updateStats(stats) {
    if (!stats) return;

    document.getElementById('statPass').textContent = stats.passed;
    document.getElementById('statFail').textContent = stats.failed;
    document.getElementById('statPassRate').textContent = stats.pass_rate.toFixed(1) + '%';
    document.getElementById('legendPass').textContent = stats.passed;
    document.getElementById('legendFail').textContent = stats.failed;
    document.getElementById('donutRate').textContent = stats.pass_rate.toFixed(1) + '%';

    // 도넛 차트 업데이트
    drawDonut(stats.passed, stats.failed);
}

// ─── 최신 검사 결과 ───
function updateLatestInspection(result) {
    document.getElementById('noResult').style.display = 'none';
    document.getElementById('latestResult').style.display = 'block';

    // 배지
    const badge = document.getElementById('latestBadge');
    badge.style.display = 'inline-block';
    if (result.passed) {
        badge.textContent = 'PASS';
        badge.className = 'result-badge pass';
    } else {
        badge.textContent = 'FAIL';
        badge.className = 'result-badge fail';
    }

    // 메트릭
    setMetric('mSolidity', result.solidity, 0.85, 1.0);
    setMetric('mAreaRatio', result.area_ratio, 0.90, 1.10);
    setMetricInt('mHoleCount', result.hole_count, 0, 2);
    setMetric('mHoleArea', result.hole_area_ratio, 0, 0.05);
    setMetricInverse('mTexture', result.texture_variance, 500);
    document.getElementById('mTime').textContent = result.inspection_time_sec.toFixed(2) + 's';

    // SSIM 바
    renderSsimBars(result.ssim_scores);

    // 히스토리 테이블 업데이트
    addHistoryRow(result);
}

function setMetric(id, value, min, max) {
    const el = document.getElementById(id);
    const v = parseFloat(value);
    el.textContent = v.toFixed(3);
    if (v >= min && v <= max) {
        el.className = 'metric-value ok';
    } else if (Math.abs(v - min) < 0.05 || Math.abs(v - max) < 0.05) {
        el.className = 'metric-value warn';
    } else {
        el.className = 'metric-value bad';
    }
}

function setMetricInt(id, value, min, max) {
    const el = document.getElementById(id);
    el.textContent = value;
    if (value >= min && value <= max) {
        el.className = 'metric-value ok';
    } else {
        el.className = 'metric-value bad';
    }
}

function setMetricInverse(id, value, threshold) {
    const el = document.getElementById(id);
    const v = parseFloat(value);
    el.textContent = v.toFixed(1);
    if (v <= threshold) {
        el.className = 'metric-value ok';
    } else {
        el.className = 'metric-value bad';
    }
}

// ─── SSIM 바 렌더링 ───
function renderSsimBars(scores) {
    const container = document.getElementById('ssimBars');
    const angles = [0, 90, 180, 270];
    const threshold = 0.85;

    let html = '';
    for (let i = 0; i < angles.length; i++) {
        const score = scores[i] || 0;
        const pct = Math.min(score * 100, 100);
        let cls = 'ok';
        if (score < threshold) cls = 'bad';
        else if (score < 0.90) cls = 'warn';

        html += `
        <div class="ssim-bar-row">
            <span class="ssim-angle">${angles[i]}°</span>
            <div class="ssim-bar-track">
                <div class="ssim-bar-fill ${cls}" style="width:${pct}%"></div>
            </div>
            <span class="ssim-value" style="color:var(--accent-${cls === 'ok' ? 'green' : cls === 'warn' ? 'yellow' : 'red'})">${score.toFixed(3)}</span>
        </div>`;
    }
    container.innerHTML = html;
}

// ─── 히스토리 테이블 ───
let historyData = [];

function addHistoryRow(result) {
    historyData.unshift(result);
    if (historyData.length > 50) historyData.pop();

    document.getElementById('historyCount').textContent = historyData.length + '건';

    const tbody = document.getElementById('historyBody');
    const avgSsim = result.ssim_scores.length > 0
        ? (result.ssim_scores.reduce((a, b) => a + b, 0) / result.ssim_scores.length)
        : 0;

    let rows = '';
    historyData.forEach((r, idx) => {
        const avg = r.ssim_scores.length > 0
            ? (r.ssim_scores.reduce((a, b) => a + b, 0) / r.ssim_scores.length)
            : 0;
        const badgeCls = r.passed ? 'pass' : 'fail';
        const badgeText = r.passed ? 'PASS' : 'FAIL';
        const time = r.timestamp ? new Date(r.timestamp).toLocaleTimeString('ko-KR') : '-';

        rows += `<tr>
            <td>${r.object_index}</td>
            <td class="time-cell">${time}</td>
            <td><span class="result-badge ${badgeCls}">${badgeText}</span></td>
            <td style="font-family:var(--font-mono)">${avg.toFixed(3)}</td>
            <td style="font-family:var(--font-mono)">${r.solidity.toFixed(3)}</td>
            <td style="font-family:var(--font-mono)">${r.hole_count}</td>
            <td style="font-family:var(--font-mono)">${r.texture_variance.toFixed(1)}</td>
            <td style="font-family:var(--font-mono)">${r.inspection_time_sec.toFixed(2)}s</td>
            <td style="color:var(--accent-red);font-size:12px">${r.fail_reason || '-'}</td>
        </tr>`;
    });

    tbody.innerHTML = rows;
}

// ─── 도넛 차트 (Canvas) ───
function drawDonut(pass, fail) {
    const canvas = document.getElementById('donutChart');
    const ctx = canvas.getContext('2d');
    const total = pass + fail;
    const cx = 70, cy = 70, r = 55, lw = 14;

    ctx.clearRect(0, 0, 140, 140);

    if (total === 0) {
        // 빈 링
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = lw;
        ctx.stroke();
        return;
    }

    const passAngle = (pass / total) * Math.PI * 2;
    const startAngle = -Math.PI / 2;

    // PASS (초록)
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + passAngle);
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.stroke();

    // FAIL (빨강)
    if (fail > 0) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, startAngle + passAngle, startAngle + Math.PI * 2);
        ctx.strokeStyle = '#f85149';
        ctx.lineWidth = lw;
        ctx.lineCap = 'round';
        ctx.stroke();
    }
}

// ─── 제어 API ───
async function sendCommand(cmd) {
    if (cmd === 'estop') {
        if (!confirm('비상정지를 실행합니까?')) return;
    }
    try {
        const res = await fetch(`/api/command/${cmd}`, { method: 'POST' });
        const data = await res.json();
        console.log(`[QUVI] 명령 전송: ${cmd}`, data);
    } catch (e) {
        console.error('[QUVI] 명령 실패:', e);
    }
}

async function triggerDetection() {
    await fetch('/api/trigger/detection', { method: 'POST' });
}

async function triggerInspection() {
    await fetch('/api/trigger/inspection', { method: 'POST' });
}

// ─── 시계 ───
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('ko-KR', { hour12: false });
    const dateStr = now.toLocaleDateString('ko-KR', {
        year: 'numeric', month: '2-digit', day: '2-digit'
    });
    document.getElementById('clock').textContent = `${dateStr} ${timeStr}`;
}

setInterval(updateClock, 1000);
updateClock();

// ─── 초기 데이터 로드 ───
async function loadInitialData() {
    try {
        const [statusRes, historyRes] = await Promise.all([
            fetch('/api/status'),
            fetch('/api/inspection/history'),
        ]);
        const status = await statusRes.json();
        const history = await historyRes.json();

        updateStatus(status);

        if (history.length > 0) {
            historyData = history.reverse().slice(0, 50);
            const last = historyData[0];
            updateLatestInspection(last);

            const pass = historyData.filter(h => h.passed).length;
            const fail = historyData.length - pass;
            updateStats({
                passed: pass,
                failed: fail,
                pass_rate: historyData.length > 0 ? (pass / historyData.length * 100) : 0,
            });
        }
    } catch (e) {
        console.log('[QUVI] 초기 데이터 로드 실패 (서버 대기 중)');
    }
}

loadInitialData();

// 초기 도넛 차트 그리기
drawDonut(0, 0);
