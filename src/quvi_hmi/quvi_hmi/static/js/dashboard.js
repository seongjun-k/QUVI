/* ═══════════════════════════════════════════
   QUVI HMI Dashboard — Client-Side Logic
   WebSocket 실시간 업데이트 + UI 렌더링
   ═══════════════════════════════════════════ */

// ─── WebSocket 연결 ───
const socket = io();

function setConnBadge(online) {
    const badge = document.getElementById('connBadge');
    const text = document.getElementById('connText');
    if (!badge || !text) return;
    badge.className = 'conn-badge ' + (online ? 'online' : 'offline');
    text.textContent = online ? '서버 연결됨' : '서버 연결 끊김';
}

socket.on('connect', () => {
    console.log('[QUVI] WebSocket 연결됨');
    setConnBadge(true);
});

socket.on('disconnect', () => {
    console.log('[QUVI] WebSocket 연결 끊김');
    setConnBadge(false);
});

// ─── 실시간 업데이트 수신 ───
socket.on('status_update', (data) => {
    updateStatus(data.status);
    updateStats(data.stats);
    if (data.latest_inspection) {
        updateLatestInspection(data.latest_inspection);
    }
    // 수동 제어 패널 UI 실시간 갱신
    updateManualControlPanel(data.status);
});

// ─── 시스템 상태 업데이트 ───
function updateStatus(status) {
    const stateEl = document.getElementById('systemState');
    const stateText = document.getElementById('stateText');
    const errorMsg = document.getElementById('errorMsg');

    const state = status.current_state || 'IDLE';
    stateText.textContent = state;

    stateEl.className = 'status-indicator';
    if (state === 'IDLE') stateEl.classList.add('idle');
    else if (state === 'ERROR' || state === 'ESTOP') stateEl.classList.add('error');
    else if (state === 'DONE' || state === 'FINISHED') stateEl.classList.add('done');
    else stateEl.classList.add('running');

    // 진행 단계 라벨 (사람이 읽기 쉬운 한국어)
    updateStageLabel(state);
    if (status.error_message) {
        errorMsg.textContent = status.error_message;
        errorMsg.style.display = 'block';
    } else {
        errorMsg.style.display = 'none';
    }

    document.getElementById('statTotal').textContent = status.total_objects;
    document.getElementById('statProcessed').textContent = status.processed_count;

    setSubsystem('subGrasp', status.grasp_ready);
    setSubsystem('subInspect', status.inspect_ready);
    setSubsystem('subMotor', status.motor_ready);

    // 텔레오퍼레이션 상태 동기화
    const teleopToggle = document.getElementById('teleopToggle');
    const teleopBadge = document.getElementById('teleopStateBadge');

    if (teleopToggle && teleopBadge) {
        const isActive = status.teleop_active || (state === 'TELEOPING');
        // /robot/status 가 "텔레옵 에러"를 알리면 에러 상태로 구분 표시
        const isTeleopError = !!(status.error_message &&
            /텔레옵|teleop|리더/i.test(status.error_message));

        teleopToggle.checked = isActive;
        setTeleopBadge(isActive ? 'on' : (isTeleopError ? 'error' : 'off'));
    }

    // ─── 시스템 상태 탭 업데이트 ───
    updateStatusTab(status);
}

// ─── 시스템 상태 탭 상세 업데이트 ───
function updateStatusTab(status) {
    // 1. 관절 상태 (Joint States)
    if (status.joint_positions && Array.isArray(status.joint_positions)) {
        status.joint_positions.forEach((rad, idx) => {
            if (idx >= 6) return;
            
            const valEl = document.getElementById(`j${idx}_val`);
            const barEl = document.getElementById(`j${idx}_bar`);
            if (!valEl || !barEl) return;
            
            if (idx === 5) {
                // 그리퍼 (ID 6, 1800 ~ 2300 ticks)
                const ticks = Math.round((rad * 4095.0) / (2 * Math.PI));
                valEl.textContent = `${ticks} ticks`;
                
                // 퍼센트 계산
                let pct = ((ticks - 1800) / (2300 - 1800)) * 100;
                pct = Math.max(0, Math.min(100, pct));
                barEl.style.width = `${pct}%`;
                
                // 그리퍼 상태 색상 힌트
                if (ticks <= 1850) {
                    barEl.className = "joint-bar-fill gripper-bar closed";
                } else if (ticks >= 2250) {
                    barEl.className = "joint-bar-fill gripper-bar open";
                } else {
                    barEl.className = "joint-bar-fill gripper-bar";
                }
            } else {
                // 일반 관절 (0 ~ 360 -> -180 ~ +180)
                let deg = (rad * 180 / Math.PI) - 180;
                // 주기화 (-180 ~ 180 범위로 안착)
                while (deg < -180) deg += 360;
                while (deg > 180) deg -= 360;
                
                valEl.textContent = `${deg.toFixed(1)}°`;
                const pct = ((deg + 180) / 360) * 100;
                barEl.style.width = `${pct}%`;
            }
        });
        
        // 동기화 시간 업데이트
        const syncTimeEl = document.getElementById('jointSyncTime');
        if (syncTimeEl) {
            syncTimeEl.textContent = '실시간 (30Hz)';
            syncTimeEl.style.color = 'var(--accent-green)';
        }
    }

    // 2. 리니어 레일 위치 (Linear Rail - API 응답의 rail_station_map 기반 동적화)
    if (status.rail_position !== undefined) {
        const railPos = parseInt(status.rail_position);
        const stationMap = status.rail_station_map || [
            { name: 'INSPECT (A)', steps: 1000 },
            { name: 'PASS (B)',    steps: 2000 },
            { name: 'FAIL (C)',    steps: 10000 },
            { name: 'BED (D)',      steps: 30500 },
        ];
        const maxSteps = Math.max(...stationMap.map(s => s.steps)) || 30500;

        // railPos(실제 스텝 값)과 가장 가까운 스테이션 탐색 (허용 오차 100 스텝)
        let station = stationMap.find(s => Math.abs(s.steps - railPos) < 100);
        if (!station) {
            station = { name: '이동 중...', steps: railPos };
        }

        // Carriage 위치 계산
        const carriage = document.getElementById('railCarriage');
        if (carriage) {
            const pct = (station.steps / maxSteps) * 85 + 7.5;
            carriage.style.left = `${pct}%`;
        }

        // 역 정보 텍스트 표시
        const infoEl = document.getElementById('railStepsText');
        if (infoEl) {
            infoEl.innerHTML = `<strong>${station.name}</strong> (${station.steps} steps)`;
        }

        // 구역 표시핀 하이라이트 및 스텝 값 동적 반영
        const matchedIdx = stationMap.findIndex(s => s.name === station.name);
        const stationIds = ['station_inspect', 'station_pass', 'station_fail', 'station_bed'];
        stationIds.forEach((id, idx) => {
            const el = document.getElementById(id);
            if (el) {
                el.classList.toggle('active', idx === matchedIdx);
                // 스텝 숫자 동적 갱신
                const stepEl = el.querySelector('.station-step');
                if (stepEl && stationMap[idx]) {
                    stepEl.textContent = stationMap[idx].steps;
                }
            }
        });
    }

    // 3. 턴테이블 회전 각도 (Turntable)
    if (status.turntable_angle !== undefined) {
        const angle = parseInt(status.turntable_angle);
        
        const dial = document.getElementById('turntableDial');
        if (dial) {
            dial.style.transform = `rotate(${angle}deg)`;
        }
        
        const textEl = document.getElementById('turntableAngleText');
        if (textEl) {
            textEl.textContent = `${angle}°`;
        }
    }

    // 4. FSM Flow 하이라이트 (ERROR / ESTOP 포함)
    if (status.current_state) {
        const state = status.current_state;
        const fsmNodes = [
            'INIT', 'IDLE', 'GRASPING', 'INSPECTING',
            'SORTING', 'RELEASING', 'HOMING', 'TELEOPING', 'FINISHED',
            'ERROR', 'ESTOP',
        ];

        let activeNode = null;
        // ESTOP 여부 검사 (ERROR 상태이면서 에러 메시지에 ESTOP/비상정지 포함 시)
        if (state === 'ERROR' && status.error_message && /ESTOP|비상정지/i.test(status.error_message)) {
            activeNode = 'fsm_ESTOP';
        } else {
            for (const node of fsmNodes) {
                if (state === node || state.startsWith(node)) {
                    activeNode = `fsm_${node}`;
                    break;
                }
            }
        }
        
        if (state === 'DONE') activeNode = 'fsm_FINISHED';
        
        // FSM 노드들의 active 클래스 토글
        fsmNodes.forEach(node => {
            const nodeId = `fsm_${node}`;
            const el = document.getElementById(nodeId);
            if (el) {
                el.classList.toggle('active', nodeId === activeNode);
            }
        });
    }
}

// ─── 진행 단계 라벨 ───
const STAGE_LABELS = {
    'INIT':       '시스템 초기화 중',
    'IDLE':       '대기 중',
    'GRASPING':   'ACT 파지 진행 중',
    'INSPECTING': '품질 검사 중',
    'SORTING':    '분류 이동 중',
    'RELEASING':  '출력물 투하 중',
    'HOMING':     '홈 복귀 중',
    'TELEOPING':  '텔레오퍼레이션 동작 중',
    'FINISHED':   '작업 완료',
    'DONE':       '작업 완료',
    'ERROR':      '오류 발생 — 확인 필요',
    'ESTOP':      '비상 정지 — E-STOP 활성',
};

function updateStageLabel(state) {
    const el = document.getElementById('stageText');
    if (!el) return;
    // "GRASPING_WAIT" 같은 세부 상태도 접두어로 매칭
    let label = STAGE_LABELS[state];
    if (!label) {
        const prefix = Object.keys(STAGE_LABELS).find(k => state.startsWith(k));
        label = prefix ? STAGE_LABELS[prefix] : state;
    }
    el.textContent = label;
    el.className = 'stage-pill';
    if (state === 'ERROR' || state === 'ESTOP') el.classList.add('error');
    else if (state === 'TELEOPING') el.classList.add('teleop');
    else if (state === 'IDLE' || state === 'INIT') el.classList.add('idle');
    else if (state === 'FINISHED' || state === 'DONE') el.classList.add('done');
    else el.classList.add('running');
}

// ─── 텔레옵 배지/힌트 (on | off | error) ───
function setTeleopBadge(mode) {
    const badge = document.getElementById('teleopStateBadge');
    const hint = document.getElementById('teleopHint');
    if (!badge) return;

    if (mode === 'on') {
        badge.textContent = 'ON';
        badge.className = 'teleop-state-badge on';
        if (hint) hint.textContent = '리더 암을 움직이면 팔로워가 30Hz로 따라갑니다.';
    } else if (mode === 'error') {
        badge.textContent = 'ERROR';
        badge.className = 'teleop-state-badge error';
        if (hint) hint.textContent = '리더 암 연결/포트 오류입니다. E-STOP 후 연결을 확인하세요.';
    } else {
        badge.textContent = 'OFF';
        badge.className = 'teleop-state-badge off';
        if (hint) hint.textContent = '리더 암을 연결한 뒤 토글을 켜세요.';
    }
}

function setSubsystem(id, online) {
    const el = document.getElementById(id);
    if (el) el.className = 'subsystem-dot ' + (online ? 'online' : 'offline');
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
    drawDonut(stats.passed, stats.failed);
}

// ─── 최신 검사 결과 ───
function updateLatestInspection(result) {
    document.getElementById('noResult').style.display = 'none';
    document.getElementById('latestResult').style.display = 'block';

    const badge = document.getElementById('latestBadge');
    badge.style.display = 'inline-block';
    if (result.passed) {
        badge.textContent = 'PASS';
        badge.className = 'result-badge pass';
    } else {
        badge.textContent = 'FAIL';
        badge.className = 'result-badge fail';
    }

    setMetric('mSolidity', result.solidity, 0.85, 1.0);
    setMetric('mAreaRatio', result.area_ratio, 0.90, 1.10);
    setMetricInt('mHoleCount', result.hole_count, 0, 2);
    setMetric('mHoleArea', result.hole_area_ratio, 0, 0.05);
    setMetricInverse('mTexture', result.texture_variance, 500);
    document.getElementById('mTime').textContent = result.inspection_time_sec.toFixed(2) + 's';

    renderSsimBars(result.ssim_scores);
    addHistoryRow(result);
    // 검사 상세 탭 업데이트
    updateInspectionDetailTab(result);
}

// ─── 검사 상세 내역 탭 업데이트 ───
function updateInspectionDetailTab(result) {
    const noResult = document.getElementById('noResult');
    const detailContainer = document.getElementById('detailContainer');
    const badge = document.getElementById('detailBadge');
    
    noResult.style.display = 'none';
    detailContainer.style.display = 'block';

    badge.style.display = 'inline-block';
    if (result.passed) {
        badge.textContent = 'PASS';
        badge.className = 'result-badge pass';
    } else {
        badge.textContent = 'FAIL';
        badge.className = 'result-badge fail';
    }
    const failReason = document.getElementById('detailFailReason');
    if (result.passed) {
        failReason.textContent = '정상 (None)';
        failReason.style.color = 'var(--accent-green)';
        failReason.style.background = 'rgba(63, 185, 80, 0.1)';
        failReason.style.borderColor = 'rgba(63, 185, 80, 0.2)';
    } else {
        failReason.textContent = result.fail_reason || '불명 (Unknown)';
        failReason.style.color = 'var(--accent-red)';
        failReason.style.background = 'rgba(248,81,73,0.1)';
        failReason.style.borderColor = 'rgba(248,81,73,0.2)';
    }
    // 테이블 값 채우기 및 판정
    function fillTableRow(valId, evalId, val, min, max, isInt = false, isInverse = false) {
        const valEl = document.getElementById(valId);
        const evalEl = document.getElementById(evalId);
        
        let formattedVal = isInt ? val : parseFloat(val).toFixed(3);
        if (valId === 'detTexture') formattedVal = parseFloat(val).toFixed(1);
        valEl.textContent = formattedVal;
        
        let passed = false;
        if (isInverse) {
            passed = val <= max; // max is used as threshold here
        } else {
            passed = val >= min && val <= max;
        }
        
        if (passed) {
            evalEl.innerHTML = '<span style="color: var(--accent-green); font-weight: bold;">OK</span>';
            valEl.style.color = 'var(--text-primary)';
        } else {
            evalEl.innerHTML = '<span style="color: var(--accent-red); font-weight: bold;">FAIL</span>';
            valEl.style.color = 'var(--accent-red)';
        }
    }
    fillTableRow('detSolidity', 'detSolidityEval', result.solidity, 0.85, 1.0);
    fillTableRow('detAreaRatio', 'detAreaRatioEval', result.area_ratio, 0.90, 1.10);
    fillTableRow('detHoleCount', 'detHoleCountEval', result.hole_count, 0, 2, true);
    fillTableRow('detHoleArea', 'detHoleAreaEval', result.hole_area_ratio, 0, 0.05);
    fillTableRow('detTexture', 'detTextureEval', result.texture_variance, 0, 500, false, true);
    
    // SSIM
    const ssimContainer = document.getElementById('detailSsimBars');
    const angles = [0, 90, 180, 270];
    const threshold = 0.85;
    let html = '';
    for (let i = 0; i < angles.length; i++) {
        const score = result.ssim_scores[i] || 0;
        const pct = Math.min(score * 100, 100);
        let cls = 'ok';
        let evalText = '<span style="color: var(--accent-green); font-weight: bold;">OK</span>';
        if (score < threshold) {
            cls = 'bad';
            evalText = '<span style="color: var(--accent-red); font-weight: bold;">FAIL</span>';
        }
        else if (score < 0.90) cls = 'warn';
        
        html += `
        <div class="ssim-bar-row" style="margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
            <span class="ssim-angle" style="width: 60px;">${angles[i]}°</span>
            <div class="ssim-bar-track" style="flex: 1; margin: 0 10px;">
                <div class="ssim-bar-fill ${cls}" style="width:${pct}%"></div>
            </div>
            <span class="ssim-value" style="color:var(--accent-${cls === 'ok' ? 'green' : cls === 'warn' ? 'yellow' : 'red'}); width: 60px; text-align: right; margin-right: 15px;">${score.toFixed(3)}</span>
            <span style="width: 40px; text-align: center;">${evalText}</span>
        </div>`;
    }
    ssimContainer.innerHTML = html;
}

function setMetric(id, value, min, max) {
    const el = document.getElementById(id);
    const v = parseFloat(value);
    el.textContent = v.toFixed(3);
    if (v >= min && v <= max) el.className = 'metric-value ok';
    else if (Math.abs(v - min) < 0.05 || Math.abs(v - max) < 0.05) el.className = 'metric-value warn';
    else el.className = 'metric-value bad';
}

function setMetricInt(id, value, min, max) {
    const el = document.getElementById(id);
    el.textContent = value;
    el.className = 'metric-value ' + (value >= min && value <= max ? 'ok' : 'bad');
}

function setMetricInverse(id, value, threshold) {
    const el = document.getElementById(id);
    const v = parseFloat(value);
    el.textContent = v.toFixed(1);
    el.className = 'metric-value ' + (v <= threshold ? 'ok' : 'bad');
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
        let cls = score < threshold ? 'bad' : (score < 0.90 ? 'warn' : 'ok');
        const scoreColor = cls === 'ok' ? 'green' : (cls === 'warn' ? 'yellow' : 'red');
        html += `
        <div class="ssim-bar-row">
            <span class="ssim-angle">${angles[i]}°</span>
            <div class="ssim-bar-track">
                <div class="ssim-bar-fill ${cls}" style="width:${pct}%"></div>
            </div>
            <span class="ssim-value" style="color:var(--accent-${scoreColor})">${score.toFixed(3)}</span>
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
    let rows = '';
    historyData.forEach((r) => {
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

// ─── 도넛 차트 ───
function drawDonut(pass, fail) {
    const canvas = document.getElementById('donutChart');
    const ctx = canvas.getContext('2d');
    const total = pass + fail;
    const cx = 70, cy = 70, r = 55, lw = 14;
    ctx.clearRect(0, 0, 140, 140);

    if (total === 0) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = lw;
        ctx.stroke();
        return;
    }

    const passAngle = (pass / total) * Math.PI * 2;
    const startAngle = -Math.PI / 2;

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + passAngle);
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.stroke();

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
    try {
        const res = await fetch(`/api/command/${cmd}`, { method: 'POST' });
        const data = await res.json();
        console.log(`[QUVI] 명령 전송: ${cmd}`, data);
    } catch (e) {
        console.error('[QUVI] 명령 실패:', e);
    }
}

async function triggerDetection() {
    const res = await fetch('/api/trigger/detection', { method: 'POST' });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        console.error('[QUVI] 수동 탐지 거부:', data.error || res.status);
    }
}

async function triggerInspection() {
    const res = await fetch('/api/trigger/inspection', { method: 'POST' });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        console.error('[QUVI] 수동 검사 거부:', data.error || res.status);
    }
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
drawDonut(0, 0);

// ─── 탭 전환 ───
function switchTab(tabName) {
    // 탭 판넬 전환
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    const targetTab = document.getElementById(`tab-${tabName}`);
    if (targetTab) targetTab.classList.add('active');

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const targetMenu = document.getElementById(`menu-${tabName}`);
    if (targetMenu) targetMenu.classList.add('active');
}

// ─── 텔레오퍼레이션 토글 API 호출 ───
async function toggleTeleop(enable) {
    const action = enable ? 'on' : 'off';
    console.log(`[QUVI] 텔레오퍼레이션 요청: ${action}`);
    
    try {
        const res = await fetch(`/api/teleop/${action}`, { method: 'POST' });
        const data = await res.json();
        console.log(`[QUVI] 텔레오퍼레이션 응답:`, data);
        
        // UI 즉시 업데이트 (배지 색상 변경) — 이후 status_update가 최종 동기화
        setTeleopBadge(enable ? 'on' : 'off');
    } catch (e) {
        console.error('[QUVI] 텔레오퍼레이션 토글 실패:', e);
        // 실패 시 스위치 롤백 + 에러 표시
        const teleopToggle = document.getElementById('teleopToggle');
        if (teleopToggle) teleopToggle.checked = !enable;
        setTeleopBadge('error');
    }
}

// ═══════════════════════════════════════════
// 수동 제어 패널 — 레일 / 턴테이블 / LED
// ═══════════════════════════════════════════

// 입력값 범위 클램프
function clampInput(el, min, max) {
    const v = parseFloat(el.value);
    if (!isNaN(v)) {
        if (v < min) el.value = min;
        if (v > max) el.value = max;
    }
}

// 레일 프리셋 칩
function setRailPreset(mm) {
    document.getElementById('railMmInput').value = mm;
}

// 턴테이블 프리셋 칩
function setTurnPreset(deg) {
    document.getElementById('turnAngleInput').value = deg;
}

// 레일 이동 실행
async function execRailMove() {
    const input = document.getElementById('railMmInput');
    const btn   = document.getElementById('railExecBtn');
    const mm    = parseFloat(input.value);
    if (isNaN(mm) || mm < 0 || mm > 420) {
        alert('레일 위치를 0~420mm 범위로 입력해주세요.');
        return;
    }
    btn.disabled = true;
    btn.textContent = '이동 중…';
    try {
        const res  = await fetch('/api/rail/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mm })
        });
        const data = await res.json();
        if (data.ok) {
            console.log(`[QUVI] 레일 명령 완료: ${mm}mm`);
            document.getElementById('railCurrentDisp').textContent = `명령됨: ${mm} mm`;
        } else {
            alert(`레일 오류: ${data.error}`);
        }
    } catch (e) {
        console.error('[QUVI] 레일 이동 실패:', e);
        alert('레일 명령 전송 실패');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> 이동';
    }
}

// 턴테이블 회전 실행
async function execTurnMove() {
    const input = document.getElementById('turnAngleInput');
    const btn   = document.getElementById('turnExecBtn');
    const angle = parseInt(input.value);
    if (isNaN(angle) || angle < 0 || angle > 360) {
        alert('각도를 0~360° 범위로 입력해주세요.');
        return;
    }
    btn.disabled = true;
    btn.textContent = '회전 중…';
    try {
        const res  = await fetch('/api/turntable/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ angle })
        });
        const data = await res.json();
        if (data.ok) {
            console.log(`[QUVI] 턴테이블 명령 완료: ${angle}°`);
            document.getElementById('turnCurrentDisp').textContent = `명령됨: ${angle}°`;
        } else {
            alert(`턴테이블 오류: ${data.error}`);
        }
    } catch (e) {
        console.error('[QUVI] 턴테이블 이동 실패:', e);
        alert('턴테이블 명령 전송 실패');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> 회전';
    }
}

// LED 토글
async function toggleLed(on) {
    try {
        const action = on ? 'on' : 'off';
        const res  = await fetch(`/api/led/${action}`, { method: 'POST' });
        const data = await res.json();
        if (data.ok) {
            _applyLedUi(data.led);
        } else {
            // 롤백
            document.getElementById('ledToggle').checked = !on;
        }
    } catch (e) {
        console.error('[QUVI] LED 토글 실패:', e);
        document.getElementById('ledToggle').checked = !on;
    }
}

// LED UI 상태 반영 (WebSocket 업데이트에서도 호출)
function _applyLedUi(on) {
    const indicator = document.getElementById('ledIndicator');
    const stateText = document.getElementById('ledStateText');
    const toggle    = document.getElementById('ledToggle');
    if (!indicator) return;
    if (on) {
        indicator.classList.add('on');
        // [fix] ledStateText null 체크
        if (stateText) stateText.textContent = 'ON';
    } else {
        indicator.classList.remove('on');
        // [fix] ledStateText null 체크
        if (stateText) stateText.textContent = 'OFF';
    }
    if (toggle) toggle.checked = !!on;
}

// WebSocket status_update 시 수동 제어 패널 UI 갱신
// (기존 onStatusUpdate 훅에 병합 — 기존 JS 파일의 소켓 이벤트 핸들러에서 호출되도록)
function updateManualControlPanel(status) {
    // 레일 현재 표시
    const railDisp = document.getElementById('railCurrentDisp');
    if (railDisp && status.rail_position !== undefined) {
        railDisp.textContent = `현재: ${parseFloat(status.rail_position).toFixed(2)} mm`;
    }
    // 턴테이블 현재 표시
    const turnDisp = document.getElementById('turnCurrentDisp');
    if (turnDisp && status.turntable_angle !== undefined) {
        turnDisp.textContent = `현재: ${status.turntable_angle}°`;
    }
    // LED 상태
    if (status.led_state !== undefined) {
        _applyLedUi(status.led_state);
    }
}
