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
    else if (state === 'DONE' || state === 'FINISHED') stateEl.classList.add('done');
    else stateEl.classList.add('running');

    // 진행 단계 라벨 (사람이 읽기 쉬운 한국어)
    updateStageLabel(state);

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

    // 2. 리니어 레일 위치 (Linear Rail)
    if (status.rail_position !== undefined) {
        const railPos = parseInt(status.rail_position);
        const stepsMap = [0, 1000, 1700, 2400];
        const namesMap = ['BED (D)', 'INSPECT (A)', 'PASS (B)', 'FAIL (C)'];
        
        const currentStep = stepsMap[railPos] !== undefined ? stepsMap[railPos] : 0;
        const currentName = namesMap[railPos] !== undefined ? namesMap[railPos] : 'UNKNOWN';
        
        // Carriage 위치 계산 (0스텝 -> 7.5%, 2400스텝 -> 92.5%)
        const carriage = document.getElementById('railCarriage');
        if (carriage) {
            const pct = (currentStep / 2400) * 85 + 7.5;
            carriage.style.left = `${pct}%`;
        }
        
        // 역 정보 텍스트 표시
        const infoEl = document.getElementById('railStepsText');
        if (infoEl) {
            infoEl.innerHTML = `<strong>${currentName}</strong> (${currentStep} steps)`;
        }
        
        // 구역 표시핀 하이라이트
        const stationIds = ['station_bed', 'station_inspect', 'station_pass', 'station_fail'];
        stationIds.forEach((id, idx) => {
            const el = document.getElementById(id);
            if (el) {
                if (idx === railPos) {
                    el.classList.add('active');
                } else {
                    el.classList.remove('active');
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

    // 4. FSM Flow 하이라이트
    if (status.current_state) {
        const state = status.current_state;
        const fsmNodes = ['INIT', 'IDLE', 'DETECTING', 'GRASPING', 'INSPECTING', 'SORTING', 'RELEASING', 'HOMING', 'TELEOPING', 'FINISHED'];
        
        let activeNode = null;
        for (const node of fsmNodes) {
            if (state.startsWith(node)) {
                activeNode = `fsm_${node}`;
                break;
            }
        }
        
        if (state === 'DONE') activeNode = 'fsm_FINISHED';
        
        // FSM 노드들의 active 클래스 토글
        fsmNodes.forEach(node => {
            const nodeId = `fsm_${node}`;
            const el = document.getElementById(nodeId);
            if (el) {
                if (nodeId === activeNode) {
                    el.classList.add('active');
                } else {
                    el.classList.remove('active');
                }
            }
        });
    }
}

// ─── 진행 단계 라벨 ───
const STAGE_LABELS = {
    'INIT': '시스템 초기화 중',
    'IDLE': '대기 중',
    'DETECTING': '출력물 탐지 중',
    'GRASPING': 'ACT 파지 진행 중',
    'INSPECTING': '품질 검사 중',
    'SORTING': '분류 이동 중',
    'RELEASING': '출력물 투하 중',
    'HOMING': '홈 복귀 중',
    'TELEOPING': '텔레오퍼레이션 동작 중',
    'FINISHED': '작업 완료',
    'DONE': '작업 완료',
    'ERROR': '오류 발생 — 확인 필요',
};

function updateStageLabel(state) {
    const el = document.getElementById('stageText');
    if (!el) return;
    // "DETECTING_WAIT" 같은 세부 상태도 접두어로 매칭
    let label = STAGE_LABELS[state];
    if (!label) {
        const prefix = Object.keys(STAGE_LABELS).find(k => state.startsWith(k));
        label = prefix ? STAGE_LABELS[prefix] : state;
    }
    el.textContent = label;
    el.className = 'stage-pill';
    if (state === 'ERROR') el.classList.add('error');
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

// ─── 탭 전환 함수 ───
function switchTab(tabName) {
    // 탭 판넬 전환
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    const targetTab = document.getElementById(`tab-${tabName}`);
    if (targetTab) targetTab.classList.add('active');

    // 네비게이션 아이템 활성화
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const targetMenu = document.getElementById(`menu-${tabName}`);
    if (targetMenu) targetMenu.classList.add('active');

    console.log(`[QUVI] 탭 전환: ${tabName}`);
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
