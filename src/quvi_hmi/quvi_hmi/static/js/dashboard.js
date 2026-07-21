/* ═══════════════════════════════════════════
   QUVI HMI Dashboard — Client-Side Logic
   WebSocket 실시간 업데이트 + UI 렌더링
   ═══════════════════════════════════════════ */

/* ═══════════════════════════════════════════
   [A] 자주 수정하는 영역 — 표시 상수·라벨
   ═══════════════════════════════════════════ */

// ─── 진행 단계 라벨 ───
// SSoT: 발원지 main_orchestrator_node.py FsmState — 변경 시 hmi_node.py FSM_DISPLAY_STATES·dashboard.html fsm_* 노드와 동시 수정
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
    'ERROR':      '오류 발생 — 확인 필요',
    'ESTOP':      '비상 정지 — E-STOP 활성',
};

// ─── FSM Flow 노드 목록 ───
// SSoT: 발원지 main_orchestrator_node.py FsmState — 변경 시 hmi_node.py FSM_DISPLAY_STATES·dashboard.html fsm_* 노드와 동시 수정
const FSM_NODES = [
    'INIT', 'IDLE', 'GRASPING', 'INSPECTING',
    'SORTING', 'RELEASING', 'HOMING', 'TELEOPING', 'FINISHED',
    'ERROR', 'ESTOP',
];

// ─── 촬영 각도 (턴테이블 캡처 공통) ───
const CAPTURE_ANGLES = [0, 90, 180, 270];

// ─── 검사 판정 기준값 ───
// SSoT: 검사 노드 판정 기준 표시용 — dashboard.html 기준표 텍스트와 일치
const THRESHOLDS = {
    solidity: [0.85, 1.0],
    areaRatio: [0.80, 1.50],  // inspect_node feature_area_ratio_min/max와 일치 유지
    holeCount: [0, 0],        // inspect_node hole_count_max와 일치 유지 — 1개부터 FAIL
    holeAreaRatio: [0, 0.05],
    textureMax: 500,
};

// ─── 그리퍼 개폐 기준값 (ticks) ───
// SSoT: robot_control_node.py GRIPPER_OPEN/CLOSE 와 일치
const GRIPPER = { close: 1800, open: 2500, closedHint: 1850, openHint: 2250 };

/* ═══════════════════════════════════════════
   [B] 콘텐츠 렌더링·제어 로직
   ═══════════════════════════════════════════ */

// ─── HTML 이스케이프 (innerHTML 삽입 전 XSS/렌더 깨짐 방지) ───
function escapeHtml(str) {
    if (!str) return str;
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

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
    else if (state === 'FINISHED') stateEl.classList.add('done');
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
    _updateJointStates(status);
    _updateRailPosition(status);
    _updateTurntableAngle(status);
    _updateFsmHighlight(status);
}

// 1. 관절 상태 (Joint States)
function _updateJointStates(status) {
    if (status.joint_positions && Array.isArray(status.joint_positions)) {
        status.joint_positions.forEach((rad, idx) => {
            if (idx >= 6) return;

            const valEl = document.getElementById(`j${idx}_val`);
            const barEl = document.getElementById(`j${idx}_bar`);
            if (!valEl || !barEl) return;

            if (idx === 5) {
                // 그리퍼 (ID 6, 1800 ~ 2300 ticks)
                const ticks = Math.round((rad * 4096.0) / (2 * Math.PI));
                valEl.textContent = `${ticks} ticks`;

                // 퍼센트 계산
                let pct = ((ticks - GRIPPER.close) / (GRIPPER.open - GRIPPER.close)) * 100;
                pct = Math.max(0, Math.min(100, pct));
                barEl.style.width = `${pct}%`;

                // 그리퍼 상태 색상 힌트
                if (ticks <= GRIPPER.closedHint) {
                    barEl.className = "joint-bar-fill gripper-bar closed";
                } else if (ticks >= GRIPPER.openHint) {
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
            syncTimeEl.textContent = '실시간 (10Hz)';
            syncTimeEl.style.color = 'var(--accent-green)';
        }
    }
}

// 2. 리니어 레일 위치 (Linear Rail - API 응답의 rail_station_map 기반 동적화)
function _updateRailPosition(status) {
    if (status.rail_position !== undefined) {
        const railPos = parseFloat(status.rail_position);
        // SSoT: hmi_node.py RAIL_STATION_MAP 과 일치
        const stationMap = status.rail_station_map || [
            { name: 'INSPECT (A)', mm: 12.5 },
            { name: 'PASS (B)',    mm: 25.0 },
            { name: 'FAIL (C)',    mm: 125.0 },
            { name: 'BED (D)',      mm: 381.25 },
        ];
        const maxMm = Math.max(...stationMap.map(s => s.mm)) || 381.25;

        // 프리셋 칩 mm 값 동적 갱신 (SSoT: hmi_node.py RAIL_STATION_MAP)
        stationMap.forEach((s, idx) => {
            const chip = document.getElementById(`railPreset_${idx}`);
            if (chip) chip.dataset.mm = s.mm;
        });

        // railPos(마지막 명령 목표값, mm)과 가장 가까운 스테이션 탐색 (부동소수 오차 허용 0.25mm)
        let station = stationMap.find(s => Math.abs(s.mm - railPos) < 0.25);
        if (!station) {
            station = { name: '이동 중...', mm: railPos };
        }

        // Carriage 위치 계산
        const carriage = document.getElementById('railCarriage');
        if (carriage) {
            const pct = (station.mm / maxMm) * 85 + 7.5;
            carriage.style.left = `${pct}%`;
        }

        // 역 정보 텍스트 표시
        const infoEl = document.getElementById('railStepsText');
        if (infoEl) {
            infoEl.innerHTML = `<strong>${station.name}</strong> (${station.mm} mm)`;
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
                    stepEl.textContent = `${stationMap[idx].mm} mm`;
                }
            }
        });
    }
}

// 3. 턴테이블 회전 각도 (Turntable)
function _updateTurntableAngle(status) {
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
}

// 4. FSM Flow 하이라이트 (ERROR / ESTOP 포함)
function _updateFsmHighlight(status) {
    if (status.current_state) {
        const state = status.current_state;

        let activeNode = null;
        // ESTOP 여부 검사 (ERROR 상태이면서 에러 메시지에 ESTOP/비상정지 포함 시)
        if (state === 'ERROR' && status.error_message && /ESTOP|비상정지/i.test(status.error_message)) {
            activeNode = 'fsm_ESTOP';
        } else {
            for (const node of FSM_NODES) {
                if (state === node || state.startsWith(node)) {
                    activeNode = `fsm_${node}`;
                    break;
                }
            }
        }

        // FSM 노드들의 active 클래스 토글
        FSM_NODES.forEach(node => {
            const nodeId = `fsm_${node}`;
            const el = document.getElementById(nodeId);
            if (el) {
                el.classList.toggle('active', nodeId === activeNode);
            }
        });
    }
}

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
    else if (state === 'FINISHED') el.classList.add('done');
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
        if (hint) hint.textContent = '토글을 켜면 텔레오퍼레이션이 시작됩니다.';
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

    setMetric('mSolidity', result.solidity, THRESHOLDS.solidity[0], THRESHOLDS.solidity[1]);
    setMetric('mAreaRatio', result.area_ratio, THRESHOLDS.areaRatio[0], THRESHOLDS.areaRatio[1]);
    setMetricInt('mHoleCount', result.hole_count, THRESHOLDS.holeCount[0], THRESHOLDS.holeCount[1]);
    setMetric('mHoleArea', result.hole_area_ratio, THRESHOLDS.holeAreaRatio[0], THRESHOLDS.holeAreaRatio[1]);
    setMetricInverse('mTexture', result.texture_variance, THRESHOLDS.textureMax);
    document.getElementById('mTime').textContent = result.inspection_time_sec.toFixed(2) + 's';

    addHistoryRow(result);
    // 검사 상세 탭 업데이트
    updateInspectionDetailTab(result);
}

// ─── 검사 상세 내역 탭 업데이트 ───
function updateInspectionDetailTab(result) {
    const noResult = document.getElementById('detailNoResult');
    const detailContainer = document.getElementById('detailResult');
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
    _fillDetailTable(result);
}

// 테이블 값 채우기 및 판정 (단일 행)
function _fillTableRow(valId, evalId, val, min, max, isInt = false, isInverse = false) {
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

// 검사 상세 테이블 전체 채우기
function _fillDetailTable(result) {
    _fillTableRow('detSolidity', 'detSolidityEval', result.solidity, THRESHOLDS.solidity[0], THRESHOLDS.solidity[1]);
    _fillTableRow('detAreaRatio', 'detAreaRatioEval', result.area_ratio, THRESHOLDS.areaRatio[0], THRESHOLDS.areaRatio[1]);
    _fillTableRow('detHoleCount', 'detHoleCountEval', result.hole_count, THRESHOLDS.holeCount[0], THRESHOLDS.holeCount[1], true);
    _fillTableRow('detHoleArea', 'detHoleAreaEval', result.hole_area_ratio, THRESHOLDS.holeAreaRatio[0], THRESHOLDS.holeAreaRatio[1]);
    _fillTableRow('detTexture', 'detTextureEval', result.texture_variance, 0, THRESHOLDS.textureMax, false, true);
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

// ─── 히스토리 테이블 ───
let historyData = [];

function addHistoryRow(result) {
    // 0.1s 주기 status_update가 같은 최신 결과를 반복 전송 — timestamp로 중복 차단.
    // 이력 레코드 timestamp는 hmi_node에서 isoformat으로 생성되어 검사 1건당 고유
    if (historyData.length && historyData[0].timestamp === result.timestamp) return;
    historyData.unshift(result);
    if (historyData.length > 50) historyData.pop();
    renderHistoryTable();
}

function renderHistoryTable() {
    document.getElementById('historyCount').textContent = historyData.length + '건';

    const tbody = document.getElementById('historyBody');
    let rows = '';
    historyData.forEach((r) => {
        const badgeCls = r.passed ? 'pass' : 'fail';
        const badgeText = r.passed ? 'PASS' : 'FAIL';
        const time = r.timestamp ? new Date(r.timestamp).toLocaleTimeString('ko-KR') : '-';
        rows += `<tr>
            <td>${r.object_index}</td>
            <td class="time-cell">${time}</td>
            <td><span class="result-badge ${badgeCls}">${badgeText}</span></td>
            <td style="font-family:var(--font-mono)">${r.solidity.toFixed(3)}</td>
            <td style="font-family:var(--font-mono)">${r.hole_count}</td>
            <td style="font-family:var(--font-mono)">${r.texture_variance.toFixed(1)}</td>
            <td style="font-family:var(--font-mono)">${r.inspection_time_sec.toFixed(2)}s</td>
            <td style="color:var(--accent-red);font-size:12px">${escapeHtml(r.fail_reason) || '-'}</td>
        </tr>`;
    });
    tbody.innerHTML = rows;
}

// ─── 도넛 차트 (CSS conic-gradient) ───
function drawDonut(pass, fail) {
    const el = document.getElementById('donutChart');
    const total = pass + fail;
    if (total === 0) {
        el.style.background = 'conic-gradient(rgba(255, 255, 255, 0.06) 0% 100%)';
        return;
    }
    const passPct = (pass / total) * 100;
    el.style.background = `conic-gradient(var(--accent-green) 0% ${passPct}%, var(--accent-red) ${passPct}% 100%)`;
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
        if (stateText) stateText.textContent = 'ON';
    } else {
        indicator.classList.remove('on');
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

// ─── 캡쳐/촬영 시퀀스 공통 헬퍼 (기준 이미지 캡쳐 · 데이터셋 촬영 공유) ───
async function _startCaptureSequence({ url, body, statusElId, startBtnId, progressText, completeText, totalMs, errorLogPrefix }) {
    const statusEl = document.getElementById(statusElId);
    const startBtn = document.getElementById(startBtnId);

    if (statusEl) { statusEl.textContent = progressText; statusEl.style.color = 'var(--accent-green)'; }
    if (startBtn) startBtn.disabled = true;

    try {
        const res  = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.ok) {
            setTimeout(() => {
                if (statusEl) { statusEl.textContent = completeText; statusEl.style.color = 'var(--accent-green)'; }
                if (startBtn) startBtn.disabled = false;
            }, totalMs);
        } else {
            if (statusEl) { statusEl.textContent = '오류: ' + (data.error || '알 수 없음'); statusEl.style.color = 'var(--accent-red)'; }
            if (startBtn) startBtn.disabled = false;
        }
    } catch (e) {
        console.error(errorLogPrefix, e);
        if (statusEl) { statusEl.textContent = '네트워크 오류'; statusEl.style.color = 'var(--accent-red)'; }
        if (startBtn) startBtn.disabled = false;
    }
}

async function _stopCaptureSequence({ url, statusElId, startBtnId, stopText, errorLogPrefix }) {
    const statusEl = document.getElementById(statusElId);
    const startBtn = document.getElementById(startBtnId);
    try {
        await fetch(url, { method: 'POST' });
        if (statusEl) { statusEl.textContent = stopText; statusEl.style.color = 'var(--text-muted)'; }
        if (startBtn) startBtn.disabled = false;
    } catch (e) {
        console.error(errorLogPrefix, e);
    }
}

// ─── 기준 이미지 캡쳐 ───
async function startRefCapture() {
    const delay = parseFloat(document.getElementById('refCaptureDelay')?.value || 1.5);
    // delay × 4각도 후 완료로 표시
    const totalMs = delay * 4 * 1000 + 1000;
    await _startCaptureSequence({
        url: '/api/capture/reference/start',
        body: { angles: CAPTURE_ANGLES, delay_sec: delay },
        statusElId: 'refCaptureStatus',
        startBtnId: 'refCaptureStartBtn',
        progressText: '캡쳐 진행 중... (0° → 90° → 180° → 270°)',
        completeText: '캡쳐 완료 ✓',
        totalMs,
        errorLogPrefix: '[QUVI] 기준 캡쳐 시작 실패:',
    });
}

async function stopRefCapture() {
    await _stopCaptureSequence({
        url: '/api/capture/reference/stop',
        statusElId: 'refCaptureStatus',
        startBtnId: 'refCaptureStartBtn',
        stopText: '캡쳐 중단됨',
        errorLogPrefix: '[QUVI] 기준 캡쳐 중단 실패:',
    });
}

// ─── 데이터셋 촬영 (ML 정상품 수집) ───
async function startDatasetCapture() {
    const settleSec = parseFloat(document.getElementById('dsCaptureSettle')?.value || 2.0);
    const postSec = 1.0;
    let rounds = parseInt(document.getElementById('dataset_rounds')?.value, 10);
    if (!Number.isInteger(rounds) || rounds < 1 || rounds > 50) rounds = 1;
    // (settle+post) × 4각도 × rounds바퀴 후 완료로 표시
    const totalMs = (settleSec + postSec) * 4 * rounds * 1000 + 1000;
    await _startCaptureSequence({
        url: '/api/capture/dataset/start',
        body: { angles: CAPTURE_ANGLES, settle_sec: settleSec, post_capture_sec: postSec, rounds },
        statusElId: 'dsCaptureStatus',
        startBtnId: 'dsCaptureStartBtn',
        progressText: `촬영 진행 중... (0° → 90° → 180° → 270°) × ${rounds}바퀴`,
        completeText: '촬영 완료 ✓',
        totalMs,
        errorLogPrefix: '[QUVI] 데이터셋 촬영 시작 실패:',
    });
}

async function stopDatasetCapture() {
    await _stopCaptureSequence({
        url: '/api/capture/dataset/stop',
        statusElId: 'dsCaptureStatus',
        startBtnId: 'dsCaptureStartBtn',
        stopText: '촬영 중단됨',
        errorLogPrefix: '[QUVI] 데이터셋 촬영 중단 실패:',
    });
}

// ─── 검사 단독 테스트 (로봇/FSM 없이 검사 사이클만 실행) ───
async function startInspectionTest() {
    // LED 안정화 5초 + (턴테이블 대기 최대 5초 + 캡처 여유 2.5초) × 4각도 + 여유
    const totalMs = 5000 + 4 * 7500 + 1000;
    await _startCaptureSequence({
        url: '/api/inspection/test',
        body: { angles: CAPTURE_ANGLES },
        statusElId: 'inspectTestStatus',
        startBtnId: 'btn_inspect_test',
        progressText: '검사 테스트 진행 중... (0° → 90° → 180° → 270°)',
        completeText: '검사 테스트 완료 ✓',
        totalMs,
        errorLogPrefix: '[QUVI] 검사 단독 테스트 시작 실패:',
    });
}

// ─── ACT 모델 선택 ───
let _actModelUserTouched = false;

async function refreshActModels() {
    try {
        const res = await fetch('/api/act/models');
        const data = await res.json();
        const sel = document.getElementById('actModelSelect');
        const models = data.models || [];
        const current = data.current || {};

        if (sel) {
            // 사용자가 드롭다운을 건드리는 중엔 옵션 재구성 생략 (선택 유지)
            if (!_actModelUserTouched || sel.options.length <= 1) {
                const prev = sel.value;
                sel.innerHTML = '';
                if (models.length === 0) {
                    sel.innerHTML = '<option value="">사용 가능한 모델 없음</option>';
                } else {
                    for (const m of models) {
                        const opt = document.createElement('option');
                        opt.value = m.path;
                        opt.textContent = `${m.name} (step ${m.step})`;
                        if (m.path === (prev || current.path)) opt.selected = true;
                        sel.appendChild(opt);
                    }
                }
            }
        }

        const curEl = document.getElementById('actCurrentModel');
        if (curEl) curEl.textContent = current.name || (current.path ? '(경로 지정됨)' : '없음');
        const useEl = document.getElementById('actUseState');
        if (useEl) {
            useEl.textContent = current.use_act ? 'ON' : 'OFF';
            useEl.style.color = current.use_act ? 'var(--accent-green)' : 'var(--text-muted)';
        }
        const stEl = document.getElementById('actModelState');
        const loadBtn = document.getElementById('actModelLoadBtn');
        if (current.loading) {
            if (stEl) { stEl.textContent = '로딩 중...'; stEl.style.color = 'var(--accent-orange, #e9a)'; }
            if (loadBtn) loadBtn.disabled = true;
        } else {
            if (stEl) {
                stEl.textContent = current.ready ? '준비됨' : '미로드';
                stEl.style.color = current.ready ? 'var(--accent-green)' : 'var(--text-muted)';
            }
            if (loadBtn) loadBtn.disabled = false;
        }
    } catch (e) {
        // 서버 대기 중일 수 있음 — 조용히 무시
    }
}

async function loadActModel() {
    const sel = document.getElementById('actModelSelect');
    const msgEl = document.getElementById('actModelMsg');
    const path = sel ? sel.value : '';
    if (!path) {
        if (msgEl) { msgEl.textContent = '모델을 선택하세요.'; msgEl.style.color = 'var(--accent-red)'; }
        return;
    }
    _actModelUserTouched = false;
    if (msgEl) { msgEl.textContent = '로드 요청 전송...'; msgEl.style.color = 'var(--text-muted)'; }
    try {
        const res = await fetch('/api/act/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });
        const data = await res.json();
        if (data.ok) {
            if (msgEl) { msgEl.textContent = '로드 요청됨 — 로봇이 IDLE일 때만 적용됩니다.'; msgEl.style.color = 'var(--accent-green)'; }
            // 로딩 상태 반영 위해 잠시 자주 폴링
            let n = 0;
            const t = setInterval(() => { refreshActModels(); if (++n > 15) clearInterval(t); }, 1000);
        } else {
            if (msgEl) { msgEl.textContent = '오류: ' + (data.error || '알 수 없음'); msgEl.style.color = 'var(--accent-red)'; }
        }
    } catch (e) {
        if (msgEl) { msgEl.textContent = '네트워크 오류'; msgEl.style.color = 'var(--accent-red)'; }
    }
}

// ─── 장치 설정 (카메라/로봇/ESP USB) ───
function _shortDev(path) {
    // by-id 경로는 길어서 마지막 요소만 짧게 표시
    if (!path) return '';
    const parts = path.split('/');
    return parts[parts.length - 1];
}

async function refreshDevices() {
    const grid = document.getElementById('deviceRolesGrid');
    if (!grid) return;
    try {
        const res = await fetch('/api/devices');
        const data = await res.json();
        const roles = data.roles || [];
        const cands = data.candidates || {};
        const current = data.current || {};

        grid.innerHTML = '';
        for (const role of roles) {
            const list = (cands[role.type] || []).slice();
            const cur = current[role.key] || '';
            // 현재 값이 후보에 없어도 선택지에 포함(연결 안 됐거나 심링크)
            if (cur && !list.includes(cur)) list.unshift(cur);

            const cell = document.createElement('div');
            const lbl = document.createElement('div');
            lbl.style.cssText = 'font-size:11px;color:var(--text-muted);margin-bottom:4px;';
            lbl.textContent = role.label;
            const sel = document.createElement('select');
            sel.id = 'dev_' + role.key;
            sel.style.cssText = 'width:100%;padding:6px;box-sizing:border-box;';
            if (list.length === 0) {
                const o = document.createElement('option');
                o.value = ''; o.textContent = '(후보 없음)';
                sel.appendChild(o);
            }
            for (const d of list) {
                const o = document.createElement('option');
                o.value = d;
                o.textContent = _shortDev(d);
                o.title = d;
                if (d === cur) o.selected = true;
                sel.appendChild(o);
            }
            cell.appendChild(lbl);
            cell.appendChild(sel);
            grid.appendChild(cell);
        }
    } catch (e) {
        grid.innerHTML = '<div style="color:var(--accent-red);font-size:12px;">장치 목록 로드 실패</div>';
    }
}

async function applyDeviceConfig() {
    const msgEl = document.getElementById('deviceMsg');
    const btn = document.getElementById('deviceApplyBtn');
    const roleKeys = ['sidecam_device', 'fixed_cam_device', 'dxl_port', 'leader_dxl_port', 'micro_ros_port'];
    const config = {};
    for (const k of roleKeys) {
        const sel = document.getElementById('dev_' + k);
        if (sel && sel.value) config[k] = sel.value;
    }
    if (Object.keys(config).length === 0) {
        if (msgEl) { msgEl.textContent = '선택된 장치가 없습니다.'; msgEl.style.color = 'var(--accent-red)'; }
        return;
    }
    if (!confirm('장치 설정을 저장하고 5초 뒤 시스템을 재시작합니다.\n계속하시겠습니까?')) return;

    if (btn) btn.disabled = true;
    try {
        const res = await fetch('/api/devices/apply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config, delay_sec: 5 }),
        });
        const data = await res.json();
        if (data.ok) {
            let sec = data.restart_in || 5;
            if (msgEl) msgEl.style.color = 'var(--accent-green)';
            const tick = setInterval(() => {
                if (msgEl) msgEl.textContent = `저장됨 — ${sec}초 뒤 재시작...`;
                if (sec-- <= 0) {
                    clearInterval(tick);
                    if (msgEl) msgEl.textContent = '재시작 중... 잠시 후 페이지를 새로고침하세요.';
                }
            }, 1000);
        } else {
            if (msgEl) { msgEl.textContent = '오류: ' + (data.error || '알 수 없음'); msgEl.style.color = 'var(--accent-red)'; }
            if (btn) btn.disabled = false;
        }
    } catch (e) {
        if (msgEl) { msgEl.textContent = '네트워크 오류'; msgEl.style.color = 'var(--accent-red)'; }
        if (btn) btn.disabled = false;
    }
}

/* ═══════════════════════════════════════════
   [C] 고정 셸 — WebSocket 연결·시계·탭 전환·부트스트랩
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

// ─── 시계 ───
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('ko-KR', { hour12: false });
    const dateStr = now.toLocaleDateString('ko-KR', {
        year: 'numeric', month: '2-digit', day: '2-digit'
    });
    document.getElementById('clock').textContent = `${dateStr} ${timeStr}`;
}

// ─── 초기 데이터 로드 ───
async function loadInitialData() {
    try {
        const [statusRes, historyRes, statsRes] = await Promise.all([
            fetch('/api/status'),
            fetch('/api/inspection/history'),
            fetch('/api/inspection/stats'),
        ]);
        const status = await statusRes.json();
        const history = await historyRes.json();
        const stats = await statsRes.json();

        updateStatus(status);
        // pass/fail 통계는 오케스트레이터 카운트 단일 출처
        updateStats(stats);

        if (history.length > 0) {
            historyData = history.reverse().slice(0, 50);
            renderHistoryTable();
            updateLatestInspection(historyData[0]);
        }
    } catch (e) {
        console.log('[QUVI] 초기 데이터 로드 실패 (서버 대기 중)');
    }
}

// ─── 탭 전환 (?tab= 파라미터로 초기 탭 딥링크 가능) ───
function switchTab(tabName) {
    // 탭 판넬 전환
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    const targetTab = document.getElementById(`tab-${tabName}`);
    if (targetTab) targetTab.classList.add('active');

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const targetMenu = document.getElementById(`menu-${tabName}`);
    if (targetMenu) targetMenu.classList.add('active');
}

// ─── 부트스트랩 ───
setInterval(updateClock, 1000);
updateClock();

loadInitialData();
drawDonut(0, 0);

const _initialTab = new URLSearchParams(location.search).get('tab');
// 존재하는 탭일 때만 전환 — 잘못된 값이면 기본 탭 유지 (빈 화면 방지)
if (_initialTab && document.getElementById(`tab-${_initialTab}`)) switchTab(_initialTab);

document.addEventListener('change', (e) => {
    if (e.target && e.target.id === 'actModelSelect') _actModelUserTouched = true;
});

refreshActModels();
setInterval(refreshActModels, 5000);

refreshDevices();

// ─── ACT 추론 모니터링(rerun 웹 뷰어) ───
// 호스트명은 하드코딩하지 않고 현재 접속 호스트 기준으로 조립 (network_mode: host, 포트 9090 고정)
(function initRerunViewer() {
    const frame = document.getElementById('rerunViewerFrame');
    const fallbackMsg = document.getElementById('rerunFallbackMsg');
    const toggleBtn = document.getElementById('rerunToggleBtn');
    const panelBody = document.getElementById('rerunPanelBody');
    if (!frame || !toggleBtn || !panelBody) return;

    // ?url= 없이 열면 뷰어가 데이터 소스(ws 9877)에 자동 연결하지 않고 웰컴 화면만 표시
    const wsUrl = encodeURIComponent(`ws://${window.location.hostname}:9877`);
    frame.src = `http://${window.location.hostname}:9090/?url=${wsUrl}`;
    frame.addEventListener('error', () => { fallbackMsg.style.display = 'block'; });

    // 버튼이 패널 내부(camera-label)에 있으므로 iframe만 접는다 — 패널을 접으면 버튼도 사라짐
    toggleBtn.addEventListener('click', () => {
        const collapsed = frame.style.display === 'none';
        frame.style.display = collapsed ? '' : 'none';
        toggleBtn.textContent = collapsed ? '접기' : '펼치기';
    });
})();

// ─── 로봇 전체 뷰 (데모 영상) ───
// static/demo/robot_overview.mp4 가 존재할 때만 패널을 표시하고 rerun 패널을 오른쪽 2칸으로 줄인다
(function initOverviewPanel() {
    const panel = document.getElementById('overviewPanel');
    const video = document.getElementById('overviewVideo');
    const rerunPanel = document.getElementById('rerunPanelBody');
    if (!panel || !video || !rerunPanel) return;

    video.addEventListener('loadeddata', () => {
        panel.style.display = '';
        rerunPanel.style.gridColumn = '2 / -1';
        video.play().catch(() => {});

        // 데모 영상이 있으면 데모 모드로 간주 — 접속(새로고침 포함)마다 안내 모달 표시
        const overlay = document.getElementById('demoNoticeOverlay');
        const closeBtn = document.getElementById('demoNoticeCloseBtn');
        if (overlay && closeBtn) {
            overlay.style.display = '';
            closeBtn.addEventListener('click', () => {
                overlay.style.display = 'none';
            });
        }
    });
})();

// ─── 사이드바 접기/펼치기 ───
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const collapsed = sidebar.classList.toggle('collapsed');
    localStorage.setItem('sidebarCollapsed', collapsed ? '1' : '0');
}

if (localStorage.getItem('sidebarCollapsed') === '1') {
    document.getElementById('sidebar').classList.add('collapsed');
}
