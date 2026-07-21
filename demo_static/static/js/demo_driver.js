/* ═══════════════════════════════════════════
   QUVI 정적 데모 드라이버 (GitHub Pages 전용)
   실제 백엔드 없이 dashboard.js를 무수정 재사용하기 위해
   window.io()와 window.fetch를 가짜로 대체하고, FSM 시나리오를
   status_update 이벤트로 재생한다. dashboard.js보다 먼저 로드되어야 한다.
   ═══════════════════════════════════════════ */
(function () {
    const _origFetch = window.fetch.bind(window);

    // ─── 가짜 socket.io — dashboard.js의 on(event,cb) 등록만 지원 ───
    const listeners = {};
    const fakeSocket = {
        on(event, cb) {
            (listeners[event] = listeners[event] || []).push(cb);
            // dashboard.js 로드가 느리면 아래 setTimeout 발화를 놓친다 —
            // connect는 등록 즉시 비동기로 한 번 더 전달해 레이스를 제거
            if (event === 'connect') setTimeout(() => cb(), 0);
        },
        emit() { /* 데모에서는 서버로 보낼 것이 없음 */ },
    };
    window.io = () => fakeSocket;
    function fire(event, data) {
        (listeners[event] || []).forEach((cb) => { try { cb(data); } catch (e) { console.error(e); } });
    }

    // ─── 우하단 토스트 ───
    let toastTimer = null;
    function toast(msg) {
        let el = document.getElementById('demoToast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'demoToast';
            el.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:2000;' +
                'background:rgba(20,26,38,0.95);color:#fff;padding:10px 16px;border-radius:8px;' +
                'font-size:12px;border:1px solid rgba(255,255,255,0.12);box-shadow:0 4px 16px rgba(0,0,0,0.4);' +
                'transition:opacity .2s;opacity:0;';
            document.body.appendChild(el);
        }
        el.textContent = msg;
        el.style.opacity = '1';
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => { el.style.opacity = '0'; }, 2500);
    }

    // ─── 레일/관절 등 표시용 상수 (hmi_node.py RAIL_STATION_MAP·실제 검사 로그 기반) ───
    const RAIL = { HOME: 0, INSPECT: 12.5, PASS: 25, FAIL: 125, BED: 381.25 };
    const RAIL_STATION_MAP = [
        { name: 'INSPECT (A)', mm: RAIL.INSPECT },
        { name: 'PASS (B)', mm: RAIL.PASS },
        { name: 'FAIL (C)', mm: RAIL.FAIL },
        { name: 'BED (D)', mm: RAIL.BED },
    ];
    // 판정값은 data/inspection_logs 실제 로그(PASS: 20260626_045336, FAIL: 20260626_045644)에서 가져옴
    const SCENARIOS = {
        PASS: {
            passed: true, fail_reason: '',
            solidity: 0.9919, area_ratio: 1.02, hole_count: 0, hole_area_ratio: 0.0,
            texture_variance: 4.01, inspection_time_sec: 1.82,
            image: './assets/captured_pass.png',
        },
        FAIL: {
            passed: false, fail_reason: 'Solidity 0.576 미달(워핑) + 구멍 3개 검출',
            solidity: 0.5755, area_ratio: 5.6148, hole_count: 3, hole_area_ratio: 0.1730,
            texture_variance: 9.28, inspection_time_sec: 2.14,
            image: './assets/captured_fail.png',
        },
    };

    // dashboard.js 관절 변환 역산: 일반 관절 deg=(rad*180/pi)-180, 그리퍼 ticks=rad*4096/(2*pi)
    function degToRad(deg) { return (deg + 180) * Math.PI / 180; }
    function gripperRad(ticks) { return ticks * 2 * Math.PI / 4096; }
    const J_HOME       = [degToRad(0),   degToRad(0),  degToRad(0),   degToRad(0), degToRad(0),  gripperRad(2300)];
    const J_GRASP      = [degToRad(-25), degToRad(40), degToRad(-60), degToRad(20), degToRad(10), gripperRad(1800)];
    const J_INSPECT    = [degToRad(15),  degToRad(20), degToRad(-30), degToRad(5), degToRad(-5), gripperRad(1800)];
    const J_SORT_PASS  = [degToRad(70),  degToRad(30), degToRad(-45), degToRad(10), degToRad(0), gripperRad(1800)];
    const J_SORT_FAIL  = [degToRad(-70), degToRad(30), degToRad(-45), degToRad(10), degToRad(0), gripperRad(1800)];

    // ─── 시나리오 상태 ───
    let objectIndex = -1;          // 0-based, GRASPING 시작 시 +1
    let nextResult = 'PASS';       // START 클릭마다 PASS/FAIL 교대
    let stats = { passed: 0, failed: 0 };
    let history = [];
    let latest = null;
    let running = false;
    let timers = [];
    let currentState = 'IDLE';
    let errorMessage = '';
    let joints = J_HOME, rail = RAIL.HOME, turn = 0, led = false, teleopActive = false;

    function statsPayload() {
        const total = stats.passed + stats.failed;
        return { passed: stats.passed, failed: stats.failed, pass_rate: total ? (stats.passed / total * 100) : 0 };
    }
    function status() {
        return {
            current_state: currentState,
            total_objects: objectIndex + 1,
            processed_count: stats.passed + stats.failed,
            pass_count: stats.passed,
            fail_count: stats.failed,
            grasp_ready: true, inspect_ready: true, motor_ready: true,
            teleop_active: teleopActive,
            error_message: errorMessage,
            joint_positions: joints,
            rail_position: rail,
            turntable_angle: turn,
            rail_station_map: RAIL_STATION_MAP,
            led_state: led,
        };
    }
    function emit(partial) {
        if (partial) {
            if (partial.state !== undefined) currentState = partial.state;
            if (partial.joints) joints = partial.joints;
            if (partial.rail !== undefined) rail = partial.rail;
            if (partial.turn !== undefined) turn = partial.turn;
            if (partial.led !== undefined) led = partial.led;
            if (partial.teleop !== undefined) teleopActive = partial.teleop;
            if (partial.error !== undefined) errorMessage = partial.error;
        }
        fire('status_update', { status: status(), stats: statsPayload(), latest_inspection: latest });
    }

    // ─── 카메라 표시 (sidecam/camera2 비디오, inspect_debug 실사 이미지) ───
    const elSide = document.getElementById('vidSidecam');
    const elCam2 = document.getElementById('vidCamera2');
    const elDebug = document.getElementById('imgInspectDebug');
    const elCam2Insp = document.getElementById('vidCamera2Insp');
    const elDebugInsp = document.getElementById('imgInspectDebugInsp');

    function setCams({ side, cam2, debugImg }) {
        [elSide].forEach((v) => {
            if (!v) return;
            v.style.display = side ? '' : 'none';
            if (side) v.play().catch(() => {}); else v.pause();
        });
        [elCam2, elCam2Insp].forEach((v) => {
            if (!v) return;
            v.style.display = cam2 ? '' : 'none';
            if (cam2) v.play().catch(() => {}); else v.pause();
        });
        [elDebug, elDebugInsp].forEach((img) => {
            if (!img) return;
            if (debugImg) { img.src = debugImg; img.style.display = ''; }
            else { img.style.display = 'none'; }
        });
    }

    // ─── FSM 시나리오 엔진 ───
    function clearTimers() { timers.forEach(clearTimeout); timers = []; }
    function after(ms, fn) { timers.push(setTimeout(fn, ms)); }

    function startCycle() {
        if (running) return; // 재생 중 재클릭 무시
        running = true;
        const label = nextResult;
        const scn = SCENARIOS[label];
        nextResult = (label === 'PASS') ? 'FAIL' : 'PASS';
        objectIndex += 1;

        emit({ state: 'GRASPING', joints: J_GRASP, rail: RAIL.BED, turn: 0 });
        setCams({ side: true, cam2: false, debugImg: null });

        after(3000, () => {
            emit({ state: 'INSPECTING', joints: J_INSPECT, rail: RAIL.INSPECT, turn: 0 });
            setCams({ side: false, cam2: true, debugImg: null });
        });
        after(3800, () => emit({ turn: 90 }));
        after(4600, () => emit({ turn: 180 }));
        after(5400, () => {
            emit({ turn: 270 });
            setCams({ side: false, cam2: true, debugImg: scn.image }); // 검사캠 + 판정 패널에 촬영 이미지
        });

        after(7000, () => {
            if (scn.passed) stats.passed += 1; else stats.failed += 1;
            latest = Object.assign({ timestamp: new Date().toISOString(), object_index: objectIndex }, scn);
            history.unshift(latest);
            if (history.length > 50) history.pop();
            const target = scn.passed ? RAIL.PASS : RAIL.FAIL;
            emit({ state: 'SORTING', joints: scn.passed ? J_SORT_PASS : J_SORT_FAIL, rail: target, turn: 0 });
            setCams({ side: false, cam2: false, debugImg: scn.image });
        });

        after(9500, () => emit({ state: 'RELEASING' }));

        after(11500, () => {
            emit({ state: 'HOMING', joints: J_HOME, rail: RAIL.HOME, turn: 0 });
            setCams({ side: false, cam2: false, debugImg: null });
        });

        after(14500, () => emit({ state: 'FINISHED' }));

        after(16500, () => {
            emit({ state: 'IDLE' });
            running = false;
        });
    }

    function stopCycle() {
        clearTimers();
        running = false;
        setCams({ side: false, cam2: false, debugImg: null });
        emit({ state: 'IDLE', joints: J_HOME, rail: RAIL.HOME, turn: 0 });
    }

    function estopCycle() {
        clearTimers();
        running = false;
        setCams({ side: false, cam2: false, debugImg: null });
        // "비상정지"·"ESTOP" 문자열은 dashboard.js _updateFsmHighlight 정규식(/ESTOP|비상정지/i)과 매칭되어야 함
        emit({ state: 'ERROR', error: 'ESTOP 비상정지 활성 — RESET으로 해제하세요.' });
    }

    function resetCycle() {
        clearTimers();
        running = false;
        setCams({ side: false, cam2: false, debugImg: null });
        emit({ state: 'IDLE', joints: J_HOME, rail: RAIL.HOME, turn: 0, error: '' });
    }

    // ─── 가짜 fetch ───
    function jsonResponse(obj) {
        return Promise.resolve(new Response(JSON.stringify(obj), {
            status: 200, headers: { 'Content-Type': 'application/json' },
        }));
    }

    window.fetch = function (url, opts) {
        const u = typeof url === 'string' ? url : (url && url.url) || '';
        if (!u.startsWith('/api/')) return _origFetch(url, opts);

        if (u.startsWith('/api/command/')) {
            const cmd = u.split('/').pop();
            if (cmd === 'start') startCycle();
            else if (cmd === 'stop') stopCycle();
            else if (cmd === 'estop') estopCycle();
            else if (cmd === 'reset') resetCycle();
            return jsonResponse({ ok: true, result: 'demo' });
        }
        if (u === '/api/status') return jsonResponse(status());
        if (u === '/api/inspection/history') return jsonResponse(history);
        if (u === '/api/inspection/stats') return jsonResponse(statsPayload());

        // ─── 수동 제어 (데모에서 실제 반영) ───
        if (u === '/api/rail/move' || u === '/api/turntable/move') {
            let body = {};
            try { body = JSON.parse((opts && opts.body) || '{}'); } catch (e) { /* 무시 */ }
            if (u === '/api/rail/move') emit({ rail: Number(body.mm) || 0 });
            else emit({ turn: Number(body.angle) || 0 });
            return jsonResponse({ ok: true });
        }
        if (u === '/api/led/on' || u === '/api/led/off') {
            const on = u.endsWith('/on');
            emit({ led: on });
            return jsonResponse({ ok: true, led: on });
        }

        // ─── 텔레옵: 활성화하면 녹화 영상 재생, 끝나면 자동 OFF ───
        if (u === '/api/teleop/on' || u === '/api/teleop/off') {
            const on = u.endsWith('/on');
            const tv = document.getElementById('teleopVideo');
            if (tv) {
                if (on) {
                    tv.src = './assets/teleop.mp4';
                    tv.loop = false;
                    tv.play().catch(() => {});
                    tv.onended = () => {
                        const tog = document.getElementById('teleopToggle');
                        if (tog) tog.checked = false;
                        window.fetch('/api/teleop/off', { method: 'POST' });
                        if (typeof setTeleopBadge === 'function') setTeleopBadge('off');
                    };
                } else {
                    tv.onended = null;
                    tv.src = './assets/sidecam.mp4';
                    tv.loop = true;
                    tv.play().catch(() => {});
                }
            }
            emit({ teleop: on });
            return jsonResponse({ ok: true, teleop: on });
        }

        toast('데모 모드 — 실기에서만 동작합니다');
        return jsonResponse({ ok: true, result: 'demo' });
    };

    // dashboard.js가 소켓 connect 이벤트로 연결 배지를 켜도록 즉시 발화
    setTimeout(() => fire('connect'), 0);
})();
