const canvas = document.getElementById('ascii-canvas');
const ctx = canvas.getContext('2d');
const handsMeter = document.getElementById('hands-meter');
const wsStatus = document.getElementById('ws-status');

let width, height, dpr;
function resize() {
    dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
}
window.addEventListener('resize', resize);
resize();

// ═══════════════════════════════════════════════════════════════════════════
// 1. PRISTINE SPHERE GEOMETRY (Golden Angle Fibonacci Lattice)
// ═══════════════════════════════════════════════════════════════════════════
const POINT_COUNT = 1400;
const R0 = 230; // Constant base radius (px)
const nodes = [];
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

for (let i = 0; i < POINT_COUNT; i++) {
    const y = 1 - (i / (POINT_COUNT - 1)) * 2;
    const phi = Math.acos(Math.max(-1, Math.min(1, y)));
    const theta = GOLDEN_ANGLE * i;

    nodes.push({
        theta: theta,
        phi: phi,
        nx: Math.sin(phi) * Math.cos(theta),
        ny: Math.cos(phi),
        nz: Math.sin(phi) * Math.sin(theta),
        baseR: R0
    });
}

// Ordered glyph density: deep background (dim dust) → high-contrast foreground
const ASCII_CHARS = ['·', '.', ':', '+', 'x', '*', '%', '0', '#', '@'];

// ═══════════════════════════════════════════════════════════════════════════
// 2. ORB STATE & KINEMATICS
// ═══════════════════════════════════════════════════════════════════════════
let orbCenter = { x: width / 2, y: height / 2 };
let targetOrbCenter = { x: width / 2, y: height / 2 };
let rotX = 0, rotY = 0;
let targetRotX = 0, targetRotY = 0;
let angularVelX = 0, angularVelY = 0;
let currentScale = 1.0;
let targetScale = 1.0;

// Gesture interaction state anchors
let gestureState = 'IDLE'; // IDLE | HOVER | GRAB | DUAL_GRAB | SLAP
let anchorHand = { x: 0, y: 0, depth: 1.0 };
let anchorOrbPos = { x: width / 2, y: height / 2 };
let anchorScale = 1.0;
let anchorDualDist = 0.0;

let currentTheme = 'dark';
let sensitivity = 1.5;
let speedMult = 1.0;
let audioMode = 'both';
let hasHands = false;

// ═══════════════════════════════════════════════════════════════════════════
// 3. AMPLIFIED ACOUSTIC NEAR-FIELD FILTERING & GATING
// ═══════════════════════════════════════════════════════════════════════════
let audioCtx, analyser, biquadFilter, micSource;
let audioDataArray;
let smoothAudio = 0.0;
let ambientNoiseFloor = 0.012;

function initAudio() {
    if (audioCtx) return;
    try {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AC();

        // Extended vocal bandpass (180Hz – 4200Hz, Q = 0.65)
        biquadFilter = audioCtx.createBiquadFilter();
        biquadFilter.type = 'bandpass';
        biquadFilter.frequency.value = 2190;  // geometric mean of 180 & 4200
        biquadFilter.Q.value = 0.65;

        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.25;
        audioDataArray = new Uint8Array(analyser.frequencyBinCount);

        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            micSource = audioCtx.createMediaStreamSource(stream);
            micSource.connect(biquadFilter);
            biquadFilter.connect(analyser);
        }).catch(err => {
            console.warn("Microphone access restricted:", err);
        });
    } catch (e) {
        console.warn("Audio initialization deferred");
    }
}
window.addEventListener('click', initAudio, { once: true });
window.addEventListener('touchstart', initAudio, { once: true });

function updateAudioEnergy() {
    if (audioMode === 'off' || !analyser || !audioDataArray) {
        smoothAudio += (0.0 - smoothAudio) * 0.10;
        return;
    }

    analyser.getByteFrequencyData(audioDataArray);
    let sum = 0;
    for (let i = 0; i < audioDataArray.length; i++) {
        sum += audioDataArray[i];
    }
    const rawEnergy = (sum / audioDataArray.length) / 255.0;

    // Adaptive ambient noise floor tracking
    ambientNoiseFloor = ambientNoiseFloor * 0.993 + rawEnergy * 0.007;

    // Lowered SNR gate: 1.4x ambient threshold for responsive vocal reactivity
    let gatedEnergy = 0.0;
    const gateThreshold = ambientNoiseFloor * 1.4;
    if (rawEnergy > gateThreshold) {
        // Amplified baseline gain 2.4x
        gatedEnergy = Math.min((rawEnergy - gateThreshold) * sensitivity * 2.4, 1.0);
    }

    // Low-pass audio smoothing
    smoothAudio += (gatedEnergy - smoothAudio) * 0.10;
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. WEBSOCKET TELEMETRY CLIENT
// ═══════════════════════════════════════════════════════════════════════════
function connectWS() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        wsStatus.innerText = 'ONLINE';
    };

    ws.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            const handCount = data.hands ? data.hands.length : 0;

            // Strict footer format: `${count} HAND${count === 1 ? '' : 'S'}`
            handsMeter.innerText = `${handCount} HAND${handCount === 1 ? '' : 'S'}`;

            if (handCount > 0) {
                hasHands = true;
                const primary = data.hands[0];

                // ── STATE: DUAL GRAB ────────────────────────────────────
                if (data.dual_pinch && data.hands.length >= 2) {
                    if (gestureState !== 'DUAL_GRAB') {
                        gestureState = 'DUAL_GRAB';
                        anchorDualDist = Math.max(data.two_hand_dist, 0.05);
                        anchorOrbPos = { ...orbCenter };
                        anchorScale = currentScale;
                    }
                    targetOrbCenter.x = data.dual_pinch_center.x * width;
                    targetOrbCenter.y = data.dual_pinch_center.y * height;

                    const distRatio = data.two_hand_dist / anchorDualDist;
                    targetScale = Math.min(Math.max(anchorScale * distRatio, 0.35), 2.5);
                }
                // ── STATE: PHYSICAL GRAB (Single Pinch) ─────────────────
                else if (primary.is_pinching) {
                    if (gestureState !== 'GRAB') {
                        gestureState = 'GRAB';
                        anchorHand = {
                            x: primary.palm_x,
                            y: primary.palm_y,
                            depth: primary.depth_scale || 1.0
                        };
                        anchorOrbPos = { ...orbCenter };
                        anchorScale = currentScale;
                    }
                    // Translate orb X/Y with hand delta
                    const dx = (primary.palm_x - anchorHand.x) * width * 1.3;
                    const dy = (primary.palm_y - anchorHand.y) * height * 1.3;
                    targetOrbCenter.x = anchorOrbPos.x + dx;
                    targetOrbCenter.y = anchorOrbPos.y + dy;

                    // Scale Z proportionally to depth with spring damping
                    const depthRatio = (primary.depth_scale || 1.0) / anchorHand.depth;
                    targetScale = Math.min(Math.max(anchorScale * depthRatio * 0.85, 0.3), 2.2);
                }
                // ── STATE: HOVER (Open Hand Parallax) ───────────────────
                else {
                    gestureState = 'HOVER';

                    // 1:1 orientation mapping from palm Euler angles if available
                    if (primary.pitch !== undefined) {
                        targetRotX = primary.pitch * 1.8;
                        targetRotY = primary.yaw * 1.8;
                    } else {
                        targetRotY = (primary.palm_x - 0.5) * 3.2;
                        targetRotX = (primary.palm_y - 0.5) * 3.2;
                    }

                    if (data.two_hand_dist > 0) {
                        targetScale = THREE_MAP(data.two_hand_dist, 0.15, 0.7, 0.5, 2.0);
                    } else {
                        targetScale = primary.depth_scale || 1.0;
                    }

                    // Slap / Flick momentum injection
                    if (data.slap_impulse && data.slap_impulse.active) {
                        angularVelY += data.slap_impulse.vx * 0.14;
                        angularVelX += data.slap_impulse.vy * 0.14;
                    }
                }
            } else {
                // ── STATE: IDLE (No Hands) ──────────────────────────────
                hasHands = false;
                gestureState = 'IDLE';
                targetScale = 1.0;
                targetOrbCenter.x = width / 2;
                targetOrbCenter.y = height / 2;
            }
        } catch (err) {
            console.error("Telemetry parse error:", err);
        }
    };

    ws.onclose = () => {
        wsStatus.innerText = 'OFFLINE';
        setTimeout(connectWS, 1500);
    };
}
connectWS();

function THREE_MAP(val, inMin, inMax, outMin, outMax) {
    return outMin + (outMax - outMin) * Math.max(0, Math.min(1, (val - inMin) / (inMax - inMin)));
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. UI CONTROLS & SEGMENTED BUTTON SYNCHRONIZATION
// ═══════════════════════════════════════════════════════════════════════════
const settingsDrawer = document.getElementById('settings-drawer');
document.getElementById('settings-btn').onclick = () => settingsDrawer.classList.add('open');
document.getElementById('close-settings').onclick = () => settingsDrawer.classList.remove('open');

// Synchronize all segmented groups (.segmented-group > .segmented-btn)
document.querySelectorAll('.segmented-group').forEach(group => {
    const buttons = group.querySelectorAll('.segmented-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            if (btn.dataset.theme) {
                currentTheme = btn.dataset.theme;
                document.documentElement.setAttribute('data-theme', currentTheme);
            }
            if (btn.dataset.audioMode) {
                audioMode = btn.dataset.audioMode;
            }
        });
    });
});

document.getElementById('sens-slider').oninput = (e) => {
    sensitivity = parseFloat(e.target.value);
    document.getElementById('sens-val').innerText = `${sensitivity.toFixed(1)}x`;
};
document.getElementById('speed-slider').oninput = (e) => {
    speedMult = parseFloat(e.target.value);
    document.getElementById('speed-val').innerText = `${speedMult.toFixed(1)}x`;
};

// ═══════════════════════════════════════════════════════════════════════════
// 6. HIGH-PERFORMANCE RENDER LOOP — Amplified Fluid Physics
// ═══════════════════════════════════════════════════════════════════════════
let time = 0;
const projectedNodes = new Array(POINT_COUNT);
for (let i = 0; i < POINT_COUNT; i++) {
    projectedNodes[i] = { x: 0, y: 0, z: 0, depth: 0, glyphIndex: 0 };
}

function render() {
    requestAnimationFrame(render);
    time += 0.015 * speedMult;

    updateAudioEnergy();

    // Spring-damper kinematics with 0.12 spring factor
    const springK = 0.12;
    orbCenter.x += (targetOrbCenter.x - orbCenter.x) * springK;
    orbCenter.y += (targetOrbCenter.y - orbCenter.y) * springK;
    currentScale += (targetScale - currentScale) * springK;

    // IDLE: Slow continuous idle drift (0.002)
    if (gestureState === 'IDLE') {
        targetRotY += 0.002 * speedMult;
    }

    rotX += (targetRotX - rotX) * 0.08 + angularVelX;
    rotY += (targetRotY - rotY) * 0.08 + angularVelY;
    // Viscous friction decay (0.94)
    angularVelX *= 0.94;
    angularVelY *= 0.94;

    ctx.clearRect(0, 0, width, height);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);

    // Expanded deformation bounds [0.85 * R0, 1.22 * R0]
    const minBound = 0.85 * R0;
    const maxBound = 1.22 * R0;

    for (let i = 0; i < POINT_COUNT; i++) {
        const node = nodes[i];

        // Amplified fluid surface-tension harmonics:
        // delta_r = R0 * [0.012 * sin(2θ + 3φ + t) + smoothAudio * 0.16 * sin(4θ + 3φ + 2.5t)]
        const idleWave = 0.012 * Math.sin(2 * node.theta + 3 * node.phi + time);
        const voiceWave = smoothAudio * 0.16 * Math.sin(4 * node.theta + 3 * node.phi + 2.5 * time);
        const deltaR = R0 * (idleWave + voiceWave);

        // Strict radial boundary clamping
        const clampedR = Math.max(minBound, Math.min(maxBound, R0 + deltaR));
        const effectiveR = clampedR * currentScale;

        const nx = node.nx * effectiveR;
        const ny = node.ny * effectiveR;
        const nz = node.nz * effectiveR;

        // 3D Rotation Transformations
        const x1 = nx * cosY + nz * sinY;
        const z1 = -nx * sinY + nz * cosY;
        const y2 = ny * cosX - z1 * sinX;
        const z2 = ny * sinX + z1 * cosX;

        // Perspective Projection
        const fov = 580;
        const cameraDist = 420;
        const depth = fov / (fov + z2 + cameraDist);
        const screenX = orbCenter.x + x1 * depth;
        const screenY = orbCenter.y + y2 * depth;

        // Smooth Glyph Density based strictly on projected Z-depth
        const normalizedZ = (z2 + (R0 * currentScale)) / (2 * R0 * currentScale + 0.001);
        const clampedZ = Math.max(0, Math.min(1, normalizedZ));
        const rawGlyphIdx = Math.floor(clampedZ * (ASCII_CHARS.length - 1));

        const p = projectedNodes[i];
        p.x = screenX;
        p.y = screenY;
        p.z = z2;
        p.depth = depth;
        p.glyphIndex = rawGlyphIdx;
    }

    // Depth Sort (back-to-front)
    projectedNodes.sort((a, b) => a.z - b.z);

    const isDark = currentTheme === 'dark';

    for (let i = 0; i < POINT_COUNT; i++) {
        const p = projectedNodes[i];
        const char = ASCII_CHARS[p.glyphIndex];

        const fontSize = Math.max(7, Math.floor(16 * p.depth));
        const normalizedDepth = (p.z + (R0 * currentScale)) / (2 * R0 * currentScale + 0.001);
        const alpha = Math.min(Math.max(0.14 + normalizedDepth * 0.86, 0.14), 1.0);

        ctx.font = `${fontSize}px 'Plus Jakarta Sans', -apple-system, sans-serif`;
        ctx.fillStyle = isDark
            ? `rgba(255, 255, 255, ${alpha.toFixed(2)})`
            : `rgba(9, 9, 11, ${alpha.toFixed(2)})`;

        ctx.fillText(char, p.x, p.y);
    }
}
render();
