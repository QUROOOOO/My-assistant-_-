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
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5)); // ~2.399963 rad

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

// Ordered glyph density from background to high-contrast foreground
const ASCII_CHARS = ['·', '.', ':', '+', 'x', '*', '%', '0', '#', '@'];

// ═══════════════════════════════════════════════════════════════════════════
// 2. ORB STATE & KINEMATICS
// ═══════════════════════════════════════════════════════════════════════════
let orbCenter = { x: width / 2, y: height / 2 };
let targetOrbCenter = { x: width / 2, y: height / 2 };
let rotX = 0, rotY = 0, rotZ = 0;
let targetRotX = 0, targetRotY = 0, targetRotZ = 0;
let angularVelX = 0, angularVelY = 0, angularVelZ = 0;
let currentScale = 1.0;
let targetScale = 1.0;

// Gesture state machine
let gestureState = 'IDLE'; // IDLE | HOVER | GRAB | DUAL_GRAB | COMPRESS | DUAL_MOLD | SLAP
let lockedPinchHandId = null; // Handedness label of actively locked pinching hand
let anchorHand = { x: 0, y: 0, depth: 1.0 };
let anchorOrbPos = { x: width / 2, y: height / 2 };
let anchorScale = 1.0;
let anchorDualDist = 0.0;

// Bloom Shockwave Dynamics
let bloomActive = false;
let bloomStartTime = 0;
const BLOOM_DURATION = 0.6; // seconds

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
        biquadFilter.frequency.value = 2190;
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

    // Responsive 1.4x SNR baseline gate
    let gatedEnergy = 0.0;
    const gateThreshold = ambientNoiseFloor * 1.4;
    if (rawEnergy > gateThreshold) {
        // Boosted baseline sensitivity gain (4.32x)
        gatedEnergy = Math.min((rawEnergy - gateThreshold) * sensitivity * 4.32, 1.0);
    }

    // Low-pass audio smoothing
    smoothAudio += (gatedEnergy - smoothAudio) * 0.10;
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. MULTI-HAND CONFLICT ARBITRATION & 5-STATE GESTURE CLIENT
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

            // Streamlined footer
            handsMeter.innerText = `${handCount} HAND${handCount === 1 ? '' : 'S'}`;

            // Check Shockwave Bloom trigger from telemetry
            if (data.bloom) {
                bloomActive = true;
                bloomStartTime = performance.now() / 1000;
            }

            if (handCount > 0) {
                hasHands = true;

                // Find pinching hands
                const pinchingHands = data.hands.filter(h => h.is_pinching);

                // ── CASE A: DUAL PINCH (Both hands pinching simultaneously) ──
                if (data.dual_pinch || pinchingHands.length >= 2) {
                    lockedPinchHandId = null;
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
                // ── CASE B: PINCH PRIORITY LOCK (Single hand pinching) ──────
                else if (pinchingHands.length === 1 || lockedPinchHandId !== null) {
                    let activeGrabHand = null;

                    // If a hand was already locked, check if it's still pinching
                    if (lockedPinchHandId) {
                        activeGrabHand = data.hands.find(h => h.id === lockedPinchHandId && h.is_pinching);
                        if (!activeGrabHand) {
                            lockedPinchHandId = null; // Hand released pinch
                        }
                    }

                    // Otherwise lock onto the newly pinching hand
                    if (!activeGrabHand && pinchingHands.length > 0) {
                        activeGrabHand = pinchingHands[0];
                        lockedPinchHandId = activeGrabHand.id;
                    }

                    if (activeGrabHand) {
                        // Exclusive grab control: Suppress all other hands!
                        if (gestureState !== 'GRAB') {
                            gestureState = 'GRAB';
                            anchorHand = {
                                x: activeGrabHand.palm_x,
                                y: activeGrabHand.palm_y,
                                depth: activeGrabHand.depth_scale || 1.0
                            };
                            anchorOrbPos = { ...orbCenter };
                            anchorScale = currentScale;
                        }

                        const dx = (activeGrabHand.palm_x - anchorHand.x) * width * 1.3;
                        const dy = (activeGrabHand.palm_y - anchorHand.y) * height * 1.3;
                        targetOrbCenter.x = anchorOrbPos.x + dx;
                        targetOrbCenter.y = anchorOrbPos.y + dy;

                        const depthRatio = (activeGrabHand.depth_scale || 1.0) / anchorHand.depth;
                        targetScale = Math.min(Math.max(anchorScale * depthRatio * 0.85, 0.35), 2.2);
                    }
                }
                // ── CASE C: DUAL-HAND VOLUMETRIC MOLDING (2 open hands) ──────
                else if (handCount >= 2) {
                    lockedPinchHandId = null;
                    gestureState = 'DUAL_MOLD';

                    // Smooth volumetric scale from 1€ filtered distance
                    if (data.two_hand_dist > 0) {
                        targetScale = THREE_MAP(data.two_hand_dist, 0.12, 0.65, 0.5, 2.5);
                    }
                    if (data.dual_angle !== undefined) {
                        targetRotZ = data.dual_angle * 0.8;
                    }
                }
                // ── CASE D: SINGLE HAND INTERACTIONS ────────────────────────
                else {
                    lockedPinchHandId = null;
                    const primary = data.hands[0];

                    // State 3: Fist Compress
                    if (primary.is_fist || data.compress) {
                        gestureState = 'COMPRESS';
                        targetScale = 0.70;
                        targetRotY = (primary.palm_x - 0.5) * 1.5;
                        targetRotX = (primary.palm_y - 0.5) * 1.5;
                    }
                    // State 1: Magnetic Hover
                    else if (primary.is_open) {
                        gestureState = 'HOVER';
                        targetScale = primary.depth_scale || 1.0;

                        if (primary.pitch !== undefined) {
                            targetRotX = primary.pitch * 1.8;
                            targetRotY = primary.yaw * 1.8;
                            targetRotZ = primary.roll * 0.6;
                        } else {
                            targetRotY = (primary.palm_x - 0.5) * 3.0;
                            targetRotX = (primary.palm_y - 0.5) * 3.0;
                        }

                        // State 4: Open Palm Slap / Flick
                        if (data.slap_impulse && data.slap_impulse.active) {
                            angularVelY += data.slap_impulse.vx * 0.16;
                            angularVelX += data.slap_impulse.vy * 0.16;
                        }
                    }
                    else {
                        gestureState = 'HOVER';
                        targetRotY = (primary.palm_x - 0.5) * 2.5;
                        targetRotX = (primary.palm_y - 0.5) * 2.5;
                    }
                }
            } else {
                // State: IDLE (No Hands)
                hasHands = false;
                lockedPinchHandId = null;
                gestureState = 'IDLE';
                targetScale = 1.0;
                targetRotZ = 0.0;
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
// 5. UI CONTROLS, SENSOR STREAM TOGGLE & GESTURE ACCORDION
// ═══════════════════════════════════════════════════════════════════════════
const settingsDrawer = document.getElementById('settings-drawer');
document.getElementById('settings-btn').onclick = () => settingsDrawer.classList.add('open');
document.getElementById('close-settings').onclick = () => settingsDrawer.classList.remove('open');

// In-Browser Sensor Stream Preview Toggle
const sensorToggle = document.getElementById('sensor-toggle');
const sensorPreviewBox = document.getElementById('sensor-preview-box');
const sensorFeedImg = document.getElementById('sensor-feed-img');

if (sensorToggle && sensorPreviewBox && sensorFeedImg) {
    sensorToggle.addEventListener('change', () => {
        if (sensorToggle.checked) {
            sensorPreviewBox.classList.add('active');
            sensorFeedImg.src = '/video_feed';
        } else {
            sensorPreviewBox.classList.remove('active');
            sensorFeedImg.src = '';
        }
    });
}

// Interactive Gesture Manual Accordion Toggle
const gestureAccordion = document.getElementById('gesture-accordion');
const gestureToggle = document.getElementById('gesture-toggle');
if (gestureToggle && gestureAccordion) {
    gestureToggle.addEventListener('click', () => {
        gestureAccordion.classList.toggle('expanded');
    });
}

// Segmented group synchronizer (.segmented-group > .segmented-btn)
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
// 6. HIGH-PERFORMANCE RENDER LOOP — Shockwaves & Damped Kinematics
// ═══════════════════════════════════════════════════════════════════════════
let time = 0;
const projectedNodes = new Array(POINT_COUNT);
for (let i = 0; i < POINT_COUNT; i++) {
    projectedNodes[i] = { x: 0, y: 0, z: 0, depth: 0, glyphIndex: 0 };
}

function render() {
    requestAnimationFrame(render);
    const nowSec = performance.now() / 1000;
    time += 0.015 * speedMult;

    updateAudioEnergy();

    // Damped position & scale interpolation (heavy damping for dual-mold)
    const isDualMold = gestureState === 'DUAL_MOLD';
    const springK = isDualMold ? 0.05 : 0.12;

    orbCenter.x += (targetOrbCenter.x - orbCenter.x) * springK;
    orbCenter.y += (targetOrbCenter.y - orbCenter.y) * springK;
    currentScale += (targetScale - currentScale) * springK;

    // IDLE: Slow continuous idle drift (0.002)
    if (gestureState === 'IDLE') {
        targetRotY += 0.002 * speedMult;
    }

    // Smooth rotational kinematics
    const rotLerp = isDualMold ? 0.025 : 0.038;
    rotX += (targetRotX - rotX) * rotLerp + angularVelX;
    rotY += (targetRotY - rotY) * rotLerp + angularVelY;
    rotZ += (targetRotZ - rotZ) * rotLerp + angularVelZ;

    // Viscous friction decay (0.95)
    angularVelX *= 0.95;
    angularVelY *= 0.95;
    angularVelZ *= 0.95;

    // Shockwave Bloom elapsed time
    let bloomProgress = 0.0;
    if (bloomActive) {
        bloomProgress = nowSec - bloomStartTime;
        if (bloomProgress >= BLOOM_DURATION) {
            bloomActive = false;
        }
    }

    ctx.clearRect(0, 0, width, height);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const cosZ = Math.cos(rotZ), sinZ = Math.sin(rotZ);

    // Expanded dynamic bounds: [0.80 * R0, 1.30 * R0]
    const minBound = 0.80 * R0;
    const maxBound = 1.30 * R0;
    const isCompress = gestureState === 'COMPRESS';

    for (let i = 0; i < POINT_COUNT; i++) {
        const node = nodes[i];

        // Amplified fluid surface-tension harmonics
        const idleWave = 0.012 * Math.sin(2 * node.theta + 3 * node.phi + time);
        const voiceWave = smoothAudio * 0.18 * Math.sin(4 * node.theta + 3 * node.phi + 2.5 * time);
        let deltaR = R0 * (idleWave + voiceWave);

        // Fluid Shockwave Bloom: Delta_r = R0 * 0.35 * exp(-4t) * sin(8*PI*t - 4*phi)
        if (bloomActive && bloomProgress < BLOOM_DURATION) {
            const bloomDecay = Math.exp(-4.0 * bloomProgress);
            const bloomWave = Math.sin(8.0 * Math.PI * bloomProgress - 4.0 * node.phi);
            deltaR += R0 * 0.35 * bloomDecay * bloomWave;
        }

        // Strict radial boundary clamping
        const clampedR = Math.max(minBound, Math.min(maxBound, R0 + deltaR));
        const effectiveR = clampedR * currentScale;

        const nx = node.nx * effectiveR;
        const ny = node.ny * effectiveR;
        const nz = node.nz * effectiveR;

        // 3D Euler Rotations (Y -> X -> Z)
        const x1 = nx * cosY + nz * sinY;
        const z1 = -nx * sinY + nz * cosY;
        const y2 = ny * cosX - z1 * sinX;
        const z2 = ny * sinX + z1 * cosX;
        const x3 = x1 * cosZ - y2 * sinZ;
        const y3 = x1 * sinZ + y2 * cosZ;

        // Perspective Projection
        const fov = 580;
        const cameraDist = 420;
        const depth = fov / (fov + z2 + cameraDist);
        const screenX = orbCenter.x + x3 * depth;
        const screenY = orbCenter.y + y3 * depth;

        // Projected Z-depth character density (modulated if compressed)
        const normalizedZ = (z2 + (R0 * currentScale)) / (2 * R0 * currentScale + 0.001);
        const clampedZ = Math.max(0, Math.min(1, normalizedZ));
        const densityMultiplier = isCompress ? 1.4 : 1.0;
        const rawGlyphIdx = Math.min(
            ASCII_CHARS.length - 1,
            Math.floor(clampedZ * densityMultiplier * (ASCII_CHARS.length - 1))
        );

        const p = projectedNodes[i];
        p.x = screenX;
        p.y = screenY;
        p.z = z2;
        p.depth = depth;
        p.glyphIndex = rawGlyphIdx;
    }

    // Depth Sorting (Back-to-front)
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
