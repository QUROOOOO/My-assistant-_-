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
// 1. PRISTINE SPHERE GEOMETRY & 360° SUPERNOVA PARTICLES
// ═══════════════════════════════════════════════════════════════════════════
const POINT_COUNT = 1400;
const R0 = 230; // Constant base radius (px)
const nodes = [];
const burstParticles = [];
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5)); // ~2.399963 rad

for (let i = 0; i < POINT_COUNT; i++) {
    const y = 1 - (i / (POINT_COUNT - 1)) * 2;
    const phi = Math.acos(Math.max(-1, Math.min(1, y)));
    const theta = GOLDEN_ANGLE * i;

    const nx = Math.sin(phi) * Math.cos(theta);
    const ny = Math.cos(phi);
    const nz = Math.sin(phi) * Math.sin(theta);

    nodes.push({
        theta: theta,
        phi: phi,
        nx: nx,
        ny: ny,
        nz: nz,
        baseR: R0
    });

    // 360° Omnidirectional Cosmic Dispersion (Disperses 2.5x - 3.5x viewport scale)
    const speed = 900 + Math.sin(i * 13.7) * 420;
    const chaosX = (Math.cos(i * 7.9)) * 0.28;
    const chaosY = (Math.sin(i * 11.3)) * 0.28;
    const chaosZ = (Math.cos(i * 14.1)) * 0.28;
    const dir = { x: nx + chaosX, y: ny + chaosY, z: nz + chaosZ };
    const dirLen = Math.sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) || 1.0;

    burstParticles.push({
        vx: (dir.x / dirLen) * speed,
        vy: (dir.y / dirLen) * speed,
        vz: (dir.z / dirLen) * speed,
        lambda: 0.40, // Exp damping factor: exp(-0.4 * t)
        seed: i * 41
    });
}

// Standard ASCII density gradient
const ASCII_CHARS = ['·', '.', ':', '+', 'x', '*', '%', '0', '#', '@'];
// Dense high-energy plasma glyphs (Fist compression)
const PLASMA_CHARS = ['@', '#', '%', '&', '8'];
// Cosmic supernova burst randomized glyphs
const BURST_CHARS = ['*', '+', '·', '%', '0', '@', 'x', '¤'];

// ── Zero-Allocation String Caches for High-Performance 60/120 FPS ────────
const FONT_CACHE = {};
for (let sz = 1; sz <= 80; sz++) {
    FONT_CACHE[sz] = `${sz}px 'Plus Jakarta Sans', -apple-system, sans-serif`;
}

const DARK_COLOR_CACHE = new Array(101);
const LIGHT_COLOR_CACHE = new Array(101);
for (let a = 0; a <= 100; a++) {
    const alphaStr = (a / 100).toFixed(2);
    DARK_COLOR_CACHE[a] = `rgba(255, 255, 255, ${alphaStr})`;
    LIGHT_COLOR_CACHE[a] = `rgba(9, 9, 11, ${alphaStr})`;
}

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
let gestureState = 'IDLE'; // IDLE | HOVER | DUAL_PINCH_ZOOM | COMPRESS | BLOOM | SLAP
let anchorDualDist = 0.0;
let anchorDualScale = 1.0;

// Supernova Cosmic Burst & Inertia Reassembly
let burstActive = false;
let burstStartTime = 0;
const BURST_EXPAND_TIME = 1.9; // seconds of full 360° dispersion
const REASSEMBLY_TIME = 1.3;   // seconds of 3-phase gravitational reassembly
const TOTAL_BLOOM_TIME = BURST_EXPAND_TIME + REASSEMBLY_TIME;

// Smooth Idle-to-Hover Deceleration Timer
let handEnterTime = 0;
let hadHandsLastFrame = false;

let currentTheme = 'dark';
let sensitivity = 1.5;
let speedMult = 1.0;
let audioMode = 'both';
let hasHands = false;

// ═══════════════════════════════════════════════════════════════════════════
// 3. AMPLIFIED ACOUSTIC NEAR-FIELD FILTERING (2.5x Boosted Baseline)
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
            console.warn("Microphone access deferred:", err);
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

    ambientNoiseFloor = ambientNoiseFloor * 0.993 + rawEnergy * 0.007;

    let gatedEnergy = 0.0;
    const gateThreshold = ambientNoiseFloor * 1.4;
    if (rawEnergy > gateThreshold) {
        // Boosted baseline sensitivity gain (12.5x total multiplier)
        gatedEnergy = Math.min((rawEnergy - gateThreshold) * sensitivity * 12.5, 1.0);
    }

    smoothAudio += (gatedEnergy - smoothAudio) * 0.12;
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. REFINED GESTURE ARBITRATION & DUAL PINCH ZOOM
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

            handsMeter.innerText = `${handCount} HAND${handCount === 1 ? '' : 'S'}`;

            // Trigger Supernova Cosmic Particle Burst
            if (data.bloom) {
                burstActive = true;
                burstStartTime = performance.now() / 1000;
            }

            if (handCount > 0) {
                if (!hadHandsLastFrame) {
                    handEnterTime = performance.now() / 1000;
                }
                hadHandsLastFrame = true;
                hasHands = true;

                // ── STATE 1: DUAL-HAND PINCH ZOOM & SCALE ───────────────────
                if (data.dual_pinch) {
                    if (gestureState !== 'DUAL_PINCH_ZOOM') {
                        gestureState = 'DUAL_PINCH_ZOOM';
                        anchorDualDist = Math.max(data.two_hand_dist, 0.05);
                        anchorDualScale = currentScale;
                    }
                    targetOrbCenter.x = data.dual_pinch_center.x * width;
                    targetOrbCenter.y = data.dual_pinch_center.y * height;

                    const scaleRatio = data.two_hand_dist / anchorDualDist;
                    targetScale = Math.min(Math.max(anchorDualScale * scaleRatio, 0.40), 2.80);
                }
                // ── STATE 2: UNSTABLE PLASMA FIST COMPRESSION ───────────────
                else if (data.compress || data.hands.some(h => h.is_fist)) {
                    gestureState = 'COMPRESS';
                    targetScale = 0.58;
                    const primary = data.hands[0];
                    targetRotY = (primary.palm_x - 0.5) * 1.3;
                    targetRotX = (primary.palm_y - 0.5) * 1.3;
                }
                // ── STATE 3: MAGNETIC HOVER & ORIENTATION ───────────────────
                else {
                    gestureState = 'HOVER';
                    const primary = data.hands[0];
                    targetScale = primary.depth_scale || 1.0;

                    if (primary.pitch !== undefined) {
                        targetRotX = primary.pitch * 1.8;
                        targetRotY = primary.yaw * 1.8;
                        targetRotZ = primary.roll * 0.6;
                    } else {
                        targetRotY = (primary.palm_x - 0.5) * 2.8;
                        targetRotX = (primary.palm_y - 0.5) * 2.8;
                    }

                    // Open Palm Slap / Flick
                    if (data.slap_impulse && data.slap_impulse.active) {
                        angularVelY += data.slap_impulse.vx * 0.16;
                        angularVelX += data.slap_impulse.vy * 0.16;
                    }
                }
            } else {
                hadHandsLastFrame = false;
                hasHands = false;
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

// ═══════════════════════════════════════════════════════════════════════════
// 5. UI CONTROLS & FLOATING LIVE FEED HUD
// ═══════════════════════════════════════════════════════════════════════════
const settingsDrawer = document.getElementById('settings-drawer');
document.getElementById('settings-btn').onclick = () => settingsDrawer.classList.add('open');
document.getElementById('close-settings').onclick = () => settingsDrawer.classList.remove('open');

// Floating In-App Live Feed HUD Toggle
const cameraToggleBtn = document.getElementById('camera-feed-toggle');
const sensorHud = document.getElementById('sensor-hud');
const closeSensorHudBtn = document.getElementById('close-sensor-hud');
const sensorStreamImg = document.getElementById('sensor-stream-img');

function openSensorHud() {
    if (!sensorHud) return;
    sensorHud.classList.remove('hidden');
    if (cameraToggleBtn) cameraToggleBtn.classList.add('active');
    if (sensorStreamImg) {
        sensorStreamImg.src = '/video_feed?t=' + Date.now();
    }
}

function closeSensorHud() {
    if (!sensorHud) return;
    sensorHud.classList.add('hidden');
    if (cameraToggleBtn) cameraToggleBtn.classList.remove('active');
    if (sensorStreamImg) {
        sensorStreamImg.src = '';
        sensorStreamImg.removeAttribute('src');
    }
}

if (sensorStreamImg) {
    sensorStreamImg.onerror = () => {
        setTimeout(() => {
            if (sensorHud && !sensorHud.classList.contains('hidden')) {
                sensorStreamImg.src = '/video_feed?t=' + Date.now();
            }
        }, 1000);
    };
}

if (cameraToggleBtn) {
    cameraToggleBtn.addEventListener('click', () => {
        if (sensorHud && sensorHud.classList.contains('hidden')) {
            openSensorHud();
        } else {
            closeSensorHud();
        }
    });
}

if (closeSensorHudBtn) {
    closeSensorHudBtn.addEventListener('click', closeSensorHud);
}

// Theme and audio mode segmented toggles
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
// 6. HIGH-PERFORMANCE ZERO-ALLOCATION RENDER LOOP — Supernova & Plasma Spikes
// ═══════════════════════════════════════════════════════════════════════════
let time = 0;
const projectedNodes = new Array(POINT_COUNT);
for (let i = 0; i < POINT_COUNT; i++) {
    projectedNodes[i] = { x: 0, y: 0, z: 0, depth: 0, glyph: '·', alphaIndex: 100 };
}

function render() {
    requestAnimationFrame(render);
    const nowSec = performance.now() / 1000;
    time += 0.015 * speedMult;

    updateAudioEnergy();

    // Damped position & scale interpolation
    const springK = gestureState === 'DUAL_PINCH_ZOOM' ? 0.14 : 0.12;
    orbCenter.x += (targetOrbCenter.x - orbCenter.x) * springK;
    orbCenter.y += (targetOrbCenter.y - orbCenter.y) * springK;
    currentScale += (targetScale - currentScale) * springK;

    // Smooth Idle-to-Hover Deceleration: omega(t) = omega0 * exp(-t / 0.32)
    if (gestureState === 'IDLE') {
        targetRotY += 0.003 * speedMult; // Continuous slow auto-rotation (0.003 rad/frame)
    } else if (gestureState === 'HOVER') {
        const hoverElapsed = nowSec - handEnterTime;
        const decayRot = 0.003 * Math.exp(-hoverElapsed / 0.32);
        if (decayRot > 0.0002) {
            targetRotY += decayRot * speedMult;
        }
    }

    // Rotational kinematics
    rotX += (targetRotX - rotX) * 0.038 + angularVelX;
    rotY += (targetRotY - rotY) * 0.038 + angularVelY;
    rotZ += (targetRotZ - rotZ) * 0.038 + angularVelZ;

    // Viscous friction decay
    angularVelX *= 0.95;
    angularVelY *= 0.95;
    angularVelZ *= 0.95;

    // ── 3D Supernova Cosmic Burst & 3-Phase Gravitational Reassembly ────
    let burstTotalElapsed = 0.0;
    let inBurstPhase = false;
    let inReassemblyPhase = false;
    let reassemblyProgressFactor = 0.0;

    if (burstActive) {
        burstTotalElapsed = nowSec - burstStartTime;
        if (burstTotalElapsed < BURST_EXPAND_TIME) {
            inBurstPhase = true;
        } else if (burstTotalElapsed < TOTAL_BLOOM_TIME) {
            inReassemblyPhase = true;
            const tRe = burstTotalElapsed - BURST_EXPAND_TIME;

            // Phase 1 (0.0s – 0.5s): Slow initial inward drift (breaking outward momentum)
            if (tRe < 0.5) {
                reassemblyProgressFactor = 1.0 - 0.08 * (tRe / 0.5);
            }
            // Phase 2 (0.5s – 1.0s): Rapid non-linear gravitational acceleration toward core (a ~ 1/r^2)
            else if (tRe < 1.0) {
                const p = (tRe - 0.5) / 0.5;
                reassemblyProgressFactor = 0.92 * (1.0 - Math.pow(p, 2.8));
            }
            // Phase 3 (1.0s – 1.3s): Elastic damped spring relaxation settling cleanly
            else {
                const p = (tRe - 1.0) / 0.3;
                reassemblyProgressFactor = -0.12 * Math.sin(p * Math.PI * 2.5) * Math.exp(-6.0 * p);
            }
        } else {
            burstActive = false;
        }
    }

    ctx.clearRect(0, 0, width, height);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const cosZ = Math.cos(rotZ), sinZ = Math.sin(rotZ);

    const isCompress = gestureState === 'COMPRESS';
    const minBound = 0.55 * R0;
    const maxBound = 1.45 * R0;

    for (let i = 0; i < POINT_COUNT; i++) {
        const node = nodes[i];
        const bp = burstParticles[i];

        let px, py, pz;
        let nodeAlpha = 1.0;
        let glyphChar = '·';

        if (inBurstPhase || inReassemblyPhase) {
            // 360° Omnidirectional Cosmic Dispersion: P(t) = P0 + V * t * exp(-0.4 * t)
            const tPeak = BURST_EXPAND_TIME;
            const peakDisplacementFactor = tPeak * Math.exp(-bp.lambda * tPeak);

            const peakX = node.nx * R0 + bp.vx * peakDisplacementFactor;
            const peakY = node.ny * R0 + bp.vy * peakDisplacementFactor;
            const peakZ = node.nz * R0 + bp.vz * peakDisplacementFactor;

            if (inBurstPhase) {
                const tExp = burstTotalElapsed;
                const disp = tExp * Math.exp(-bp.lambda * tExp);
                px = node.nx * R0 + bp.vx * disp;
                py = node.ny * R0 + bp.vy * disp;
                pz = node.nz * R0 + bp.vz * disp;
                nodeAlpha = Math.max(0.18, 1.0 - (tExp / BURST_EXPAND_TIME) * 0.45);
            } else {
                const F = reassemblyProgressFactor;
                px = (node.nx * R0) + (peakX - node.nx * R0) * F;
                py = (node.ny * R0) + (peakY - node.ny * R0) * F;
                pz = (node.nz * R0) + (peakZ - node.nz * R0) * F;
                nodeAlpha = Math.min(1.0, 0.6 + (1.0 - Math.abs(F)) * 0.4);
            }

            // Distance-faded glyph scrambling during flight
            const glyphSeed = Math.floor(bp.seed + time * 18 + i) % BURST_CHARS.length;
            glyphChar = BURST_CHARS[glyphSeed];
        } else {
            // Fluid spherical surface acoustics & micro-ripples
            const idleWave = 0.012 * Math.sin(2 * node.theta + 3 * node.phi + time);
            const voiceWave = smoothAudio * 0.22 * Math.sin(4 * node.theta + 3 * node.phi + 2.5 * time);
            let deltaR = R0 * (idleWave + voiceWave);

            // Unstable High-Energy Plasma Fist Compression
            if (isCompress) {
                // Base radius contracts to 0.58 * R0 with +/- 3.5px micro-jitter
                const microVib = (Math.sin(time * 38 + i * 19) * 3.5);
                // 14–18 sharp dynamic radial energy spikes protruding outward to 1.35 * R0:
                // Delta_r_spike = R0 * 0.38 * max(0, sin(14*theta + 10*phi + 28*t))^3
                const spikeRaw = Math.sin(14 * node.theta + 10 * node.phi + 28 * time);
                const spike = R0 * 0.38 * Math.pow(Math.max(0, spikeRaw), 3.0);

                deltaR = -R0 * 0.42 + microVib + spike;
            }

            const clampedR = Math.max(minBound, Math.min(maxBound, R0 + deltaR));
            const effectiveR = clampedR * currentScale;

            px = node.nx * effectiveR;
            py = node.ny * effectiveR;
            pz = node.nz * effectiveR;
        }

        // 3D Euler Rotations (Y -> X -> Z)
        const x1 = px * cosY + pz * sinY;
        const z1 = -px * sinY + pz * cosY;
        const y2 = py * cosX - z1 * sinX;
        const z2 = py * sinX + z1 * cosX;
        const x3 = x1 * cosZ - y2 * sinZ;
        const y3 = x1 * sinZ + y2 * cosZ;

        // Perspective Projection
        const fov = 580;
        const cameraDist = 420;
        const depth = fov / (fov + z2 + cameraDist);
        const screenX = orbCenter.x + x3 * depth;
        const screenY = orbCenter.y + y3 * depth;

        if (!inBurstPhase && !inReassemblyPhase) {
            const normalizedZ = (z2 + (R0 * currentScale)) / (2 * R0 * currentScale + 0.001);
            const clampedZ = Math.max(0, Math.min(1, normalizedZ));

            if (isCompress) {
                const plasmaIdx = Math.min(
                    PLASMA_CHARS.length - 1,
                    Math.floor(clampedZ * (PLASMA_CHARS.length - 1))
                );
                glyphChar = PLASMA_CHARS[plasmaIdx];
            } else {
                const rawGlyphIdx = Math.min(
                    ASCII_CHARS.length - 1,
                    Math.floor(clampedZ * (ASCII_CHARS.length - 1))
                );
                glyphChar = ASCII_CHARS[rawGlyphIdx];
            }

            const normalizedDepth = (z2 + (R0 * currentScale)) / (2 * R0 * currentScale + 0.001);
            nodeAlpha = Math.min(Math.max(0.14 + normalizedDepth * 0.86, 0.14), 1.0);
        }

        const p = projectedNodes[i];
        p.x = screenX;
        p.y = screenY;
        p.z = z2;
        p.depth = depth;
        p.glyph = glyphChar;
        p.alphaIndex = Math.max(0, Math.min(100, Math.floor(nodeAlpha * 100)));
    }

    // Depth Sorting (Back-to-front)
    projectedNodes.sort((a, b) => a.z - b.z);

    const isDark = currentTheme === 'dark';
    const colorCache = isDark ? DARK_COLOR_CACHE : LIGHT_COLOR_CACHE;

    for (let i = 0; i < POINT_COUNT; i++) {
        const p = projectedNodes[i];

        const fontSize = Math.max(6, Math.min(80, Math.floor(16 * p.depth)));
        ctx.font = FONT_CACHE[fontSize] || FONT_CACHE[16];
        ctx.fillStyle = colorCache[p.alphaIndex];

        ctx.fillText(p.glyph, p.x, p.y);
    }
}
render();
