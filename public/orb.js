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
// 1. PRISTINE SPHERE GEOMETRY & 3D BURST KINEMATICS
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

    // Randomized cosmic explosion vectors (reaching 2.5x - 3.5x viewport scale)
    const speed = 820 + Math.sin(i * 12.3) * 380;
    const chaosX = (Math.cos(i * 7.1)) * 0.25;
    const chaosY = (Math.sin(i * 9.3)) * 0.25;
    const chaosZ = (Math.cos(i * 11.7)) * 0.25;
    const dir = { x: nx + chaosX, y: ny + chaosY, z: nz + chaosZ };
    const dirLen = Math.sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) || 1.0;

    burstParticles.push({
        vx: (dir.x / dirLen) * speed,
        vy: (dir.y / dirLen) * speed,
        vz: (dir.z / dirLen) * speed,
        lambda: 0.38 + ((i % 17) / 17.0) * 0.18,
        seed: i * 37
    });
}

// Standard ASCII density gradient
const ASCII_CHARS = ['·', '.', ':', '+', 'x', '*', '%', '0', '#', '@'];
// High-energy compressed plasma glyphs
const PLASMA_CHARS = ['@', '#', '%', '&', '8'];
// Cosmic supernova burst randomized glyphs
const BURST_CHARS = ['*', '+', 'x', '·', ':', '°', '¤', '#', '░', '▒'];

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
let lockedPinchHandId = null;
let anchorHand = { x: 0, y: 0, depth: 1.0 };
let anchorOrbPos = { x: width / 2, y: height / 2 };
let anchorScale = 1.0;
let anchorDualDist = 0.0;

// Cosmic Supernova Burst & Reassembly Dynamics
let burstActive = false;
let burstStartTime = 0;
const BURST_EXPAND_TIME = 1.9; // seconds of full expansion
const REASSEMBLY_TIME = 1.4;   // seconds of gravitational reassembly
const TOTAL_BLOOM_TIME = BURST_EXPAND_TIME + REASSEMBLY_TIME;

// Idle-to-Hover deceleration timer
let handEnterTime = 0;
let hadHandsLastFrame = false;

let currentTheme = 'dark';
let sensitivity = 1.5;
let speedMult = 1.0;
let audioMode = 'both';
let hasHands = false;

// ═══════════════════════════════════════════════════════════════════════════
// 3. AMPLIFIED ACOUSTIC NEAR-FIELD FILTERING (2.8x Boosted Gain)
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

    ambientNoiseFloor = ambientNoiseFloor * 0.993 + rawEnergy * 0.007;

    let gatedEnergy = 0.0;
    const gateThreshold = ambientNoiseFloor * 1.4;
    if (rawEnergy > gateThreshold) {
        // Boosted baseline sensitivity gain (12.1x total multiplier)
        gatedEnergy = Math.min((rawEnergy - gateThreshold) * sensitivity * 12.1, 1.0);
    }

    smoothAudio += (gatedEnergy - smoothAudio) * 0.12;
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

                const pinchingHands = data.hands.filter(h => h.is_pinching);

                // ── CASE A: DUAL PINCH (Both hands pinching) ────────────────
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

                    if (lockedPinchHandId) {
                        activeGrabHand = data.hands.find(h => h.id === lockedPinchHandId && h.is_pinching);
                        if (!activeGrabHand) {
                            lockedPinchHandId = null;
                        }
                    }

                    if (!activeGrabHand && pinchingHands.length > 0) {
                        activeGrabHand = pinchingHands[0];
                        lockedPinchHandId = activeGrabHand.id;
                    }

                    if (activeGrabHand) {
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

                    // State 3: Unstable High-Energy Fist Compress
                    if (primary.is_fist || data.compress) {
                        gestureState = 'COMPRESS';
                        targetScale = 0.62;
                        targetRotY = (primary.palm_x - 0.5) * 1.4;
                        targetRotX = (primary.palm_y - 0.5) * 1.4;
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
                hadHandsLastFrame = false;
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
// 5. UI CONTROLS & FLOATING LIVE FEED HUD
// ═══════════════════════════════════════════════════════════════════════════
const settingsDrawer = document.getElementById('settings-drawer');
document.getElementById('settings-btn').onclick = () => settingsDrawer.classList.add('open');
document.getElementById('close-settings').onclick = () => settingsDrawer.classList.remove('open');

// Floating In-App Live Feed HUD Picture-in-Picture Toggle
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
// 6. HIGH-PERFORMANCE RENDER LOOP — Cosmic Burst & Plasma Spikes
// ═══════════════════════════════════════════════════════════════════════════
let time = 0;
const projectedNodes = new Array(POINT_COUNT);
for (let i = 0; i < POINT_COUNT; i++) {
    projectedNodes[i] = { x: 0, y: 0, z: 0, depth: 0, glyph: '·', alpha: 1.0 };
}

function render() {
    requestAnimationFrame(render);
    const nowSec = performance.now() / 1000;
    time += 0.015 * speedMult;

    updateAudioEnergy();

    // Damped position & scale interpolation
    const isDualMold = gestureState === 'DUAL_MOLD';
    const springK = isDualMold ? 0.05 : 0.12;

    orbCenter.x += (targetOrbCenter.x - orbCenter.x) * springK;
    orbCenter.y += (targetOrbCenter.y - orbCenter.y) * springK;
    currentScale += (targetScale - currentScale) * springK;

    // Smooth Idle-to-Hover Deceleration: omega(t) = omega0 * exp(-t / 0.32)
    if (gestureState === 'IDLE') {
        targetRotY += 0.002 * speedMult;
    } else if (gestureState === 'HOVER') {
        const hoverElapsed = nowSec - handEnterTime;
        const decayRot = 0.002 * Math.exp(-hoverElapsed / 0.32);
        targetRotY += decayRot * speedMult;
    }

    // Smooth rotational kinematics
    const rotLerp = isDualMold ? 0.025 : 0.038;
    rotX += (targetRotX - rotX) * rotLerp + angularVelX;
    rotY += (targetRotY - rotY) * rotLerp + angularVelY;
    rotZ += (targetRotZ - rotZ) * rotLerp + angularVelZ;

    // Viscous friction decay
    angularVelX *= 0.95;
    angularVelY *= 0.95;
    angularVelZ *= 0.95;

    // ── Supernova Cosmic Burst & Gravitational Reassembly Calculation ────
    let burstTotalElapsed = 0.0;
    let inBurstPhase = false;
    let inReassemblyPhase = false;
    let reassemblyProgressFactor = 0.0; // 1.0 = peak explosion, 0.0 = equilibrium

    if (burstActive) {
        burstTotalElapsed = nowSec - burstStartTime;
        if (burstTotalElapsed < BURST_EXPAND_TIME) {
            inBurstPhase = true;
        } else if (burstTotalElapsed < TOTAL_BLOOM_TIME) {
            inReassemblyPhase = true;
            const tRe = burstTotalElapsed - BURST_EXPAND_TIME;

            // Phase 1 (0.0s – 0.6s): Slow initial inward drift (breaking momentum)
            if (tRe < 0.6) {
                reassemblyProgressFactor = 1.0 - 0.08 * (tRe / 0.6);
            }
            // Phase 2 (0.6s – 1.1s): Rapid non-linear acceleration toward core (a ~ 1/r^2)
            else if (tRe < 1.1) {
                const p = (tRe - 0.6) / 0.5;
                reassemblyProgressFactor = 0.92 * (1.0 - Math.pow(p, 2.8));
            }
            // Phase 3 (1.1s – 1.4s): Elastic damped spring settling into equilibrium
            else {
                const p = (tRe - 1.1) / 0.3;
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
    const minBound = 0.60 * R0;
    const maxBound = 1.40 * R0;

    for (let i = 0; i < POINT_COUNT; i++) {
        const node = nodes[i];
        const bp = burstParticles[i];

        let px, py, pz;
        let nodeAlpha = 1.0;
        let glyphChar = '·';

        if (inBurstPhase || inReassemblyPhase) {
            // Cosmic Supernova Burst & Reassembly
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
                nodeAlpha = Math.max(0.2, 1.0 - (tExp / BURST_EXPAND_TIME) * 0.4);
            } else {
                // Inward gravitational pull with inertia
                const F = reassemblyProgressFactor;
                px = (node.nx * R0) + (peakX - node.nx * R0) * F;
                py = (node.ny * R0) + (peakY - node.ny * R0) * F;
                pz = (node.nz * R0) + (peakZ - node.nz * R0) * F;
                nodeAlpha = Math.min(1.0, 0.6 + (1.0 - Math.abs(F)) * 0.4);
            }

            // Dynamic glyph randomization during flight
            const glyphSeed = Math.floor(bp.seed + time * 15 + i) % BURST_CHARS.length;
            glyphChar = BURST_CHARS[glyphSeed];
        } else {
            // Fluid spherical surface acoustics & micro-ripples
            const idleWave = 0.012 * Math.sin(2 * node.theta + 3 * node.phi + time);
            const voiceWave = smoothAudio * 0.22 * Math.sin(4 * node.theta + 3 * node.phi + 2.5 * time);
            let deltaR = R0 * (idleWave + voiceWave);

            // Unstable high-energy plasma compression with energy spikes
            if (isCompress) {
                // Core contracts to 0.62 * R0 with +/- 3.5px micro-vibrations
                const microVib = (Math.sin(time * 35 + i * 17) * 3.5);
                // Procedural radial energy spikes: Delta_r = R0 * 0.35 * max(0, sin(12*theta + 8*phi + 24*t))^3
                const spikeRaw = Math.sin(12 * node.theta + 8 * node.phi + 24 * time);
                const spike = R0 * 0.35 * Math.pow(Math.max(0, spikeRaw), 3.0);

                deltaR = -R0 * 0.38 + microVib + spike;
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

        // Select glyph density based on gesture state and depth
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
        p.alpha = nodeAlpha;
    }

    // Depth Sorting (Back-to-front)
    projectedNodes.sort((a, b) => a.z - b.z);

    const isDark = currentTheme === 'dark';

    for (let i = 0; i < POINT_COUNT; i++) {
        const p = projectedNodes[i];

        const fontSize = Math.max(6, Math.floor(16 * p.depth));
        ctx.font = `${fontSize}px 'Plus Jakarta Sans', -apple-system, sans-serif`;
        ctx.fillStyle = isDark
            ? `rgba(255, 255, 255, ${p.alpha.toFixed(2)})`
            : `rgba(9, 9, 11, ${p.alpha.toFixed(2)})`;

        ctx.fillText(p.glyph, p.x, p.y);
    }
}
render();
