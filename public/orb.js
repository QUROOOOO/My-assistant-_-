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
// 1. TYPEDARRAY ZERO-ALLOCATION GEOMETRY & 16 SOLAR FLARE CENTROIDS
// ═══════════════════════════════════════════════════════════════════════════
const POINT_COUNT = 1400;
const R0 = 230.0; // Base equilibrium radius (px)
const GOLDEN_ANGLE = Math.PI * (3.0 - Math.sqrt(5.0)); // ~2.399963 rad

// Static TypedArray buffers for Fibonacci nodes (Zero GC pressure)
const nodeNx = new Float32Array(POINT_COUNT);
const nodeNy = new Float32Array(POINT_COUNT);
const nodeNz = new Float32Array(POINT_COUNT);
const nodeTheta = new Float32Array(POINT_COUNT);
const nodePhi = new Float32Array(POINT_COUNT);

// Supernova Cosmic Burst Vectors
const burstVx = new Float32Array(POINT_COUNT);
const burstVy = new Float32Array(POINT_COUNT);
const burstVz = new Float32Array(POINT_COUNT);
const burstSeed = new Uint16Array(POINT_COUNT);

for (let i = 0; i < POINT_COUNT; i++) {
    const y = 1.0 - (i / (POINT_COUNT - 1.0)) * 2.0;
    const phi = Math.acos(Math.max(-1.0, Math.min(1.0, y)));
    const theta = GOLDEN_ANGLE * i;

    const nx = Math.sin(phi) * Math.cos(theta);
    const ny = Math.cos(phi);
    const nz = Math.sin(phi) * Math.sin(theta);

    nodeNx[i] = nx;
    nodeNy[i] = ny;
    nodeNz[i] = nz;
    nodeTheta[i] = theta;
    nodePhi[i] = phi;

    // 360° Omnidirectional Cosmic Dispersion (2.5x - 3.5x viewport scale)
    const speed = 900.0 + Math.sin(i * 13.7) * 420.0;
    const chaosX = Math.cos(i * 7.9) * 0.28;
    const chaosY = Math.sin(i * 11.3) * 0.28;
    const chaosZ = Math.cos(i * 14.1) * 0.28;
    const dirX = nx + chaosX;
    const dirY = ny + chaosY;
    const dirZ = nz + chaosZ;
    const dirLen = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ) || 1.0;

    burstVx[i] = (dirX / dirLen) * speed;
    burstVy[i] = (dirY / dirLen) * speed;
    burstVz[i] = (dirZ / dirLen) * speed;
    burstSeed[i] = (i * 41) % 1000;
}

// 16 Localized Solar Flare Centroid Unit Vectors (Fibonacci distributed)
const FLARE_COUNT = 16;
const flareAx = new Float32Array(FLARE_COUNT);
const flareAy = new Float32Array(FLARE_COUNT);
const flareAz = new Float32Array(FLARE_COUNT);
const flareEnergies = new Float32Array(FLARE_COUNT);

for (let k = 0; k < FLARE_COUNT; k++) {
    const y = 1.0 - (k / (FLARE_COUNT - 1.0)) * 2.0;
    const phi = Math.acos(Math.max(-1.0, Math.min(1.0, y)));
    const theta = GOLDEN_ANGLE * k * 3.5;
    flareAx[k] = Math.sin(phi) * Math.cos(theta);
    flareAy[k] = Math.cos(phi);
    flareAz[k] = Math.sin(phi) * Math.sin(theta);
    flareEnergies[k] = 0.0;
}

// Projection & Render Buffers
const projX = new Float32Array(POINT_COUNT);
const projY = new Float32Array(POINT_COUNT);
const projZ = new Float32Array(POINT_COUNT);
const projDepth = new Float32Array(POINT_COUNT);
const projAlphaIdx = new Uint8Array(POINT_COUNT);
const projGlyph = new Array(POINT_COUNT);
const renderOrder = new Uint16Array(POINT_COUNT);
for (let i = 0; i < POINT_COUNT; i++) renderOrder[i] = i;

// Density Gradients
const ASCII_CHARS = ['·', '.', ':', '+', 'x', '*', '%', '0', '#', '@'];
const PLASMA_CHARS = ['@', '#', '8', '%', '0'];
const BURST_CHARS = ['*', '+', '·', '%', '0', '@', 'x', '¤'];

// Zero-Allocation String Caches
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

let gestureState = 'IDLE'; // IDLE | HOVER | DUAL_PINCH_ZOOM | COMPRESS | BLOOM | SLAP
let anchorDualDist = 0.0;
let anchorDualScale = 1.0;

// Supernova Cosmic Burst & 3-Phase Gravitational Snap-Back
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
// 3. AMPLIFIED ACOUSTIC FILTERING & STOCHASTIC SOLAR FLARE ENGINE
// ═══════════════════════════════════════════════════════════════════════════
let audioCtx, analyser, biquadFilter, micSource;
let audioDataArray;
let smoothAudio = 0.0;
let prevRawAudio = 0.0;
let ambientNoiseFloor = 0.012;

function initAudio() {
    if (audioCtx) return;
    try {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AC();

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
    // Flare energy damping (0.88 snap-back spring decay)
    for (let k = 0; k < FLARE_COUNT; k++) {
        flareEnergies[k] *= 0.88;
    }

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
        gatedEnergy = Math.min((rawEnergy - gateThreshold) * sensitivity * 12.5, 1.0);

        // Stochastic Localized Solar Flare Trigger on Transient Audio Peaks
        const deltaRMS = rawEnergy - prevRawAudio;
        if (deltaRMS > 0.025) {
            // Trigger 1-3 random localized centroids
            const triggerCount = 1 + Math.floor(Math.random() * 3);
            for (let t = 0; t < triggerCount; t++) {
                const k = Math.floor(Math.random() * FLARE_COUNT);
                const peakMultiplier = 0.4 + Math.random() * 0.8;
                flareEnergies[k] = Math.min(1.4, flareEnergies[k] + gatedEnergy * peakMultiplier * 1.8);
            }
        }
    }
    prevRawAudio = rawEnergy;
    smoothAudio += (gatedEnergy - smoothAudio) * 0.14;
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
                // ── STATE 2: UNSTABLE PLASMA FIST COMPRESSION (0.42 * R0) ───
                else if (data.compress || data.hands.some(h => h.is_fist)) {
                    gestureState = 'COMPRESS';
                    targetScale = 0.42; // Compressed base equilibrium radius
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
// 6. HIGH-PERFORMANCE RENDER LOOP — Batched Glyph Drawing & Solar Flares
// ═══════════════════════════════════════════════════════════════════════════
let time = 0;

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
        targetRotY += 0.003 * speedMult;
    } else if (gestureState === 'HOVER') {
        const hoverElapsed = nowSec - handEnterTime;
        const decayRot = 0.003 * Math.exp(-hoverElapsed / 0.32);
        if (decayRot > 0.0002) {
            targetRotY += decayRot * speedMult;
        }
    }

    rotX += (targetRotX - rotX) * 0.038 + angularVelX;
    rotY += (targetRotY - rotY) * 0.038 + angularVelY;
    rotZ += (targetRotZ - rotZ) * 0.038 + angularVelZ;

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
            // Phase 3 (1.0s – 1.3s): Elastic damped spring relaxation settling cleanly into R0
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
    const minBound = 0.40 * R0;
    const maxBound = 1.50 * R0;
    const sigmaSq2 = 2.0 * (0.35 * 0.35); // 2 * sigma^2 for Gaussian solar flare bell curve

    for (let i = 0; i < POINT_COUNT; i++) {
        const nx = nodeNx[i];
        const ny = nodeNy[i];
        const nz = nodeNz[i];
        const theta = nodeTheta[i];
        const phi = nodePhi[i];

        let px, py, pz;
        let nodeAlpha = 1.0;
        let glyphChar = '·';

        if (inBurstPhase || inReassemblyPhase) {
            const tPeak = BURST_EXPAND_TIME;
            const peakDisp = tPeak * Math.exp(-0.40 * tPeak);

            const peakX = nx * R0 + burstVx[i] * peakDisp;
            const peakY = ny * R0 + burstVy[i] * peakDisp;
            const peakZ = nz * R0 + burstVz[i] * peakDisp;

            if (inBurstPhase) {
                const tExp = burstTotalElapsed;
                const disp = tExp * Math.exp(-0.40 * tExp);
                px = nx * R0 + burstVx[i] * disp;
                py = ny * R0 + burstVy[i] * disp;
                pz = nz * R0 + burstVz[i] * disp;
                nodeAlpha = Math.max(0.18, 1.0 - (tExp / BURST_EXPAND_TIME) * 0.45);
            } else {
                const F = reassemblyProgressFactor;
                px = (nx * R0) + (peakX - nx * R0) * F;
                py = (ny * R0) + (peakY - ny * R0) * F;
                pz = (nz * R0) + (peakZ - nz * R0) * F;
                nodeAlpha = Math.min(1.0, 0.6 + (1.0 - Math.abs(F)) * 0.4);
            }

            const glyphSeed = Math.floor(burstSeed[i] + time * 18 + i) % BURST_CHARS.length;
            glyphChar = BURST_CHARS[glyphSeed];
        } else {
            // Idle organic surface breathing
            const idleWave = 0.010 * Math.sin(2.0 * theta + 3.0 * phi + time);

            // Stochastic Localized Solar Flare Summation across 16 Centroids
            let flareSum = 0.0;
            for (let k = 0; k < FLARE_COUNT; k++) {
                if (flareEnergies[k] > 0.005) {
                    const dx = nx - flareAx[k];
                    const dy = ny - flareAy[k];
                    const dz = nz - flareAz[k];
                    const distSq = dx * dx + dy * dy + dz * dz;
                    flareSum += flareEnergies[k] * Math.exp(-distSq / sigmaSq2);
                }
            }
            let deltaR = R0 * (idleWave + flareSum * 0.35);

            // Unstable High-Energy Plasma Core Fist Compression (0.42 * R0, 45Hz Jitter, Micro-Arcs)
            if (isCompress) {
                // High-frequency 45Hz chaotic micro-vibration (+/- 2.5px)
                const microVib = Math.sin(time * 90.0 + i * 23.0) * 2.5;

                // 8–12 Stochastic plasma micro-arcs darting across compressed surface
                const arc1 = Math.sin(18.0 * theta + 14.0 * phi + 35.0 * time);
                const arc2 = Math.cos(22.0 * theta - 16.0 * phi + 28.0 * time);
                const plasmaArc = (Math.max(0.0, arc1) * Math.max(0.0, arc2)) * R0 * 0.32;

                deltaR = -R0 * 0.58 + microVib + plasmaArc;
            }

            const clampedR = Math.max(minBound, Math.min(maxBound, R0 + deltaR));
            const effectiveR = clampedR * currentScale;

            px = nx * effectiveR;
            py = ny * effectiveR;
            pz = nz * effectiveR;
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
                // Solid high-density glyphs & high opacity (0.95 - 1.0)
                const plasmaIdx = Math.min(
                    PLASMA_CHARS.length - 1,
                    Math.floor(clampedZ * (PLASMA_CHARS.length - 1))
                );
                glyphChar = PLASMA_CHARS[plasmaIdx];
                nodeAlpha = 0.95 + clampedZ * 0.05;
            } else {
                const rawGlyphIdx = Math.min(
                    ASCII_CHARS.length - 1,
                    Math.floor(clampedZ * (ASCII_CHARS.length - 1))
                );
                glyphChar = ASCII_CHARS[rawGlyphIdx];
                nodeAlpha = Math.min(Math.max(0.14 + clampedZ * 0.86, 0.14), 1.0);
            }
        }

        projX[i] = screenX;
        projY[i] = screenY;
        projZ[i] = z2;
        projDepth[i] = depth;
        projGlyph[i] = glyphChar;
        projAlphaIdx[i] = Math.max(0, Math.min(100, Math.floor(nodeAlpha * 100)));
    }

    // Depth Sorting (Back-to-front index array sort)
    renderOrder.sort((a, b) => projZ[a] - projZ[b]);

    const isDark = currentTheme === 'dark';
    const colorCache = isDark ? DARK_COLOR_CACHE : LIGHT_COLOR_CACHE;

    // Batched Glyph Rendering
    for (let j = 0; j < POINT_COUNT; j++) {
        const i = renderOrder[j];

        const fontSize = Math.max(6, Math.min(80, Math.floor(16 * projDepth[i])));
        ctx.font = FONT_CACHE[fontSize] || FONT_CACHE[16];
        ctx.fillStyle = colorCache[projAlphaIdx[i]];

        ctx.fillText(projGlyph[i], projX[i], projY[i]);
    }
}
render();
