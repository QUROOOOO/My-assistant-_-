const canvas = document.getElementById('ascii-canvas');
const ctx = canvas.getContext('2d');
const fxCanvas = document.getElementById('fx-canvas');
const fxCtx = fxCanvas.getContext('2d');

const handsMeter = document.getElementById('hands-meter');
const wsStatus = document.getElementById('ws-status');
const hudElement = document.querySelector('.hud');
const settingsDrawer = document.getElementById('settings-drawer');
const sensorHud = document.getElementById('sensor-hud');
const liveFeedCanvas = document.getElementById('live-feed-canvas');
const liveFeedCtx = liveFeedCanvas ? liveFeedCanvas.getContext('2d') : null;

let width, height, dpr;
function resize() {
    dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    fxCanvas.width = width * dpr;
    fxCanvas.height = height * dpr;
    fxCtx.scale(dpr, dpr);
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

// Supernova Cosmic Burst Vectors (360° omnidirectional expansion)
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

    // 360° Omnidirectional Cosmic Dispersion (3.0x viewport scale)
    const speed = 950.0 + Math.sin(i * 13.7) * 450.0;
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
const PLASMA_CHARS = ['@', '#', '%', '&', '8'];
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
// 2. ORB STATE & GENTLE CONTINUOUS MOTION KINEMATICS
// ═══════════════════════════════════════════════════════════════════════════
let orbCenter = { x: width / 2, y: height / 2 };
let targetOrbCenter = { x: width / 2, y: height / 2 };
let rotX = 0, rotY = 0, rotZ = 0;
let targetRotX = 0, targetRotY = 0, targetRotZ = 0;
let angularVelX = 0, angularVelY = 0, angularVelZ = 0;
let currentScale = 1.0;
let targetScale = 1.0;

let gestureState = 'IDLE'; // IDLE | HOVER | GRAB | DUAL_PINCH | COMPRESS | BLOOM | SWIPE | SNAP
let anchorDualDist = 0.0;
let anchorDualScale = 1.0;
let anchorGrabHand = { x: 0, y: 0, depth: 1.0 };
let anchorOrbPos = { x: width / 2, y: height / 2 };
let anchorGrabScale = 1.0;

// Supernova Cosmic Burst & 3-Phase Gravitational Snap-Back
let burstActive = false;
let burstStartTime = 0;
const BURST_EXPAND_TIME = 1.8; // seconds of full 360° dispersion
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
// 3. 3.5-SECOND CINEMATIC DUST DISINTEGRATION & DETERMINISTIC REINTEGRATION
// ═══════════════════════════════════════════════════════════════════════════
let isDisintegrated = false;
let blipAnimationActive = false;
let blipMode = 'DISINTEGRATE'; // 'DISINTEGRATE' | 'MATERIALIZE'
let blipStartTime = 0;
const BLIP_DURATION = 3.5; // 3.5 seconds cinematic duration

const DUST_PARTICLE_COUNT = 4000;
const dustParticles = [];

for (let p = 0; p < DUST_PARTICLE_COUNT; p++) {
    dustParticles.push({
        x: 0, y: 0,
        x0: 0, y0: 0,
        spawnX: 0, spawnY: 0,
        vx0: 0, vy0: 0,
        size: 1.0 + Math.random() * 2.0,
        alpha: 1.0,
        curlFreq: 0.012 + Math.random() * 0.018,
        curlAmp: 1.4 + Math.random() * 2.2,
        seed: Math.random() * 100,
        colorType: Math.random() > 0.4 ? 'silver' : 'charcoal'
    });
}

function collectUiAndOrbTargets() {
    const targets = [];

    // 1,400 Orb Nodes
    for (let i = 0; i < POINT_COUNT; i++) {
        targets.push({ x: projX[i] || (width / 2), y: projY[i] || (height / 2) });
    }

    // Visible DOM UI Elements
    const uiSelectors = ['.brand', '.status-pill', '#camera-feed-toggle', '#settings-btn', '.minimal-pill'];
    uiSelectors.forEach(sel => {
        const el = document.querySelector(sel);
        if (el) {
            const rect = el.getBoundingClientRect();
            const sampleCount = 90;
            for (let s = 0; s < sampleCount; s++) {
                targets.push({
                    x: rect.left + Math.random() * rect.width,
                    y: rect.top + Math.random() * rect.height
                });
            }
        }
    });

    while (targets.length < DUST_PARTICLE_COUNT) {
        targets.push({
            x: orbCenter.x + (Math.random() - 0.5) * R0 * 2.2,
            y: orbCenter.y + (Math.random() - 0.5) * R0 * 2.2
        });
    }

    return targets;
}

function triggerSnapEffect() {
    const now = performance.now() / 1000;
    blipStartTime = now;
    blipAnimationActive = true;

    if (!isDisintegrated) {
        // ── 3.5s DISINTEGRATION ("THE BLIP") ───────────────────────────
        blipMode = 'DISINTEGRATE';
        const targets = collectUiAndOrbTargets();

        for (let i = 0; i < DUST_PARTICLE_COUNT; i++) {
            const dp = dustParticles[i];
            const t = targets[i % targets.length];
            dp.x0 = t.x;
            dp.y0 = t.y;
            dp.x = t.x;
            dp.y = t.y;
            // Upward-right wind force: V_x = 1.8 + rand(0, 2.2), V_y = -2.2 - rand(0, 2.0)
            dp.vx0 = 1.8 + Math.random() * 2.2;
            dp.vy0 = -2.2 - Math.random() * 2.0;
            dp.alpha = 0.95;
        }

        // Fade DOM UI and primary canvas
        hudElement.classList.add('disintegrated');
        canvas.classList.add('disintegrated');
        settingsDrawer.classList.add('disintegrated');
        if (sensorHud) sensorHud.classList.add('disintegrated');

        isDisintegrated = true;
    } else {
        // ── 3.5s DETERMINISTIC REVERSE MATERIALIZATION ─────────────────
        blipMode = 'MATERIALIZE';
        const targets = collectUiAndOrbTargets();

        for (let i = 0; i < DUST_PARTICLE_COUNT; i++) {
            const dp = dustParticles[i];
            const t = targets[i % targets.length];
            dp.x0 = t.x;
            dp.y0 = t.y;
            // Spawn off-screen top-right corner
            dp.spawnX = width + 40 + Math.random() * 320;
            dp.spawnY = -40 - Math.random() * 320;
            dp.x = dp.spawnX;
            dp.y = dp.spawnY;
            dp.alpha = 0.15;
        }

        isDisintegrated = false;
    }
}

function updateBlipFX(nowSec) {
    if (!blipAnimationActive) return;

    const elapsed = nowSec - blipStartTime;
    fxCtx.clearRect(0, 0, width, height);

    if (blipMode === 'DISINTEGRATE') {
        const progress = Math.min(elapsed / BLIP_DURATION, 1.0);
        const tScale = 1.0 + 0.5 * elapsed;

        for (let i = 0; i < DUST_PARTICLE_COUNT; i++) {
            const dp = dustParticles[i];
            const curlX = Math.sin(dp.y * dp.curlFreq + elapsed * 2.5 + dp.seed) * dp.curlAmp;
            const curlY = Math.cos(dp.x * dp.curlFreq + elapsed * 2.5 + dp.seed) * dp.curlAmp;

            dp.x += (dp.vx0 * tScale + curlX) * 1.3;
            dp.y += (dp.vy0 * tScale + curlY) * 1.3;

            // Continuous alpha fade: alpha(t) = max(0, 1 - t / 3.5)
            dp.alpha = Math.max(0.0, 1.0 - progress);

            if (dp.alpha > 0.01) {
                fxCtx.fillStyle = dp.colorType === 'silver'
                    ? `rgba(195, 195, 210, ${dp.alpha.toFixed(2)})`
                    : `rgba(90, 90, 105, ${(dp.alpha * 0.85).toFixed(2)})`;
                fxCtx.fillRect(dp.x, dp.y, dp.size, dp.size);
            }
        }

        if (progress >= 1.0) {
            blipAnimationActive = false;
            fxCtx.clearRect(0, 0, width, height);
        }
    } else if (blipMode === 'MATERIALIZE') {
        const progress = Math.min(elapsed / BLIP_DURATION, 1.0);
        // Non-linear gravitational easing: easeOutCubic(t / 3.5)
        const ease = 1.0 - Math.pow(1.0 - progress, 3.0);

        for (let i = 0; i < DUST_PARTICLE_COUNT; i++) {
            const dp = dustParticles[i];

            // Harmonic damped settling
            const spiralCurlX = Math.sin(elapsed * 3.5 + dp.seed) * (1.0 - progress) * 16.0;
            const spiralCurlY = Math.cos(elapsed * 3.5 + dp.seed) * (1.0 - progress) * 16.0;

            dp.x = dp.spawnX + (dp.x0 - dp.spawnX) * ease + spiralCurlX;
            dp.y = dp.spawnY + (dp.y0 - dp.spawnY) * ease + spiralCurlY;
            dp.alpha = Math.min(1.0, progress * 1.15);

            fxCtx.fillStyle = dp.colorType === 'silver'
                ? `rgba(205, 195, 225, ${dp.alpha.toFixed(2)})`
                : `rgba(120, 110, 140, ${(dp.alpha * 0.9).toFixed(2)})`;
            fxCtx.fillRect(dp.x, dp.y, dp.size, dp.size);
        }

        if (progress >= 1.0) {
            blipAnimationActive = false;
            fxCtx.clearRect(0, 0, width, height);
            // Restore DOM and Canvas Visibility
            hudElement.classList.remove('disintegrated');
            canvas.classList.remove('disintegrated');
            settingsDrawer.classList.remove('disintegrated');
            if (sensorHud) sensorHud.classList.remove('disintegrated');
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. MULTIMODAL ACOUSTIC TRANSIENT ANALYZER (2.8 kHz High-Pass Filter)
// ═══════════════════════════════════════════════════════════════════════════
let audioCtx, analyser, biquadFilter, micSource;
let snapHighPassFilter, snapAnalyser, snapDataArray;
let audioDataArray;
let smoothAudio = 0.0;
let prevRawAudio = 0.0;
let prevSnapEnergy = 0.0;
let cAudio = 0.0;
let cVision = 0.0;
let lastSnapTriggerTime = 0.0;
let ambientNoiseFloor = 0.012;

function initAudio() {
    if (audioCtx) return;
    try {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AC();

        // Vocal Bandpass Filter (180Hz – 4200Hz, Q = 0.65)
        biquadFilter = audioCtx.createBiquadFilter();
        biquadFilter.type = 'bandpass';
        biquadFilter.frequency.value = 2190;
        biquadFilter.Q.value = 0.65;

        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.25;
        audioDataArray = new Uint8Array(analyser.frequencyBinCount);

        // High-Frequency Acoustic Snap Transient Filter (fc = 2800 Hz, Q = 1.4)
        snapHighPassFilter = audioCtx.createBiquadFilter();
        snapHighPassFilter.type = 'highpass';
        snapHighPassFilter.frequency.value = 2800;
        snapHighPassFilter.Q.value = 1.4;

        snapAnalyser = audioCtx.createAnalyser();
        snapAnalyser.fftSize = 256;
        snapAnalyser.smoothingTimeConstant = 0.10;
        snapDataArray = new Uint8Array(snapAnalyser.frequencyBinCount);

        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            micSource = audioCtx.createMediaStreamSource(stream);
            micSource.connect(biquadFilter);
            biquadFilter.connect(analyser);

            micSource.connect(snapHighPassFilter);
            snapHighPassFilter.connect(snapAnalyser);
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
    for (let k = 0; k < FLARE_COUNT; k++) {
        flareEnergies[k] *= 0.88;
    }

    // Continuous acoustic confidence decay
    cAudio = Math.max(0.0, cAudio - 0.04);

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

        const deltaRMS = rawEnergy - prevRawAudio;
        if (deltaRMS > 0.025) {
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

    // High-Frequency Acoustic Transient Evaluation
    if (snapAnalyser && snapDataArray) {
        snapAnalyser.getByteFrequencyData(snapDataArray);
        let snapSum = 0;
        for (let i = 0; i < snapDataArray.length; i++) {
            snapSum += snapDataArray[i];
        }
        const highEnergy = (snapSum / snapDataArray.length) / 255.0;
        const dEnergyHigh = highEnergy - prevSnapEnergy;
        prevSnapEnergy = highEnergy;

        if (dEnergyHigh > 0.035) {
            cAudio = Math.min(1.0, (dEnergyHigh - 0.035) * 18.0 + 0.55);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. REFINED GESTURE ARBITRATION & MULTIMODAL SNAP FUSION
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
            const now = performance.now() / 1000;

            handsMeter.innerText = `${handCount} HAND${handCount === 1 ? '' : 'S'}`;

            if (data.c_vision !== undefined) {
                cVision = data.c_vision;
            }

            // ── MULTIMODAL SNAP FUSION TRIGGER ─────────────────────────────
            // Condition A: Pure High-Confidence Vision (C_vision >= 0.82)
            // Condition B: Fused Vision + Sound (C_vision >= 0.50 AND C_audio >= 0.55)
            const conditionA = cVision >= 0.82 || data.event === 'SNAP' || data.snap;
            const conditionB = cVision >= 0.50 && cAudio >= 0.55;

            if ((conditionA || conditionB) && (now - lastSnapTriggerTime) > 2.0) {
                lastSnapTriggerTime = now;
                triggerSnapEffect();
            }

            // Trigger Supernova Cosmic Particle Burst
            if (data.bloom || data.state === 'BLOOM') {
                burstActive = true;
                burstStartTime = now;
            }

            if (handCount > 0) {
                if (!hadHandsLastFrame) {
                    handEnterTime = now;
                }
                hadHandsLastFrame = true;
                hasHands = true;

                // ── STATE 1: DUAL-HAND PINCH ZOOM & SCALE ───────────────────
                if (data.state === 'DUAL_PINCH' || data.dual_pinch) {
                    if (gestureState !== 'DUAL_PINCH') {
                        gestureState = 'DUAL_PINCH';
                        anchorDualDist = Math.max(data.two_hand_dist, 0.05);
                        anchorDualScale = currentScale;
                    }
                    targetOrbCenter.x = data.dual_pinch_center.x * width;
                    targetOrbCenter.y = data.dual_pinch_center.y * height;

                    const scaleRatio = data.two_hand_dist / anchorDualDist;
                    targetScale = Math.min(Math.max(anchorDualScale * scaleRatio, 0.40), 2.80);
                }
                // ── STATE 2: SINGLE-HAND PINCH DRAG (GRAB) ──────────────────
                else if (data.state === 'GRAB' || data.grab_hand) {
                    const grabHand = data.hands.find(h => h.id === data.grab_hand) || data.hands[0];
                    if (gestureState !== 'GRAB') {
                        gestureState = 'GRAB';
                        anchorGrabHand = {
                            x: grabHand.palm_x,
                            y: grabHand.palm_y,
                            depth: grabHand.depth_scale || 1.0
                        };
                        anchorOrbPos = { ...orbCenter };
                        anchorGrabScale = currentScale;
                    }

                    const dx = (grabHand.palm_x - anchorGrabHand.x) * width * 1.3;
                    const dy = (grabHand.palm_y - anchorGrabHand.y) * height * 1.3;
                    targetOrbCenter.x = anchorOrbPos.x + dx;
                    targetOrbCenter.y = anchorOrbPos.y + dy;

                    const depthRatio = (grabHand.depth_scale || 1.0) / anchorGrabHand.depth;
                    targetScale = Math.min(Math.max(anchorGrabScale * depthRatio * 0.85, 0.35), 2.2);
                }
                // ── STATE 3: UNSTABLE PLASMA FIST COMPRESSION (0.40 * R0) ───
                else if (data.state === 'COMPRESS' || data.compress || data.hands.some(h => h.is_fist)) {
                    gestureState = 'COMPRESS';
                    targetScale = 0.40;
                    const primary = data.hands[0];
                    targetRotY = (primary.palm_x - 0.5) * 1.3;
                    targetRotX = (primary.palm_y - 0.5) * 1.3;
                }
                // ── STATE 4: MAGNETIC HOVER & ORIENTATION ───────────────────
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

                    // Open Palm Slap / Swipe Momentum
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
// 6. TRUE 60 FPS WEBSOCKET BINARY LIVE FEED STREAM
// ═══════════════════════════════════════════════════════════════════════════
let liveFeedWs = null;
const cameraToggleBtn = document.getElementById('camera-feed-toggle');
const closeSensorHudBtn = document.getElementById('close-sensor-hud');

function openSensorHud() {
    if (!sensorHud) return;
    sensorHud.classList.remove('hidden');
    if (cameraToggleBtn) cameraToggleBtn.classList.add('active');

    if (!liveFeedWs || liveFeedWs.readyState !== WebSocket.OPEN) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        liveFeedWs = new WebSocket(`${protocol}//${window.location.host}/ws/live_feed`);
        liveFeedWs.binaryType = 'blob';

        liveFeedWs.onmessage = async (evt) => {
            if (evt.data instanceof Blob) {
                try {
                    const bmp = await createImageBitmap(evt.data);
                    if (liveFeedCtx) {
                        liveFeedCtx.drawImage(bmp, 0, 0, 480, 270);
                    }
                    if (bmp.close) bmp.close();
                } catch (e) {
                    // Bitmap decode frame skip
                }
            }
        };

        liveFeedWs.onclose = () => {
            liveFeedWs = null;
        };
    }
}

function closeSensorHud() {
    if (!sensorHud) return;
    sensorHud.classList.add('hidden');
    if (cameraToggleBtn) cameraToggleBtn.classList.remove('active');

    if (liveFeedWs) {
        liveFeedWs.close();
        liveFeedWs = null;
    }
    if (liveFeedCtx) {
        liveFeedCtx.clearRect(0, 0, 480, 270);
    }
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

document.getElementById('settings-btn').onclick = () => settingsDrawer.classList.add('open');
document.getElementById('close-settings').onclick = () => settingsDrawer.classList.remove('open');

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
// 7. HIGH-PERFORMANCE RENDER LOOP WITH GENTLE CONTINUOUS DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════
let time = 0;

function render() {
    requestAnimationFrame(render);
    const nowSec = performance.now() / 1000;
    time += 0.015 * speedMult;

    updateAudioEnergy();
    updateBlipFX(nowSec);

    // Gentle Continuous Exponential Smoothing (0.038 - 0.050)
    orbCenter.x += (targetOrbCenter.x - orbCenter.x) * 0.045;
    orbCenter.y += (targetOrbCenter.y - orbCenter.y) * 0.045;
    currentScale += (targetScale - currentScale) * 0.038;

    // Smooth Idle-to-Hover Deceleration
    if (gestureState === 'IDLE') {
        targetRotY += 0.003 * speedMult;
    } else if (gestureState === 'HOVER') {
        const hoverElapsed = nowSec - handEnterTime;
        const decayRot = 0.003 * Math.exp(-hoverElapsed / 0.32);
        if (decayRot > 0.0002) {
            targetRotY += decayRot * speedMult;
        }
    }

    rotX += (targetRotX - rotX) * 0.040 + angularVelX;
    rotY += (targetRotY - rotY) * 0.040 + angularVelY;
    rotZ += (targetRotZ - rotZ) * 0.040 + angularVelZ;

    angularVelX *= 0.95;
    angularVelY *= 0.95;
    angularVelZ *= 0.95;

    // Skip drawing primary orb if disintegrated into FX canvas
    if (isDisintegrated && !blipAnimationActive) {
        ctx.clearRect(0, 0, width, height);
        return;
    }

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

            // Phase 1 (0.0s – 0.5s): Slow initial inward drift
            if (tRe < 0.5) {
                reassemblyProgressFactor = 1.0 - 0.08 * (tRe / 0.5);
            }
            // Phase 2 (0.5s – 1.0s): Rapid gravitational acceleration
            else if (tRe < 1.0) {
                const p = (tRe - 0.5) / 0.5;
                reassemblyProgressFactor = 0.92 * (1.0 - Math.pow(p, 2.8));
            }
            // Phase 3 (1.0s – 1.3s): Elastic spring settling into R0
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
    const minBound = 0.38 * R0;
    const maxBound = 1.50 * R0;
    const sigmaSq2 = 2.0 * (0.35 * 0.35);

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
            const peakDisp = tPeak * Math.exp(-0.35 * tPeak);

            const peakX = nx * R0 + burstVx[i] * peakDisp;
            const peakY = ny * R0 + burstVy[i] * peakDisp;
            const peakZ = nz * R0 + burstVz[i] * peakDisp;

            if (inBurstPhase) {
                const tExp = burstTotalElapsed;
                const disp = tExp * Math.exp(-0.35 * tExp);
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
            const idleWave = 0.010 * Math.sin(2.0 * theta + 3.0 * phi + time);

            // 16 Localized Solar Flares
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

            // Unstable Energy Ball (Fist Compression 0.40 * R0)
            if (isCompress) {
                const microVib = Math.sin(time * 90.0 + i * 23.0) * 2.0;
                const arc1 = Math.sin(18.0 * theta + 14.0 * phi + 35.0 * time);
                const arc2 = Math.cos(22.0 * theta - 16.0 * phi + 28.0 * time);
                const plasmaArc = (Math.max(0.0, arc1) * Math.max(0.0, arc2)) * R0 * 0.30;

                deltaR = -R0 * 0.60 + microVib + plasmaArc;
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
                const plasmaIdx = Math.min(
                    PLASMA_CHARS.length - 1,
                    Math.floor(clampedZ * (PLASMA_CHARS.length - 1))
                );
                glyphChar = PLASMA_CHARS[plasmaIdx];
                nodeAlpha = 1.0;
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

    renderOrder.sort((a, b) => projZ[a] - projZ[b]);

    const isDark = currentTheme === 'dark';
    const colorCache = isDark ? DARK_COLOR_CACHE : LIGHT_COLOR_CACHE;

    for (let j = 0; j < POINT_COUNT; j++) {
        const i = renderOrder[j];

        const fontSize = Math.max(6, Math.min(80, Math.floor(16 * projDepth[i])));
        ctx.font = FONT_CACHE[fontSize] || FONT_CACHE[16];
        ctx.fillStyle = colorCache[projAlphaIdx[i]];

        ctx.fillText(projGlyph[i], projX[i], projY[i]);
    }
}
render();
