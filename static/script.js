import { FuzzyART } from "./fuzzyart.js";

const video = document.getElementById("video");
const gestureText = document.getElementById("gesture");
const artText = document.getElementById("art-gesture");

// ========== LOAD TFJS MODEL ==========
let model;
(async () => {
  model = await tf.loadLayersModel("model/model.json");
  console.log("ML model loaded!");
})();

// ========== INIT ART ==========
const art = new FuzzyART();

// ========== INIT MEDIAPIPE ==========
const hands = new Hands({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

async function processFrame() {
  await hands.send({ image: video });
  requestAnimationFrame(processFrame);
}

video.onloadeddata = () => processFrame();

// ========== CALLBACK ==========
hands.onResults(async results => {
  if (!results.multiHandLandmarks) {
    gestureText.textContent = "-";
    artText.textContent = "-";
    return;
  }

  const lm = results.multiHandLandmarks[0];

  // ====== Format Landmark ======
  let input = [];
  for (let i = 0; i < lm.length; i++) {
    input.push(lm[i].x);
    input.push(lm[i].y);
    input.push(lm[i].z);
  }

  const tensor = tf.tensor([input]);

  // ====== ML Prediction ======
  if (model) {
    const pred = model.predict(tensor);
    const idx = pred.argMax(1).dataSync()[0];

    gestureText.textContent = `ML Model: ${idx}`;
  }

  // ====== ART Prediction ======
  const artClass = art.predict(input);
  artText.textContent = `ART Class: ${artClass}`;
});
