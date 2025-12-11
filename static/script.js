
let lastSavedFeatures = null;


const videoElement = document.createElement("video");
videoElement.autoplay = true;
videoElement.playsInline = true;

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

const imgTag = document.getElementById("video-stream");

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }
    });

    videoElement.srcObject = stream;
    videoElement.onloadedmetadata = () => {
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      sendFramesLoop();
    };

  } catch (err) {
    console.error("Webcam error:", err);
    alert("Tidak bisa mengakses webcam.");
  }
}

async function sendFramesLoop() {
  try {
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    let frameData = canvas.toDataURL("image/jpeg", 0.8);

    const resp = await fetch("/api/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: frameData })
    });

    const data = await resp.json();

    if (data.image) {
      imgTag.src = "data:image/jpeg;base64," + data.image;
    }

  } catch (err) {
    console.error("Frame send error:", err);
  }

  requestAnimationFrame(sendFramesLoop);
}

startWebcam();


async function onSave() {
  const name = window.prompt("Enter gesture name to save:");
  if (!name) return;

  try {
    const resp = await fetch("/api/gesture");
    const data = await resp.json();

    if (!data.features) {
      alert("No hand detected. Please show your hand.");
      return;
    }

    const saveResp = await fetch("/api/save_gesture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gesture_name: name,
        features: data.features
      })
    });

    const result = await saveResp.json();
    const status = document.getElementById("status");

    if (saveResp.ok) {
      status.textContent = `Saved "${name}"`;
      updateGestureInfo();
    } else {
      status.textContent = `Save failed: ${result.message}`;
    }
  } catch (err) {
    console.error(err);
    alert("Error saving gesture.");
  }
}


async function updateGestureInfo() {
  try {
    const resp = await fetch("/api/gesture");
    const data = await resp.json();

    document.getElementById("gesture-name").textContent =
      data.gesture || "-";

    document.getElementById("confidence").textContent =
      data.confidence
        ? (data.confidence * 100).toFixed(1) + "%"
        : "-";

    document.getElementById("saved-count").textContent =
      data.saved_gestures ?? 0;

    lastSavedFeatures = data.features || null;

  } catch (err) {
    console.error("update error", err);
  }
}


async function toggleAutoSave() {
  try {
    const resp = await fetch("/api/auto_save/toggle", {
      method: "POST"
    });

    const data = await resp.json();
    const btn = document.getElementById("auto-save-btn");

    btn.textContent = data.auto_save_enabled
      ? "Auto-save: ON"
      : "Auto-save: OFF";

    btn.style.background = data.auto_save_enabled
      ? "#16a34a"
      : "#6b7280";

    document.getElementById("status").textContent = data.message;

  } catch (err) {
    console.error(err);
    alert("Failed to toggle auto-save.");
  }
}


function downloadExcel() {
  window.location.href = "/api/download/excel";
  document.getElementById("status").textContent = "Downloading Excel...";
}


// refresh gesture info
setInterval(updateGestureInfo, 600);
updateGestureInfo();
