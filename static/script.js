let lastSavedFeatures = null;


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


setInterval(updateGestureInfo, 600);
updateGestureInfo();
