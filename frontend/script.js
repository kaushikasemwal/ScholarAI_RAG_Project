/**
 * ScholarAI — index.html Script
 * Handles: auth guard, file upload, generate triggers, Firestore saves.
 * Results are NOT rendered here — they live on notes.html.
 */

import { auth, db } from "./firebase-config.js";
import {
  collection, addDoc, getDocs, updateDoc,
  query, where, orderBy, serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

import {
  initAuthGuard, showUserInfo, createSignOutHandler
} from "./shared/auth-guard.js";
import { showToast, showSuccess, showError, showInfo, showWarning } from "./shared/toast.js";
import { formatBytes, capitalize, delay, copyText, escapeHtml } from "./shared/utils.js";
import { uploadFile, generateOutput, getApiBase, createJob, getJobStatus, listJobs, connectJobWebSocket } from "./shared/api.js";

// ─── CONFIG ────────────────────────────────────────────────────────
const API_BASE = getApiBase();

// ─── STATE ────────────────────────────────────────────────────────
let currentUser = null;
let uploadedFileId = null;
let currentDocRef = null;
let currentFile = null;
let fileQueue = [];
let activeGenerations = new Map(); // type -> { jobId, ws, progressCard }

// ─── AUTH GUARD ───────────────────────────────────────────────────
initAuthGuard(auth, async (user) => {
  currentUser = user;
  showUserInfo(user);
  await loadHistory();
  initTheme();
  initKeyboardShortcuts();
});

window.handleSignOut = createSignOutHandler(auth);

// ─── DOM REFS ─────────────────────────────────────────────────────
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const viewNotesContainer = document.getElementById("viewNotesContainer");
const generateGrid = document.querySelector(".generate-grid");

// ─── THEME ────────────────────────────────────────────────────────
function initTheme() {
  // Check for saved theme or system preference
  const savedTheme = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = savedTheme || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
  
  // Create theme toggle button in header
  createThemeToggle();
  
  // Listen for system theme changes
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!localStorage.getItem("theme")) {
      document.documentElement.setAttribute("data-theme", e.matches ? "dark" : "light");
    }
  });
}

function createThemeToggle() {
  const headerRight = document.querySelector(".header-right");
  if (!headerRight || document.getElementById("themeToggle")) return;
  
  const toggle = document.createElement("button");
  toggle.id = "themeToggle";
  toggle.className = "theme-toggle";
  toggle.setAttribute("aria-label", "Toggle theme");
  toggle.innerHTML = `
    <span class="theme-toggle-icon" aria-hidden="true"></span>
  `;
  toggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const newTheme = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
  });
  
  // Insert before sign out button
  const signOutBtn = document.getElementById("signOutBtn");
  headerRight.insertBefore(toggle, signOutBtn);
}

// ─── KEYBOARD SHORTCUTS ──────────────────────────────────────────
function initKeyboardShortcuts() {
  // Create shortcuts hint
  const hint = document.createElement("div");
  hint.className = "shortcuts-hint";
  hint.innerHTML = `
    <kbd>U</kbd> Upload  <kbd>Enter</kbd> Generate  <kbd>T</kbd> Theme  <kbd>?</kbd> Help
  `;
  document.body.appendChild(hint);
  
  document.addEventListener("keydown", (e) => {
    // Ignore if typing in input
    if (e.target.matches("input, textarea, [contenteditable]")) return;
    
    switch (e.key.toLowerCase()) {
      case "u":
        e.preventDefault();
        fileInput?.click();
        break;
      case "enter":
        if (uploadedFileId && !document.getElementById("btn-all").disabled) {
          e.preventDefault();
          window.generateAll();
        }
        break;
      case "t":
        e.preventDefault();
        document.getElementById("themeToggle")?.click();
        break;
      case "?":
      case "/":
        e.preventDefault();
        showShortcutsHelp();
        break;
      case "escape":
        closeAllModals();
        break;
    }
  });
}

function showShortcutsHelp() {
  showInfo(`
    <strong>Keyboard Shortcuts:</strong><br>
    <kbd>U</kbd> - Upload file<br>
    <kbd>Enter</kbd> - Generate All<br>
    <kbd>T</kbd> - Toggle theme<br>
    <kbd>?</kbd> / <kbd>/</kbd> - Show this help<br>
    <kbd>Esc</kbd> - Close modals
  `, 8000);
}

function closeAllModals() {
  document.querySelectorAll(".quiz-modal-overlay").forEach(m => m.remove());
}

// ─── DRAG-AND-DROP ────────────────────────────────────────────────
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault(); 
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault(); 
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) {
    [...e.dataTransfer.files].forEach(f => handleFile(f));
  }
});
dropZone.addEventListener("click", (e) => {
  if (e.target.classList.contains("btn-browse")) return;
  fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    [...fileInput.files].forEach(f => handleFile(f));
  }
});

// ─── FILE QUEUE ───────────────────────────────────────────────────
function handleFile(file) {
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (![".pdf", ".pptx", ".ppt"].includes(ext)) {
    showError("Only PDF and PPTX files are supported."); 
    return;
  }
  
  // Add to queue
  const queueItem = {
    file,
    id: crypto.randomUUID(),
    status: "pending", // pending, uploading, completed, error
    uploadedFileId: null
  };
  fileQueue.push(queueItem);
  renderFileQueue();
  
  // Start upload if first in queue
  if (fileQueue.filter(f => f.status === "pending" || f.status === "uploading").length === 1) {
    processQueue();
  }
}

function renderFileQueue() {
  // Create queue container if not exists
  let queueContainer = document.getElementById("fileQueue");
  if (!queueContainer) {
    queueContainer = document.createElement("div");
    queueContainer.id = "fileQueue";
    queueContainer.className = "file-queue";
    fileInfo.parentNode.insertBefore(queueContainer, fileInfo.nextSibling);
  }
  
  queueContainer.innerHTML = fileQueue.map(item => `
    <div class="queue-item ${item.status}" data-id="${item.id}">
      <span class="queue-icon">${getFileIcon(item.file.name)}</span>
      <div class="queue-info">
        <div class="queue-name">${escapeHtml(item.file.name)}</div>
        <div class="queue-status">${getStatusText(item.status)}</div>
        ${(item.status === "uploading" || item.status === "completed" || item.status === "error") ? 
          `<div class="queue-progress ${item.status === "uploading" ? "skeleton" : ""}" style="width: ${item.status === "completed" ? "100%" : item.status === "error" ? "100%" : "0%"}; background: ${item.status === "completed" ? "linear-gradient(90deg, var(--success), var(--success-light))" : item.status === "error" ? "var(--error)" : ""}"></div>` : ""}
      </div>
      <button class="queue-remove" onclick="removeFromQueue('${item.id}')" aria-label="Remove file">&times;</button>
    </div>
  `).join("");
}

function getFileIcon(filename) {
  const ext = filename.split(".").pop().toLowerCase();
  return ext === "pdf" ? "📄" : "📊";
}

function getStatusText(status) {
  const texts = {
    pending: "Waiting…",
    uploading: "Uploading…",
    completed: "Ready",
    error: "Failed"
  };
  return texts[status] || status;
}

window.removeFromQueue = function(id) {
  const index = fileQueue.findIndex(f => f.id === id);
  if (index !== -1) {
    fileQueue.splice(index, 1);
    renderFileQueue();
    // If current file was removed, clear main file info
    if (currentFile && fileQueue.length === 0) {
      clearFile();
    }
  }
};

async function processQueue() {
  const uploadProgress = document.getElementById("uploadProgress");
  const uploadProgressFill = document.getElementById("uploadProgressFill");
  const uploadProgressText = document.getElementById("uploadProgressText");
  const uploadSpeed = document.getElementById("uploadSpeed");
  
  for (const item of fileQueue) {
    if (item.status !== "pending") continue;
    
    item.status = "uploading";
    renderFileQueue();
    
    // Show main progress bar for first file
    if (!currentFile) {
      uploadProgress.style.display = "block";
      uploadProgressFill.style.width = "0%";
      uploadProgressText.textContent = `Uploading ${item.file.name}...`;
      uploadSpeed.textContent = "";
    }
    
    try {
      const startTime = Date.now();
      const data = await uploadFile(item.file, (percent, loaded, total) => {
        if (!currentFile) {
          uploadProgressFill.style.width = `${percent}%`;
          uploadProgressText.textContent = `Uploading ${item.file.name}... ${percent}%`;
          
          // Calculate upload speed
          const elapsed = (Date.now() - startTime) / 1000;
          if (elapsed > 0) {
            const speed = loaded / elapsed;
            uploadSpeed.textContent = `${formatBytes(speed)}/s`;
          }
        }
        
        // Also update queue item progress
        const queueItem = document.querySelector(`.queue-item[data-id="${item.id}"] .queue-progress`);
        if (queueItem) {
          queueItem.style.width = `${percent}%`;
          queueItem.classList.remove("skeleton");
          queueItem.style.background = "linear-gradient(90deg, var(--primary), var(--secondary))";
        }
      });
      
      item.uploadedFileId = data.file_id;
      item.status = "completed";
      
      // If this is the first successful upload, set as current
      if (!currentFile) {
        currentFile = item.file;
        uploadedFileId = data.file_id;
        fileName.textContent = item.file.name;
        fileSize.textContent = formatBytes(item.file.size);
        fileInfo.style.display = "flex";
        
        // Complete progress bar
        uploadProgressFill.style.width = "100%";
        uploadProgressText.textContent = "Upload complete!";
        uploadSpeed.textContent = "";
        
        // Create Firestore session
        currentDocRef = await addDoc(collection(db, "sessions"), {
          uid: currentUser.uid,
          fileId: data.file_id,
          fileName: item.file.name,
          fileSize: item.file.size,
          createdAt: serverTimestamp(),
          summary: null,
          quiz: null,
          audioB64: null,
          videoUrl: null,
        });
        
        // Hide progress bar after delay
        setTimeout(() => {
          uploadProgress.style.display = "none";
        }, 1500);
      }
      
      showSuccess(`${item.file.name} uploaded securely.`);
    } catch (err) {
      item.status = "error";
      
      // Show error in progress bar
      if (!currentFile) {
        uploadProgressFill.style.width = "100%";
        uploadProgressFill.style.background = "var(--error)";
        uploadProgressText.textContent = `Failed: ${err.message}`;
        uploadSpeed.textContent = "";
        
        setTimeout(() => {
          uploadProgress.style.display = "none";
          uploadProgressFill.style.background = "";
        }, 3000);
      }
      
      showError(`Failed to upload ${item.file.name}: ${err.message}`);
      console.error(err);
    }
    
    renderFileQueue();
    await delay(300);
  }
}

window.clearFile = function () {
  currentFile = null; uploadedFileId = null; currentDocRef = null;
  fileQueue = [];
  fileInfo.style.display = "none";
  fileInput.value = "";
  
  // Reset progress bar
  const uploadProgress = document.getElementById("uploadProgress");
  const uploadProgressFill = document.getElementById("uploadProgressFill");
  if (uploadProgress) {
    uploadProgress.style.display = "none";
    uploadProgressFill.style.width = "0%";
    uploadProgressFill.style.background = "";
  }
  
  const banner = document.getElementById("viewNotesBtn");
  if (banner) banner.remove();
  const queueContainer = document.getElementById("fileQueue");
  if (queueContainer) queueContainer.remove();
};

// ─── GENERATION WITH WEBSOCKET PROGRESS ──────────────────────────
window.generateOutput = async function (type) {
  if (!uploadedFileId) {
    showError("Please upload a document first."); return;
  }
  
  const btn = document.getElementById(`btn-${type}`);
  setButtonLoading(btn, true);
  
  // Create progress card
  const progressCard = createProgressCard(type);
  activeGenerations.set(type, { progressCard });
  
  try {
    // Create async job
    const jobData = await createJob(type, uploadedFileId);
    const jobId = jobData.job_id;
    activeGenerations.get(type).jobId = jobId;
    
    // Connect WebSocket for real-time progress
    const ws = connectJobWebSocket(jobId, (data) => {
      updateProgressCard(type, data);
    });
    activeGenerations.get(type).ws = ws;
    
    // Poll for completion as fallback
    pollJobCompletion(type, jobId);
    
  } catch (err) {
    showError(`${capitalize(type)} failed to start.`);
    console.error(err);
    removeProgressCard(type);
  } finally {
    setButtonLoading(btn, false);
  }
};

function createProgressCard(type) {
  const card = document.createElement("div");
  card.className = "gen-progress-card";
  card.id = `progress-${type}`;
  card.innerHTML = `
    <div class="gen-progress-header">
      <span class="gen-progress-icon">${getTypeIcon(type)}</span>
      <span class="gen-progress-title">${capitalize(type)} Generation</span>
      <span class="gen-progress-status pending">Pending</span>
    </div>
    <div class="gen-progress-bar">
      <div class="gen-progress-fill"></div>
    </div>
    <div class="gen-progress-time">
      <span>Elapsed: <span class="elapsed-time">0s</span></span>
      <span class="eta">Estimating…</span>
    </div>
    <button class="gen-progress-cancel" onclick="cancelGeneration('${type}')">Cancel</button>
  `;
  
  // Insert after generate grid
  generateGrid.parentNode.insertBefore(card, generateGrid.nextSibling);
  
  const startTime = Date.now();
  const elapsedInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const el = card.querySelector(".elapsed-time");
    if (el) el.textContent = `${elapsed}s`;
  }, 1000);
  
  card.dataset.elapsedInterval = elapsedInterval;
  return card;
}

function getTypeIcon(type) {
  const icons = { summary: "🧠", quiz: "❓", audio: "🔊", video: "🎬" };
  return icons[type] || "⚙️";
}

function updateProgressCard(type, data) {
  const card = document.getElementById(`progress-${type}`);
  if (!card) return;
  
  const statusEl = card.querySelector(".gen-progress-status");
  const fillEl = card.querySelector(".gen-progress-fill");
  const etaEl = card.querySelector(".eta");
  
  if (data.progress !== undefined) {
    fillEl.style.width = `${data.progress}%`;
  }
  
  if (data.status) {
    statusEl.textContent = capitalize(data.status);
    statusEl.className = `gen-progress-status ${data.status}`;
  }
  
  if (data.status === "completed" && data.result) {
    clearInterval(card.dataset.elapsedInterval);
    etaEl.textContent = "Done!";
    
    // Persist result
    persistResult(type, data.result).then(() => {
      showSuccess(`${capitalize(type)} generated.`);
      showViewNotesBtn();
    });
    
    // Auto-remove after delay
    setTimeout(() => removeProgressCard(type), 3000);
  } else if (data.status === "failed") {
    clearInterval(card.dataset.elapsedInterval);
    statusEl.textContent = "Failed";
    statusEl.className = "gen-progress-status failed";
    etaEl.textContent = data.error || "Unknown error";
    showError(`${capitalize(type)} failed: ${data.error || "Unknown error"}`);
  } else if (data.status === "running") {
    // Estimate remaining time based on progress
    const elapsed = (Date.now() - parseInt(card.dataset.startTime || Date.now())) / 1000;
    if (data.progress > 5) {
      const estimatedTotal = elapsed / (data.progress / 100);
      const remaining = Math.max(0, estimatedTotal - elapsed);
      etaEl.textContent = `~${Math.round(remaining)}s remaining`;
    }
  }
}

window.cancelGeneration = function(type) {
  const gen = activeGenerations.get(type);
  if (gen?.ws) {
    gen.ws.close();
  }
  if (gen?.jobId) {
    // Could implement job cancellation endpoint
    showWarning(`${capitalize(type)} cancellation requested.`);
  }
  removeProgressCard(type);
};

function removeProgressCard(type) {
  const card = document.getElementById(`progress-${type}`);
  if (card) {
    clearInterval(card.dataset.elapsedInterval);
    card.remove();
  }
  activeGenerations.delete(type);
}

async function pollJobCompletion(type, jobId) {
  // Fallback polling in case WebSocket fails
  for (let i = 0; i < 120; i++) { // Max 2 minutes
    await delay(1000);
    try {
      const status = await getJobStatus(jobId);
      if (status.status === "completed" || status.status === "failed") {
        updateProgressCard(type, {
          progress: 100,
          status: status.status,
          result: status.result,
          error: status.error
        });
        break;
      }
    } catch (err) {
      console.warn("Poll failed:", err);
    }
  }
}

// ─── GENERATE ALL ─────────────────────────────────────────────────
window.generateAll = async function () {
  if (!uploadedFileId) {
    showError("Please upload a document first."); return;
  }
  
  const btn = document.getElementById("btn-all");
  btn.disabled = true; 
  btn.textContent = "Generating…";
  
  // Start all generations in parallel
  const types = ["summary", "quiz", "audio", "video"];
  const promises = types.map(type => window.generateOutput(type));
  
  await Promise.all(promises);
  
  btn.disabled = false; 
  btn.textContent = "⚡ Generate All & Open Notes";
  
  if (currentDocRef) {
    showSuccess("All done! Opening your notes…");
    await delay(800);
    window.location.href = `notes.html?id=${currentDocRef.id}`;
  }
};

// ─── PERSIST TO FIRESTORE ────────────────────────────────────────
async function persistResult(type, data) {
  if (type === "summary") {
    await saveToFirestore({ 
      summary: data.summary || "",
      summaryFallbackUsed: data.fallback_used,
      summaryFallbackReason: data.fallback_reason,
      summaryModelUsed: data.model_used
    });
  }

  if (type === "quiz" && data.questions) {
    await saveToFirestore({ quiz: data.questions });
  }

  if (type === "audio" && data.audio_url) {
    const backendUrl = `${API_BASE}${data.audio_url}`;
    await saveAudioAsBase64(backendUrl);
  }

  if (type === "video" && data.video_url) {
    await saveVideoAsBase64(`${API_BASE}${data.video_url}`);
  }
}

// ─── AUDIO → BASE64 → FIRESTORE ──────────────────────────────────
async function saveAudioAsBase64(url) {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return;
    const blob = await resp.blob();
    if (blob.size < 500) return;
    if (blob.size > 900_000) {
      await saveToFirestore({ audioB64: url });
      return;
    }
    const b64 = await blobToBase64(blob);
    await saveToFirestore({ audioB64: b64 });
  } catch (err) {
    console.warn("Could not save audio:", err.message);
  }
}

// ─── VIDEO → BASE64 → FIRESTORE ──────────────────────────────────
async function saveVideoAsBase64(url) {
  try {
    showToast("Saving video…");
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("Could not fetch video from backend");
    const blob = await resp.blob();
    if (blob.size < 10_000) throw new Error("Video too small, likely failed");
    if (blob.size > 900_000) {
      await saveToFirestore({ videoUrl: url });
      showSuccess("Video saved (temporary link).");
      return;
    }
    const b64 = await blobToBase64(blob);
    await saveToFirestore({ videoUrl: b64 });
    showSuccess("Video saved.");
  } catch (err) {
    console.warn("Could not save video:", err.message);
    showError("Video generated but could not be saved.");
  }
}

// ─── BLOB → BASE64 HELPER ────────────────────────────────────────
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// ─── FIRESTORE SAVE ──────────────────────────────────────────────
async function saveToFirestore(fields) {
  if (!currentDocRef) return;
  try { await updateDoc(currentDocRef, fields); }
  catch (err) { console.warn("Firestore save failed:", err); }
}

// ─── HISTORY (updates My Notes badge count) ──────────────────────
async function loadHistory() {
  if (!currentUser) return;
  try {
    const q = query(
      collection(db, "sessions"),
      where("uid", "==", currentUser.uid),
      orderBy("createdAt", "desc")
    );
    const snap = await getDocs(q);
    if (snap.size > 0) {
      const btn = document.querySelector(".btn-my-notes");
      if (btn) btn.textContent = `📚 My Notes (${snap.size})`;
    }
  } catch (err) {
    if (err.message?.includes("index")) {
      console.warn("Firestore index needed:", err);
    }
  }
}

// ─── VIEW NOTES BANNER ───────────────────────────────────────────
function showViewNotesBtn() {
  if (!currentDocRef || document.getElementById("viewNotesBtn")) return;
  if (!viewNotesContainer) return;
  const div = document.createElement("div");
  div.id = "viewNotesBtn";
  div.className = "view-notes-banner";
  div.innerHTML = `
    <span>✅ Content saved to your account.</span>
    <a href="notes.html?id=${currentDocRef.id}" class="btn-view-notes">View Full Notes →</a>`;
  viewNotesContainer.appendChild(div);
}

// ─── HELPERS ─────────────────────────────────────────────────────
function setButtonLoading(btn, loading) {
  if (!btn) return;
  btn.disabled = loading;
  const text = btn.querySelector(".btn-text");
  const spinner = btn.querySelector(".btn-spinner");
  if (text) text.style.display = loading ? "none" : "inline";
  if (spinner) spinner.style.display = loading ? "inline" : "none";
}

// Export copyText for global use
window.copyText = (id) => copyText(id, showToast);