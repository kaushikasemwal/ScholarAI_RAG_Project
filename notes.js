/**
 * notes.js — Session Notes Page
 * Loads a single Firestore session by ID from the URL query param
 * and renders summary, quiz, audio, video in a tabbed layout.
 * Enhanced with: interactive quiz, audio speed control, video PiP, export, accessibility
 */

import { auth, db } from "./firebase-config.js";
import { doc, getDoc } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

import {
  initAuthGuard, showUserInfo, createSignOutHandler
} from "./shared/auth-guard.js";
import { showToast, showSuccess, showError, showInfo } from "./shared/toast.js";
import { formatBytes, formatShortDate, escapeHtml, copyText } from "./shared/utils.js";

// ─── AUTH GUARD ───────────────────────────────────────────────────
initAuthGuard(auth, async (user) => {
  showUserInfo(user);
  await loadSession();
  initTheme();
});

window.handleSignOut = createSignOutHandler(auth);

// ─── THEME ────────────────────────────────────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = savedTheme || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
  
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!localStorage.getItem("theme")) {
      document.documentElement.setAttribute("data-theme", e.matches ? "dark" : "light");
    }
  });
}

// ─── LOAD SESSION ────────────────────────────────────────────────
async function loadSession() {
  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get("id");

  if (!sessionId) {
    showErrorMsg("No session ID provided.");
    return;
  }

  try {
    const snap = await getDoc(doc(db, "sessions", sessionId));
    if (!snap.exists()) {
      showErrorMsg("Session not found. It may have been deleted.");
      return;
    }
    renderSession({ id: snap.id, ...snap.data() });
  } catch (err) {
    showErrorMsg("Failed to load session: " + err.message);
  }
}

function showErrorMsg(msg) {
  document.getElementById("notesLoading").innerHTML = `
    <p style="color:var(--red-accent);font-size:1rem;">${msg}</p>
    <a href="index.html" style="color:var(--amber);margin-top:1rem;display:inline-block;">← Back to Upload</a>`;
}

// ─── RENDER SESSION ──────────────────────────────────────────────
function renderSession(session) {
  document.getElementById("notesLoading").style.display = "none";
  document.getElementById("notesPage").style.display = "block";

  document.title = `ScholarAI — ${session.fileName || "Notes"}`;
  document.getElementById("notesFileName").textContent = session.fileName || "Untitled";

  const date = formatShortDate(session.createdAt);
  document.getElementById("notesFileDate").textContent = date;
  document.getElementById("notesFileSize").textContent = session.fileSize ? formatBytes(session.fileSize) : "";

  // Badges
  const badges = document.getElementById("notesBadges");
  if (session.summary) badges.innerHTML += '<span class="badge badge-green">Summary</span>';
  if (session.quiz) badges.innerHTML += '<span class="badge badge-blue">Quiz</span>';
  if (session.audioB64) badges.innerHTML += '<span class="badge badge-amber">Audio</span>';
  if (session.videoUrl) badges.innerHTML += '<span class="badge badge-teal">Video</span>';
  if (session.summaryFallbackUsed) badges.innerHTML += '<span class="badge badge-amber" title="Used extractive fallback">Fallback</span>';

  // Hide tabs for missing content
  if (!session.summary) hideTab("summary");
  if (!session.quiz) hideTab("quiz");
  if (!session.audioB64) hideTab("audio");
  if (!session.videoUrl) hideTab("video");

  // Activate first available tab
  const available = ["summary", "quiz", "audio", "video"]
    .find(t => session[t === "audio" ? "audioB64" : t === "video" ? "videoUrl" : t]);
  if (available) switchTab(available);

  // ── Summary ──────────────────────────────────────────────────
  if (session.summary) {
    const fallbackNotice = session.summaryFallbackUsed 
      ? `<div class="fallback-notice" style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:8px;padding:0.75rem;margin-bottom:1rem;font-size:0.85rem;color:var(--amber);">
        ⚠️ <strong>Note:</strong> This summary was generated using extractive fallback (${escapeHtml(session.summaryFallbackReason || "ML models unavailable")}). 
        For best results, ensure the backend has internet access for Pegasus model.
      </div>`
      : "";
    document.getElementById("summaryText").innerHTML = fallbackNotice + `<p>${escapeHtml(session.summary)}</p>`;
  }

  // ── Quiz ─────────────────────────────────────────────────────
  if (session.quiz?.length) {
    window._quizData = session.quiz;
    document.getElementById("quizContent").innerHTML =
      session.quiz.map((q, i) => `
        <div class="quiz-preview-item">
          <span class="q-num">Q${i + 1}.</span>
          <span>${escapeHtml(q.question)}</span>
        </div>`).join("");
    document.getElementById("btnStartQuiz").style.display = "inline-flex";
  }

  // ── Audio ─────────────────────────────────────────────────────
  if (session.audioB64) {
    setupAudioPlayer(session.audioB64, session.fileName);
  }

  // ── Video ─────────────────────────────────────────────────────
  if (session.videoUrl) {
    setupVideoPlayer(session.videoUrl, session.fileName);
  }
}

// ─── AUDIO PLAYER SETUP ──────────────────────────────────────────
function setupAudioPlayer(src, fileName) {
  const player = document.getElementById("audioPlayer");
  const dlBtn = document.getElementById("audioDownload");
  const audioBody = document.getElementById("audioBody");
  
  player.src = src;
  player.load();
  dlBtn.href = src;
  dlBtn.download = `${(fileName || "summary").replace(/\.[^.]+$/, "")}_audio.mp3`;
  dlBtn.style.display = "inline-block";

  // Add speed control
  const speedSelect = document.createElement("select");
  speedSelect.className = "audio-speed-select";
  speedSelect.setAttribute("aria-label", "Playback speed");
  [0.5, 0.75, 1, 1.25, 1.5, 2].forEach(speed => {
    const opt = document.createElement("option");
    opt.value = speed;
    opt.textContent = `${speed}x`;
    if (speed === 1) opt.selected = true;
    speedSelect.appendChild(opt);
  });
  speedSelect.addEventListener("change", () => {
    player.playbackRate = parseFloat(speedSelect.value);
  });

  // Add keyboard controls
  player.addEventListener("keydown", (e) => {
    switch (e.key) {
      case " ": case "k": e.preventDefault(); player.paused ? player.play() : player.pause(); break;
      case "ArrowLeft": e.preventDefault(); player.currentTime = Math.max(0, player.currentTime - 10); break;
      case "ArrowRight": e.preventDefault(); player.currentTime = Math.min(player.duration, player.currentTime + 10); break;
      case "ArrowUp": e.preventDefault(); player.volume = Math.min(1, player.volume + 0.1); break;
      case "ArrowDown": e.preventDefault(); player.volume = Math.max(0, player.volume - 0.1); break;
      case "m": e.preventDefault(); player.muted = !player.muted; break;
    }
  });

  // Wrap player with controls
  const wrapper = document.createElement("div");
  wrapper.className = "audio-player-wrapper";
  wrapper.innerHTML = `
    <div class="audio-player-controls">
      <span class="audio-time" id="audioCurrentTime">0:00</span>
    </div>
  `;
  
  player.parentNode.insertBefore(wrapper, player);
  wrapper.appendChild(player);
  wrapper.querySelector(".audio-player-controls").appendChild(speedSelect);
  wrapper.querySelector(".audio-player-controls").appendChild(dlBtn);
  
  const timeDisplay = wrapper.querySelector("#audioCurrentTime");
  player.addEventListener("timeupdate", () => {
    timeDisplay.textContent = formatTime(player.currentTime) + " / " + formatTime(player.duration);
  });
}

// ─── VIDEO PLAYER SETUP ──────────────────────────────────────────
function setupVideoPlayer(src, fileName) {
  const player = document.getElementById("videoPlayer");
  const dlBtn = document.getElementById("videoDownload");
  const videoBody = document.getElementById("videoBody");
  
  player.src = src;
  player.load();
  dlBtn.href = src;
  dlBtn.download = `${(fileName || "summary").replace(/\.[^.]+$/, "")}_video.mp4`;
  dlBtn.style.display = "inline-block";

  // Add Picture-in-Picture button
  const pipBtn = document.createElement("button");
  pipBtn.className = "video-pip-btn";
  pipBtn.textContent = "⛶ PiP";
  pipBtn.setAttribute("aria-label", "Toggle Picture-in-Picture");
  pipBtn.addEventListener("click", async () => {
    try {
      if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
      } else if (document.pictureInPictureEnabled) {
        await player.requestPictureInPicture();
      }
    } catch (err) {
      console.warn("PiP failed:", err);
    }
  });

  // Add keyboard controls
  player.addEventListener("keydown", (e) => {
    switch (e.key) {
      case " ": case "k": e.preventDefault(); player.paused ? player.play() : player.pause(); break;
      case "ArrowLeft": e.preventDefault(); player.currentTime = Math.max(0, player.currentTime - 10); break;
      case "ArrowRight": e.preventDefault(); player.currentTime = Math.min(player.duration, player.currentTime + 10); break;
      case "ArrowUp": e.preventDefault(); player.volume = Math.min(1, player.volume + 0.1); break;
      case "ArrowDown": e.preventDefault(); player.volume = Math.max(0, player.volume - 0.1); break;
      case "m": e.preventDefault(); player.muted = !player.muted; break;
      case "f": e.preventDefault(); if (player.requestFullscreen) player.requestFullscreen(); break;
    }
  });

  // Wrap player
  const wrapper = document.createElement("div");
  wrapper.className = "video-player-wrapper";
  player.parentNode.insertBefore(wrapper, player);
  wrapper.appendChild(player);
  wrapper.appendChild(pipBtn);

  player.onerror = () => {
    videoBody.innerHTML = `
      <p style="color:#EF4444;font-size:0.9rem;">
        Video link may have expired. Re-upload the file and regenerate.
      </p>
      <a class="btn-download" href="${src}" download>Try Download</a>`;
  };
}

// ─── TIME FORMATTING ─────────────────────────────────────────────
function formatTime(seconds) {
  if (isNaN(seconds)) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

// ─── TABS ────────────────────────────────────────────────────────
window.switchTab = function (tab) {
  document.querySelectorAll(".notes-tab").forEach(t => {
    t.classList.toggle("active", t.dataset.tab === tab);
  });
  document.querySelectorAll(".notes-panel").forEach(p => {
    p.style.display = p.id === `panel-${tab}` ? "block" : "none";
  });
  
  // Announce tab change for screen readers
  const activeTab = document.querySelector(`.notes-tab[data-tab="${tab}"]`);
  if (activeTab) {
    activeTab.setAttribute("aria-selected", "true");
    announceToScreenReader(`${activeTab.textContent.trim()} tab selected`);
  }
  
  document.querySelectorAll(".notes-tab").forEach(t => {
    if (t.dataset.tab !== tab) t.setAttribute("aria-selected", "false");
  });
};

function hideTab(tab) {
  const btn = document.querySelector(`.notes-tab[data-tab="${tab}"]`);
  if (btn) btn.style.display = "none";
}

// ─── SCREEN READER ANNOUNCEMENTS ─────────────────────────────────
function announceToScreenReader(message) {
  let liveRegion = document.getElementById("sr-announcer");
  if (!liveRegion) {
    liveRegion = document.createElement("div");
    liveRegion.id = "sr-announcer";
    liveRegion.setAttribute("role", "status");
    liveRegion.setAttribute("aria-live", "polite");
    liveRegion.setAttribute("aria-atomic", "true");
    liveRegion.style.position = "absolute";
    liveRegion.style.width = "1px";
    liveRegion.style.height = "1px";
    liveRegion.style.padding = "0";
    liveRegion.style.margin = "-1px";
    liveRegion.style.overflow = "hidden";
    liveRegion.style.clip = "rect(0, 0, 0, 0)";
    liveRegion.style.whiteSpace = "nowrap";
    liveRegion.style.border = "0";
    document.body.appendChild(liveRegion);
  }
  liveRegion.textContent = message;
}

// ─── INTERACTIVE QUIZ ENGINE ─────────────────────────────────────
let _quizIndex = 0, _quizScore = 0, _quizAnswered = false;

window.startQuiz = function () {
  if (!window._quizData?.length) return;
  _quizIndex = 0; _quizScore = 0; _quizAnswered = false;
  document.getElementById("quizModal").style.display = "flex";
  document.body.style.overflow = "hidden";
  // Focus management for accessibility
  setTimeout(() => {
    document.getElementById("quizQuestion")?.focus();
  }, 100);
  renderQuestion();
};

window.closeQuiz = function () {
  document.getElementById("quizModal").style.display = "none";
  document.body.style.overflow = "";
};

function renderQuestion() {
  const questions = window._quizData;
  const q = questions[_quizIndex];
  _quizAnswered = false;

  const pct = (_quizIndex / questions.length) * 100;
  document.getElementById("quizProgressBar").style.width = pct + "%";
  document.getElementById("quizCounter").textContent = `${_quizIndex + 1} / ${questions.length}`;
  document.getElementById("quizQuestion").textContent = q.question;

  const labels = ["A", "B", "C", "D"];
  document.getElementById("quizOptions").innerHTML = q.options.map((opt, i) => `
    <button class="quiz-option" data-index="${i}" onclick="selectOption(${i})" tabindex="0">
      <span class="quiz-option-label">${labels[i]}</span>
      <span class="quiz-option-text">${escapeHtml(opt)}</span>
    </button>`).join("");

  // Add keyboard navigation for quiz options
  const options = document.querySelectorAll(".quiz-option");
  options.forEach((opt, i) => {
    opt.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectOption(i);
      } else if (e.key === "ArrowDown" && i < options.length - 1) {
        e.preventDefault();
        options[i + 1].focus();
      } else if (e.key === "ArrowUp" && i > 0) {
        e.preventDefault();
        options[i - 1].focus();
      }
    });
  });
  
  // Focus first option
  setTimeout(() => options[0]?.focus(), 50);

  document.getElementById("quizFeedback").style.display = "none";
  document.getElementById("quizFeedbackInner").innerHTML = "";
}

window.selectOption = function (selectedIdx) {
  if (_quizAnswered) return;
  _quizAnswered = true;

  const q = window._quizData[_quizIndex];
  const correctIdx = q.correct;
  const isCorrect = selectedIdx === correctIdx;
  const labels = ["A", "B", "C", "D"];

  if (isCorrect) _quizScore++;

  document.querySelectorAll(".quiz-option").forEach((btn, i) => {
    btn.disabled = true;
    if (i === correctIdx) btn.classList.add("correct");
    else if (i === selectedIdx) btn.classList.add("incorrect");
  });

  const fb = document.getElementById("quizFeedback");
  const fbInner = document.getElementById("quizFeedbackInner");
  const btnNext = document.getElementById("btnNext");
  fb.style.display = "block";

  if (isCorrect) {
    fbInner.innerHTML = `<div class="feedback-correct">✅ Correct!</div>`;
    btnNext.style.display = "none";
    announceToScreenReader("Correct answer!");
    setTimeout(() => { if (_quizAnswered) nextQuestion(); }, 1500);
  } else {
    fbInner.innerHTML = `
      <div class="feedback-incorrect">
        ❌ Incorrect. The correct answer is <strong>${labels[correctIdx]}. ${escapeHtml(q.answer)}</strong>
      </div>
      <div class="feedback-reasoning">💡 ${escapeHtml(q.reasoning)}</div>`;
    btnNext.style.display = "inline-block";
    announceToScreenReader(`Incorrect. Correct answer is ${labels[correctIdx]}. ${q.answer}`);
  }
};

window.nextQuestion = function () {
  _quizIndex++;
  if (_quizIndex >= window._quizData.length) showFinalScore();
  else renderQuestion();
};

function showFinalScore() {
  const total = window._quizData.length;
  const pct = Math.round((_quizScore / total) * 100);
  let grade = "🔴 Keep studying!";
  if (pct >= 90) grade = "🏆 Excellent!";
  else if (pct >= 70) grade = "✅ Good job!";
  else if (pct >= 50) grade = "📚 Not bad, review the summary.";

  document.getElementById("quizProgressBar").style.width = "100%";
  document.getElementById("quizCounter").textContent = `${total} / ${total}`;
  document.getElementById("quizQuestion").innerHTML = `
    <div class="quiz-final-score">
      <div class="quiz-score-number">${_quizScore} / ${total}</div>
      <div class="quiz-score-pct">${pct}%</div>
      <div class="quiz-score-grade">${grade}</div>
    </div>`;
  document.getElementById("quizOptions").innerHTML = `
    <button class="btn-generate-all" style="margin-top:1rem" onclick="startQuiz()">🔄 Retake Quiz</button>`;
  document.getElementById("quizFeedback").style.display = "none";
  
  announceToScreenReader(`Quiz complete. Score: ${_quizScore} out of ${total}, ${pct} percent. ${grade}`);
}

// ─── COPY/EXPORT HELPERS ─────────────────────────────────────────
window.copyText = (id) => copyText(id, showToast);

window.exportQuiz = function () {
  if (!window._quizData?.length) return;
  const json = JSON.stringify(window._quizData, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "quiz_export.json";
  a.click();
  URL.revokeObjectURL(url);
  showSuccess("Quiz exported as JSON");
};

window.exportSummary = function () {
  const summary = document.getElementById("summaryText")?.textContent;
  if (!summary) return showError("No summary to export");
  const blob = new Blob([summary], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "summary.txt";
  a.click();
  URL.revokeObjectURL(url);
  showSuccess("Summary exported");
};

window.exportAll = async function () {
  const summary = document.getElementById("summaryText")?.textContent || "";
  const quiz = window._quizData || [];
  const content = `SUMMARY\n=======\n${summary}\n\nQUIZ\n====\n${JSON.stringify(quiz, null, 2)}`;
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "notes_export.txt";
  a.click();
  URL.revokeObjectURL(url);
  showSuccess("All notes exported");
};

// ─── HELPERS ─────────────────────────────────────────────────────
function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&")
    .replace(/</g, "<")
    .replace(/>/g, ">")
    .replace(/"/g, """)
    .replace(/'/g, "&#039;");
}