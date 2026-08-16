/**
 * my-notes.js — Notes Library Page
 * Shows all sessions for the logged-in user with search + filter + bulk actions.
 * Enhanced with: grid/list toggle, bulk actions, storage indicators, keyboard nav
 */

import { auth, db } from "./firebase-config.js";
import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";
import { collection, getDocs, deleteDoc, doc, query, where, orderBy } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js";

import {
  initAuthGuard, showUserInfo, createSignOutHandler
} from "./shared/auth-guard.js";
import { showToast, showSuccess, showError } from "./shared/toast.js";
import { formatBytes, formatShortDate, escapeHtml } from "./shared/utils.js";

let allSessions = [];
let activeFilter = "all";
let currentView = "grid"; // grid or list
let selectedSessions = new Set();
let bulkActionsBar = null;

// ─── AUTH GUARD ───────────────────────────────────────────────────
initAuthGuard(auth, async (user) => {
  showUserInfo(user);
  await loadNotes();
  initTheme();
  initKeyboardShortcuts();
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

// ─── KEYBOARD SHORTCUTS ──────────────────────────────────────────
function initKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, [contenteditable]")) return;
    
    if (e.key === "/" || e.key === "?") {
      e.preventDefault();
      document.getElementById("searchInput")?.focus();
    }
    if (e.key === "Escape") {
      clearSelection();
      document.activeElement?.blur();
    }
    if (e.key === "g") {
      e.preventDefault();
      toggleView();
    }
    if (e.key === "a" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      selectAll();
    }
  });
}

// ─── LOAD NOTES ───────────────────────────────────────────────────
async function loadNotes() {
  const user = auth.currentUser;
  try {
    const q = query(
      collection(db, "sessions"),
      where("uid", "==", user.uid),
      orderBy("createdAt", "desc")
    );
    const snap = await getDocs(q);
    snap.forEach(d => allSessions.push({ id: d.id, ...d.data() }));

    document.getElementById("notesLoading").style.display = "none";
    createBulkActionsBar();
    createViewToggle();

    if (allSessions.length === 0) {
      document.getElementById("notesEmpty").style.display = "block";
    } else {
      document.getElementById("notesGrid").style.display = "grid";
      renderGrid(allSessions);
    }
  } catch (err) {
    document.getElementById("notesLoading").innerHTML =
      `<p style="color:var(--red-accent)">Failed to load notes: ${err.message}</p>`;
  }
}

function createViewToggle() {
  const filterPills = document.querySelector(".notes-filter-pills");
  if (!filterPills || document.getElementById("viewToggle")) return;
  
  const toggle = document.createElement("div");
  toggle.id = "viewToggle";
  toggle.className = "view-toggle";
  toggle.innerHTML = `
    <button class="view-btn active" data-view="grid" onclick="setView('grid')" aria-label="Grid view" title="Grid view (G)">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
    </button>
    <button class="view-btn" data-view="list" onclick="setView('list')" aria-label="List view" title="List view (G)">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
    </button>
  `;
  filterPills.parentNode.insertBefore(toggle, filterPills.nextSibling);
}

window.setView = function(view) {
  currentView = view;
  document.querySelectorAll(".view-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  const grid = document.getElementById("notesGrid");
  grid.classList.toggle("list-view", view === "list");
};

function createBulkActionsBar() {
  bulkActionsBar = document.createElement("div");
  bulkActionsBar.className = "bulk-actions";
  bulkActionsBar.innerHTML = `
    <span class="bulk-count"><span id="selectedCount">0</span> selected</span>
    <button class="bulk-btn" onclick="bulkDelete()">🗑 Delete</button>
    <button class="bulk-btn secondary" onclick="clearSelection()">✕ Clear</button>
  `;
  document.body.appendChild(bulkActionsBar);
}

function renderGrid(sessions) {
  const grid = document.getElementById("notesGrid");
  if (sessions.length === 0) {
    grid.innerHTML = `<p style="color:var(--text-muted);grid-column:1/-1;text-align:center;padding:2rem">
      No notes match your search.</p>`;
    return;
  }
  
  grid.innerHTML = sessions.map(s => {
    const date = formatShortDate(s.createdAt);
    const size = s.fileSize ? formatBytes(s.fileSize) : "";
    const isSelected = selectedSessions.has(s.id);
    const storageType = getStorageType(s);
    return `
      <div class="history-card-wrapper${isSelected ? " selected" : ""}" data-id="${s.id}">
        <input type="checkbox" class="select-checkbox" ${isSelected ? "checked" : ""} onchange="toggleSelection('${s.id}', this)" aria-label="Select ${escapeHtml(s.fileName || "Untitled")}">
        <a class="history-card${isSelected ? " selected" : ""}" href="notes.html?id=${s.id}">
          <div class="history-card-icon">📄</div>
          <div class="history-card-info">
            <div class="history-card-name">${escapeHtml(s.fileName || "Untitled")}</div>
            <div class="history-card-date">${date}${size ? " · " + size : ""}</div>
            <div class="history-card-badges">
              ${s.summary ? '<span class="badge badge-green">Summary</span>' : ""}
              ${s.quiz ? '<span class="badge badge-blue">Quiz</span>' : ""}
              ${s.audioB64 ? '<span class="badge badge-amber">Audio</span>' : ""}
              ${s.videoUrl ? '<span class="badge badge-teal">Video</span>' : ""}
              ${s.summaryFallbackUsed ? '<span class="badge badge-amber" title="Used extractive fallback">Fallback</span>' : ""}
              ${storageType}
            </div>
          </div>
          <div class="history-card-arrow">→</div>
        </a>
        <button class="btn-delete-note" title="Delete note" onclick="deleteNote('${s.id}', this)" aria-label="Delete ${escapeHtml(s.fileName || "note")}">🗑</button>
      </div>`;
  }).join("");
  
  // Add keyboard navigation for cards
  setTimeout(() => {
    document.querySelectorAll(".history-card").forEach(card => {
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          window.location.href = card.href;
        } else if (e.key === " ") {
          e.preventDefault();
          const checkbox = card.closest(".history-card-wrapper")?.querySelector(".select-checkbox");
          if (checkbox) checkbox.click();
        }
      });
    });
  }, 0);
}

function getStorageType(session) {
  // Check if audio/video are base64 (stored in Firestore) or URLs (stored externally)
  const audioStored = session.audioB64 && session.audioB64.startsWith("data:");
  const videoStored = session.videoUrl && session.videoUrl.startsWith("data:");
  const hasExternal = session.audioB64 && !audioStored || session.videoUrl && !videoStored;
  
  if (hasExternal) {
    return '<span class="badge badge-blue" title="Media stored externally (link)">External</span>';
  }
  return '<span class="badge badge-green" title="All content stored in Firestore">Local</span>';
}

// ─── SELECTION ───────────────────────────────────────────────────
function toggleSelection(sessionId, checkbox) {
  if (checkbox.checked) {
    selectedSessions.add(sessionId);
    checkbox.closest(".history-card-wrapper").classList.add("selected");
    checkbox.closest(".history-card-wrapper")?.querySelector(".history-card")?.classList.add("selected");
  } else {
    selectedSessions.delete(sessionId);
    checkbox.closest(".history-card-wrapper").classList.remove("selected");
    checkbox.closest(".history-card-wrapper")?.querySelector(".history-card")?.classList.remove("selected");
  }
  updateBulkActions();
}

function selectAll() {
  const visibleCards = document.querySelectorAll("#notesGrid .history-card-wrapper");
  visibleCards.forEach(wrapper => {
    const checkbox = wrapper.querySelector(".select-checkbox");
    const sessionId = wrapper.dataset.id;
    if (checkbox && !checkbox.checked) {
      checkbox.checked = true;
      selectedSessions.add(sessionId);
      wrapper.classList.add("selected");
      wrapper.querySelector(".history-card")?.classList.add("selected");
    }
  });
  updateBulkActions();
}

function clearSelection() {
  selectedSessions.clear();
  document.querySelectorAll(".select-checkbox").forEach(cb => cb.checked = false);
  document.querySelectorAll(".history-card-wrapper").forEach(w => w.classList.remove("selected"));
  document.querySelectorAll(".history-card").forEach(c => c.classList.remove("selected"));
  updateBulkActions();
}

function updateBulkActions() {
  const count = selectedSessions.size;
  const countEl = document.getElementById("selectedCount");
  if (countEl) countEl.textContent = count;
  
  if (count > 0) {
    bulkActionsBar.classList.add("visible");
  } else {
    bulkActionsBar.classList.remove("visible");
  }
}

async function bulkDelete() {
  if (selectedSessions.size === 0) return;
  if (!confirm(`Delete ${selectedSessions.size} notes? This cannot be undone.`)) return;
  
  const toDelete = [...selectedSessions];
  let deleted = 0;
  
  for (const id of toDelete) {
    try {
      await deleteDoc(doc(db, "sessions", id));
      deleted++;
    } catch (err) {
      console.error("Failed to delete", id, err);
    }
  }
  
  allSessions = allSessions.filter(s => !selectedSessions.has(s.id));
  selectedSessions.clear();
  applyFilters();
  showSuccess(`Deleted ${deleted} notes`);
  updateBulkActions();
  
  if (allSessions.length === 0) {
    document.getElementById("notesGrid").style.display = "none";
    document.getElementById("notesEmpty").style.display = "block";
  }
}

// ─── SEARCH & FILTER ────────────────────────────────────────────
window.filterNotes = function () {
  applyFilters();
};

window.setFilter = function (filter) {
  activeFilter = filter;
  document.querySelectorAll(".filter-pill").forEach(p => {
    p.classList.toggle("active", p.dataset.filter === filter);
  });
  applyFilters();
};

function applyFilters() {
  const search = document.getElementById("searchInput").value.toLowerCase();
  let filtered = allSessions.filter(s => {
    const nameMatch = (s.fileName || "").toLowerCase().includes(search);
    if (!nameMatch) return false;
    if (activeFilter === "all") return true;
    if (activeFilter === "summary") return !!s.summary;
    if (activeFilter === "quiz") return !!s.quiz;
    if (activeFilter === "audio") return !!s.audioB64;
    if (activeFilter === "video") return !!s.videoUrl;
    return true;
  });
  renderGrid(filtered);
}

// ─── DELETE ──────────────────────────────────────────────────────
window.deleteNote = async function (sessionId, btn) {
  if (!confirm("Delete this note? This cannot be undone.")) return;
  btn.disabled = true;
  btn.textContent = "⏳";
  try {
    await deleteDoc(doc(db, "sessions", sessionId));
    allSessions = allSessions.filter(s => s.id !== sessionId);
    selectedSessions.delete(sessionId);
    applyFilters();
    if (allSessions.length === 0) {
      document.getElementById("notesGrid").style.display = "none";
      document.getElementById("notesEmpty").style.display = "block";
    }
  } catch (err) {
    showError("Failed to delete: " + err.message);
    btn.disabled = false;
    btn.textContent = "🗑";
  }
};

// ─── EMPTY STATE HANDLING ───────────────────────────────────────
function checkEmptyState() {
  const grid = document.getElementById("notesGrid");
  const empty = document.getElementById("notesEmpty");
  const hasVisible = grid.querySelector(".history-card-wrapper") !== null;
  
  if (hasVisible) {
    grid.style.display = "grid";
    empty.style.display = "none";
  } else {
    grid.style.display = "none";
    empty.style.display = "block";
  }
}