/**
 * shared/toast.js — Toast notification system
 */

// Ensure toast container exists
function getToastContainer() {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }
  return container;
}

/**
 * Show a toast notification
 * @param {string} msg - Message to display
 * @param {string} type - Type: "success", "error", "info", "warning"
 * @param {number} duration - Duration in ms (default 4000)
 */
export function showToast(msg, type = "info", duration = 4000) {
  const container = getToastContainer();
  
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.setAttribute("role", "alert");
  toast.setAttribute("aria-live", "polite");
  
  toast.innerHTML = `
    <div class="toast-content">${msg}</div>
    <button class="toast-close" aria-label="Dismiss">&times;</button>
  `;
  
  container.appendChild(toast);
  
  // Close button
  toast.querySelector(".toast-close").addEventListener("click", () => {
    toast.remove();
  });
  
  // Auto-remove
  setTimeout(() => {
    if (toast.parentElement) toast.remove();
  }, duration);
}

/**
 * Show success toast
 * @param {string} msg - Message
 * @param {number} duration - Duration in ms
 */
export function showSuccess(msg, duration = 4000) {
  showToast(msg, "success", duration);
}

/**
 * Show error toast
 * @param {string} msg - Message
 * @param {number} duration - Duration in ms
 */
export function showError(msg, duration = 5000) {
  showToast(msg, "error", duration);
}

/**
 * Show info toast
 * @param {string} msg - Message
 * @param {number} duration - Duration in ms
 */
export function showInfo(msg, duration = 4000) {
  showToast(msg, "info", duration);
}

/**
 * Show warning toast
 * @param {string} msg - Message
 * @param {number} duration - Duration in ms
 */
export function showWarning(msg, duration = 4000) {
  showToast(msg, "warning", duration);
}