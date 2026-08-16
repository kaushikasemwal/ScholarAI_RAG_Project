/**
 * shared/utils.js — Common utility functions
 */

/**
 * Format bytes to human readable string
 * @param {number} b - Bytes
 * @returns {string} Formatted string (B, KB, MB)
 */
export function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

/**
 * Escape HTML to prevent XSS
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
export function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&")
    .replace(/</g, "<")
    .replace(/>/g, ">")
    .replace(/"/g, """)
    .replace(/'/g, "&#039;");
}

/**
 * Capitalize first letter of string
 * @param {string} s - String to capitalize
 * @returns {string} Capitalized string
 */
export function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Delay helper for async operations
 * @param {number} ms - Milliseconds to delay
 * @returns {Promise<void>}
 */
export function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/**
 * Copy text to clipboard and show toast
 * @param {string} id - Element ID to copy text from
 * @param {Function} showToast - Toast function to show feedback
 */
export function copyText(id, showToast) {
  const el = document.getElementById(id);
  if (!el) return;
  navigator.clipboard.writeText(el.innerText).then(() => showToast("Copied!", "success"));
}

/**
 * Format Firestore timestamp to readable date
 * @param {Object} timestamp - Firestore Timestamp object
 * @returns {string} Formatted date string
 */
export function formatDate(timestamp) {
  if (!timestamp?.toDate) return "";
  return timestamp.toDate().toLocaleDateString("en-US", {
    weekday: "short", month: "short", day: "numeric", year: "numeric"
  });
}

/**
 * Get short date format (month day, year)
 * @param {Object} timestamp - Firestore Timestamp object
 * @returns {string} Short date string
 */
export function formatShortDate(timestamp) {
  if (!timestamp?.toDate) return "";
  return timestamp.toDate().toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric"
  });
}

/**
 * Format Firestore timestamp to ISO string for sorting
 * @param {Object} timestamp - Firestore Timestamp object
 * @returns {string} ISO date string
 */
export function formatISO(timestamp) {
  if (!timestamp?.toDate) return "";
  return timestamp.toDate().toISOString();
}

/**
 * Debounce function
 * @param {Function} fn - Function to debounce
 * @param {number} ms - Milliseconds
 * @returns {Function} Debounced function
 */
export function debounce(fn, ms) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), ms);
  };
}