/**
 * shared/auth-guard.js — Firebase authentication guard and user UI
 */

import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

/**
 * Initialize auth guard - redirects to login if not authenticated
 * @param {Object} auth - Firebase auth instance
 * @param {Function} onAuthReady - Callback when user is authenticated (receives user object)
 * @param {string} loginPage - Login page URL (default: "login.html")
 */
export function initAuthGuard(auth, onAuthReady, loginPage = "login.html") {
  onAuthStateChanged(auth, async (user) => {
    if (!user) {
      window.location.href = loginPage;
      return;
    }
    if (onAuthReady) {
      await onAuthReady(user);
    }
  });
}

/**
 * Show user info in the user pill
 * @param {Object} user - Firebase user object
 * @param {Object} elements - DOM elements { pill, avatar, name, signOutBtn }
 */
export function showUserInfo(user, elements = {}) {
  const {
    pill = document.getElementById("userPill"),
    avatar = document.getElementById("userAvatar"),
    name = document.getElementById("userName"),
    signOutBtn = document.getElementById("signOutBtn")
  } = elements;

  if (!pill) return;

  if (user.photoURL) {
    avatar.src = user.photoURL;
    avatar.style.display = "";
  } else {
    avatar.style.display = "none";
  }
  
  name.textContent = user.displayName || user.email.split("@")[0];
  pill.style.display = "flex";
  
  if (signOutBtn) {
    signOutBtn.style.display = "inline-block";
  }
}

/**
 * Handle sign out
 * @param {Object} auth - Firebase auth instance
 * @param {string} loginPage - Login page URL (default: "login.html")
 */
export async function handleSignOut(auth, loginPage = "login.html") {
  await signOut(auth);
  window.location.href = loginPage;
}

/**
 * Create sign out handler for window global
 * @param {Object} auth - Firebase auth instance
 * @param {string} loginPage - Login page URL
 */
export function createSignOutHandler(auth, loginPage = "login.html") {
  return async () => {
    await signOut(auth);
    window.location.href = loginPage;
  };
}