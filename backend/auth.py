"""
auth.py — Firebase Authentication
==================================
Firebase ID token verification for backend API protection.
"""

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger(__name__)

# Firebase Admin SDK initialization
_firebase_app = None
_auth = None


def init_firebase_admin():
    """Initialize Firebase Admin SDK."""
    global _firebase_app, _auth
    if _firebase_app is not None:
        return _auth

    try:
        import firebase_admin
        from firebase_admin import auth, credentials

        from .config import get_settings

        settings = get_settings()

        # Try to get credentials from environment or use default
        cred = None
        if settings.FIREBASE_SERVICE_ACCOUNT_JSON:
            import json
            cred_dict = json.loads(settings.FIREBASE_SERVICE_ACCOUNT_JSON)
            cred = credentials.Certificate(cred_dict)
        elif settings.GOOGLE_APPLICATION_CREDENTIALS:
            cred = credentials.ApplicationDefault()
        else:
            # Try default credentials (works on GCP, Cloud Run, etc.)
            cred = credentials.ApplicationDefault()

        _firebase_app = firebase_admin.initialize_app(cred)
        _auth = auth
        log.info("Firebase Admin SDK initialized successfully")
        return _auth
    except Exception as e:
        log.warning(f"Firebase Admin SDK initialization failed: {e}. Auth will be disabled.")
        return None


def get_firebase_auth():
    """Get Firebase Auth instance, initializing if needed."""
    global _auth
    if _auth is None:
        _auth = init_firebase_admin()
    return _auth


# HTTP Bearer token security scheme
security = HTTPBearer(auto_error=False)


class CurrentUser:
    """Current authenticated user from Firebase token."""

    def __init__(self, uid: str, email: str | None = None, name: str | None = None, picture: str | None = None):
        self.uid = uid
        self.email = email
        self.name = name
        self.picture = picture

    def __repr__(self):
        return f"CurrentUser(uid={self.uid}, email={self.email})"


async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> CurrentUser:
    """
    Verify Firebase ID token and return current user.
    
    Raises:
        HTTPException: If token is missing, invalid, or expired.
    """
    auth = get_firebase_auth()

    if auth is None:
        # Auth not configured - allow request but log warning
        log.warning("Firebase Auth not configured, allowing unauthenticated request")
        return CurrentUser(uid="anonymous", email=None)

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        name = decoded_token.get("name")
        picture = decoded_token.get("picture")

        if not uid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing uid",
            )

        return CurrentUser(uid=uid, email=email, name=name, picture=picture)

    except Exception as e:
        log.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> CurrentUser | None:
    """
    Get current user if token is valid, otherwise return None.
    Useful for endpoints that work both authenticated and unauthenticated.
    """
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_auth(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Dependency that requires authentication (raises 401 if not authenticated)."""
    if current_user.uid == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user
