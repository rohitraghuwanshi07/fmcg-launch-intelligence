from __future__ import annotations
import collections
import time
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from .core.security import decode_token
from .core.settings import SETTINGS
from .db.database import get_db
from .db.models import Project, User

bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> User:
    uid = decode_token(creds.credentials) if creds else None
    user = db.get(User, uid) if uid else None
    if user is None:
        raise HTTPException(401, "Not authenticated.", headers={"WWW-Authenticate": "Bearer"})
    return user


def owned_project(project_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> Project:
    p = db.get(Project, project_id)
    if p is None or p.user_id != user.id:                 # 404, not 403: never reveal that another user's id exists
        raise HTTPException(404, "Project not found.")
    return p


_HITS: dict[str, collections.deque] = collections.defaultdict(collections.deque)


def auth_rate_limit(request: Request):
    """Sliding-window limiter on auth endpoints (brute-force protection). Per-process; use Redis when scaled out."""
    ip = request.client.host if request.client else "?"
    now = time.time(); q = _HITS[ip]
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= SETTINGS.auth_rate_limit:
        raise HTTPException(429, "Too many attempts. Try again in a minute.")
    q.append(now)
