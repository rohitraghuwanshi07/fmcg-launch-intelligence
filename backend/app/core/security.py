from __future__ import annotations
import datetime as dt
import bcrypt
import jwt
from .settings import SETTINGS

ALGO = "HS256"


def hash_password(p: str) -> str:
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(p: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(p.encode(), h.encode())
    except ValueError:
        return False


def create_token(user_id: int) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return jwt.encode({"sub": str(user_id), "iat": now, "exp": now + dt.timedelta(minutes=SETTINGS.token_minutes)}, SETTINGS.secret_key, algorithm=ALGO)


def decode_token(t: str) -> int | None:
    try:
        return int(jwt.decode(t, SETTINGS.secret_key, algorithms=[ALGO])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
