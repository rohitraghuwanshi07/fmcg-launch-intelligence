from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..core.security import create_token, hash_password, verify_password
from ..db.database import get_db
from ..db.models import User
from ..deps import auth_rate_limit, current_user
from ..schemas import LoginIn, RegisterIn, TokenOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user(u: User) -> dict:
    return dict(id=u.id, email=u.email, name=u.name)


@router.post("/register", response_model=TokenOut, status_code=201, dependencies=[Depends(auth_rate_limit)])
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.email == body.email)):
        raise HTTPException(409, "An account with this email already exists.")
    u = User(email=body.email, name=body.name.strip(), password_hash=hash_password(body.password))
    db.add(u); db.commit()
    return TokenOut(access_token=create_token(u.id), user=_user(u))


@router.post("/login", response_model=TokenOut, dependencies=[Depends(auth_rate_limit)])
def login(body: LoginIn, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Incorrect email or password.")           # same message for both cases: no user enumeration
    return TokenOut(access_token=create_token(u.id), user=_user(u))


@router.get("/me")
def me(user: User = Depends(current_user)):
    return _user(user)
