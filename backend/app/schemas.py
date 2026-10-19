from __future__ import annotations
import re
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterIn(BaseModel):
    email: str = Field(max_length=255)
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def _e(cls, v):
        v = v.strip().lower()
        if not _EMAIL.match(v): raise ValueError("Invalid email address.")
        return v


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    inputs: dict[str, Any] | None = None


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    inputs: dict[str, Any] | None = None


class AssessIn(BaseModel):
    n_sims: int = Field(2000, ge=200, le=5000)


class WhatIfIn(BaseModel):
    price_pct: float = Field(0, ge=-50, le=100)
    mfg_cost_pct: float = Field(0, ge=-50, le=100)
    marketing_pct: float = Field(0, ge=-100, le=300)
    outlets_mult: float = Field(1, ge=0.1, le=3)
    demand_pct: float = Field(0, ge=-80, le=200)
    discount: float | None = Field(default=None, ge=0, le=0.6)


class JobIn(BaseModel):
    kind: Literal["optimize", "red_team", "sobol", "pilot"]
    params: dict[str, Any] = {}


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=400)
