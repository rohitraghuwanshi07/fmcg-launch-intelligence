from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session
from src import scenario_engine as se
from ..db.database import get_db
from ..db.models import Assessment, Job, Project, User
from ..deps import current_user, owned_project
from ..schemas import AskIn, AssessIn, JobIn, ProjectIn, ProjectUpdate, WhatIfIn
from ..services import jobs
from ..services.engine import ENGINE, input_hash
from ..services.serialize import to_jsonable

router = APIRouter(prefix="/api", tags=["projects"])


def _need_models():
    if not ENGINE.ready():
        raise HTTPException(503, "Models are not trained. Run `python scripts/validate_data.py && python scripts/train_models.py`.")


def _inp(p: Project):
    try:
        return ENGINE.parse(p.inputs)
    except ValueError as e:
        raise HTTPException(422, str(e))


def _summary(p: Project, last: Assessment | None) -> dict:
    return dict(id=p.id, name=p.name, category=p.inputs.get("category"), price=p.inputs.get("price"), updated_at=p.updated_at, created_at=p.created_at,
                last=None if not last else dict(decision=last.decision, p_hurdle=last.p_hurdle, readiness=last.readiness, at=last.created_at))


def _last(db: Session, pid: int):
    return db.scalar(select(Assessment).where(Assessment.project_id == pid).order_by(Assessment.id.desc()).limit(1))


@router.post("/validate")
def validate(inputs: dict, user: User = Depends(current_user)):
    return ENGINE_validate(inputs)


def ENGINE_validate(d: dict) -> dict:
    from src.data_validation import validate_launch_inputs
    return validate_launch_inputs(se.LaunchInputs.from_dict(d).to_dict())


@router.get("/projects")
def list_projects(user: User = Depends(current_user), db: Session = Depends(get_db)):
    ps = db.scalars(select(Project).where(Project.user_id == user.id).order_by(Project.updated_at.desc())).all()
    return to_jsonable([_summary(p, _last(db, p.id)) for p in ps])


@router.post("/projects", status_code=201)
def create_project(body: ProjectIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    inputs = se.LaunchInputs.from_dict(body.inputs).to_dict() if body.inputs else ENGINE.default_inputs()
    p = Project(user_id=user.id, name=body.name.strip(), inputs=to_jsonable(inputs)); db.add(p); db.commit()
    return to_jsonable(dict(_summary(p, None), inputs=p.inputs))


@router.get("/projects/{project_id}")
def get_project(p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    return to_jsonable(dict(_summary(p, _last(db, p.id)), inputs=p.inputs))


@router.put("/projects/{project_id}")
def update_project(body: ProjectUpdate, p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    if body.name: p.name = body.name.strip()
    if body.inputs is not None: p.inputs = to_jsonable(se.LaunchInputs.from_dict(body.inputs).to_dict())
    db.commit()
    return to_jsonable(dict(_summary(p, _last(db, p.id)), inputs=p.inputs))


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    db.delete(p); db.commit()
    return Response(status_code=204)


# ------------------------------------------------------------------ analysis
@router.post("/projects/{project_id}/assess")
def assess(body: AssessIn = AssessIn(), p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    _need_models(); inp = _inp(p)
    v = ENGINE_validate(inp.to_dict())
    if v["errors"]:
        raise HTTPException(422, "; ".join(v["errors"]))
    payload = ENGINE.analyze(inp, body.n_sims); payload["validation"] = v
    a = Assessment(project_id=p.id, input_hash=input_hash(p.inputs), decision=payload["decision"]["decision"], p_hurdle=payload["p_hurdle"],
                   readiness=payload["readiness"]["index"], payload=payload)
    db.add(a); db.commit()
    return dict(payload, assessment_id=a.id, created_at=a.created_at.isoformat(), stale=False)


@router.get("/projects/{project_id}/assessments")
def history(p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    rows = db.scalars(select(Assessment).where(Assessment.project_id == p.id).order_by(Assessment.id.desc()).limit(50)).all()
    return to_jsonable([dict(id=a.id, decision=a.decision, p_hurdle=a.p_hurdle, readiness=a.readiness, at=a.created_at, price=a.payload["inputs"]["price"]) for a in rows])


@router.get("/projects/{project_id}/assessments/{aid}")
def get_assessment(aid: int, p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    a = db.get(Assessment, aid)
    if a is None or a.project_id != p.id:
        raise HTTPException(404, "Assessment not found.")
    return dict(a.payload, assessment_id=a.id, created_at=a.created_at.isoformat(), stale=a.input_hash != input_hash(p.inputs))


@router.get("/projects/{project_id}/latest")
def latest(p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    a = _last(db, p.id)
    if a is None:
        raise HTTPException(404, "No assessment yet.")
    return dict(a.payload, assessment_id=a.id, created_at=a.created_at.isoformat(), stale=a.input_hash != input_hash(p.inputs))


@router.get("/projects/{project_id}/sensitivity")
def sensitivity(p: Project = Depends(owned_project)):
    _need_models(); return ENGINE.sensitivity(_inp(p))


@router.post("/projects/{project_id}/whatif")
def whatif(body: WhatIfIn, p: Project = Depends(owned_project)):
    _need_models()
    kw = {k: v for k, v in body.model_dump().items() if v is not None}
    return ENGINE.whatif(_inp(p), kw)


@router.get("/projects/{project_id}/analogs")
def analogs(k: int = 5, p: Project = Depends(owned_project)):
    _need_models(); return ENGINE.analogs(_inp(p), max(3, min(k, 10)))


@router.post("/projects/{project_id}/ask")
def ask(body: AskIn, p: Project = Depends(owned_project)):
    from src import copilot as cp
    _need_models(); facts = ENGINE.facts(_inp(p))
    a = cp.answer(facts, body.question)
    return to_jsonable(dict(answer=a, brief=cp.brief(facts), facts=facts))


@router.get("/projects/{project_id}/report.pdf")
def report(kind: str = "executive", p: Project = Depends(owned_project)):
    _need_models()
    pdf = ENGINE.report_pdf(_inp(p), "detailed" if kind == "detailed" else "executive")
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="launch-{p.id}-{kind}.pdf"'})


# ------------------------------------------------------------------ jobs
@router.post("/projects/{project_id}/jobs", status_code=202)
def create_job(body: JobIn, user: User = Depends(current_user), p: Project = Depends(owned_project)):
    _need_models(); inp = _inp(p)
    jid = jobs.submit(user.id, p.id, body.kind, inp.to_dict(), body.params)
    return dict(id=jid, status="queued", kind=body.kind)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    j = db.get(Job, job_id)
    if j is None or j.user_id != user.id:
        raise HTTPException(404, "Job not found.")
    return to_jsonable(dict(id=j.id, kind=j.kind, status=j.status, result=j.result, error=j.error, project_id=j.project_id,
                            created_at=j.created_at, started_at=j.started_at, finished_at=j.finished_at))


@router.get("/projects/{project_id}/jobs")
def project_jobs(kind: str | None = None, p: Project = Depends(owned_project), db: Session = Depends(get_db)):
    q = select(Job).where(Job.project_id == p.id).order_by(Job.created_at.desc()).limit(20)
    if kind: q = q.where(Job.kind == kind)
    return to_jsonable([dict(id=j.id, kind=j.kind, status=j.status, created_at=j.created_at, finished_at=j.finished_at) for j in db.scalars(q).all()])
