from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from tools.build_scenario_pipeline import build_scenario_pipeline
from tools.module_registry_loader import load_module_registry
from db.session import get_db, check_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    """Startup: verify DB connectivity and load registry."""
    check_db()
    logger.info("Database connection verified")
    yield


app = FastAPI(lifespan=lifespan)

registry = load_module_registry()


class BuildRequest(BaseModel):
    original_request: str
    plan: dict


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/build")
def build(request: BuildRequest, db: Session = Depends(get_db)):
    try:
        result = build_scenario_pipeline(
            plan=request.plan,
            original_request=request.original_request,
            registry=registry,
            db_session=db,
        )
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
