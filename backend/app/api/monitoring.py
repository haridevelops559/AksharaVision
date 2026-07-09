from fastapi import APIRouter

from backend.app.services.monitoring import build_monitoring_report

router = APIRouter(tags=["monitoring"])


@router.get("/monitoring")
def get_monitoring():
    return build_monitoring_report()