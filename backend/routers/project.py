"""Project router: export/import .rlt files, inspect contents."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.rlt_service import export_rlt, import_rlt, inspect_rlt

router = APIRouter()


class ExportRequest(BaseModel):
    project_dir: str
    output_path: str


class ImportRequest(BaseModel):
    rlt_path: str
    output_dir: str


class InspectRequest(BaseModel):
    rlt_path: str


@router.post("/export-rlt")
async def api_export_rlt(req: ExportRequest):
    """Pack a project folder into an .rlt file."""
    try:
        summary = export_rlt(req.project_dir, req.output_path)
        return {"status": "ok", "summary": summary}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import-rlt")
async def api_import_rlt(req: ImportRequest):
    """Unpack an .rlt file into a project directory."""
    try:
        manifest = import_rlt(req.rlt_path, req.output_dir)
        return {"status": "ok", "manifest": manifest}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inspect-rlt")
async def api_inspect_rlt(req: InspectRequest):
    """Read metadata from an .rlt file without extracting."""
    try:
        info = inspect_rlt(req.rlt_path)
        return info
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
