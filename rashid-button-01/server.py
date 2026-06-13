import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

EMBED_BASE = os.getenv("EMBED_BASE", "http://localhost:8000/api/v1")
APP_KEY = os.getenv("APP_KEY", "7nLmv_Nb4OSqa3hKvJ5XahUwWv_eXb64FGOnnhVfRC8")


class EmbedRequest(BaseModel):
    external_user_id: str
    company_id: str
    subject: Literal["Maths", "English", "Science", "Vocational"]
    year_group: str
    topic: str
    support_profile: list[Literal["ADHD", "SEN"]] | None = None
    tier: Literal["free"] = "free"
    source: str = "school_app"
    voice_mode: bool = False


app = FastAPI()

# @app.post("/api/embed-url", tags={"demo"})
# async def _embed_url(
#     req: EmbedRequest,
#     x_app_key: str = Header(..., alias="x-app-key")
# ):
#     "Demo endpoint, just something to be called"
#     if x_app_key != APP_KEY:
#         raise HTTPException(401, "Invalid key")
#     return {"url": "http://tenticle.dev", "expires_in_seconds": 1000}


class UrlResponse(BaseModel):
    url: str


@app.post("/api/get-url")
async def make_url() -> UrlResponse:
    """
    Applicantions' way to get embed URL to app/website in a way
    that makes APP key invisible to end user.
    """
    payload = {
        "external_user_id": "student_39402",
        "company_id": "63995064-8f7a-440f-ba4e-f979de36ef27",
        "subject": "Maths",
        "year_group": "Year 7",
        "topic": "fractions-intro",
        "support_profile": ["ADHD"],
        "tier": "free",
        "source": "school_app",
        "voice_mode": False,
    }
    headers = {"x-app-key": APP_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{EMBED_BASE}/embed-url", json=payload, headers=headers
            )
            res.raise_for_status()
            url = res.json()["url"]
            # Adjust URL format if the embed service expects something different
            return UrlResponse(url=url)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")
