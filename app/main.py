from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile
from .agent import MLOpsCoPilot

app = FastAPI(title="MLOps Co-Pilot by Mohsen Mostafa")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

agent = MLOpsCoPilot()

@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("app/static/index.html").read_text()

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    config: str = Form(...),
    question: str = Form("How can I reduce the API latency to under 100ms?")
):
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        answer = agent.analyze(tmp_path, config, question)
        return JSONResponse({"answer": answer})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
