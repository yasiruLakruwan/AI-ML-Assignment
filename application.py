from pathlib import Path
import json
import shutil
import subprocess
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse


app = FastAPI(
    title="Elderly Activity AI Backend",
    description="Backend for elderly activity and fall detection",
    version="1.0"
)


# --------------------------------------------------
# Folders
# --------------------------------------------------

UPLOAD_FOLDER = Path("data/uploads")
OUTPUT_FOLDER = Path("data/output")

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Elderly Activity AI Backend",
        "docs": "/docs"
    }


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "running"
    }


# --------------------------------------------------
# Analyze video
# --------------------------------------------------

@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...)
):

    # Check file
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv"
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Only video files are allowed"
        )

    # --------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------

    input_path = (
        UPLOAD_FOLDER
        /
        f"input{extension}"
    )

    with input_path.open(
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # --------------------------------------------------
    # Run your existing AI pipeline
    # --------------------------------------------------

    command = [

        sys.executable,

        "src/main.py",

        "--video",

        str(input_path),

        "--output-dir",

        str(OUTPUT_FOLDER),

        "--sample-fps",

        "3"
    ]

    try:

        result = subprocess.run(

            command,

            capture_output=True,

            text=True,

            check=True
        )

    except subprocess.CalledProcessError as error:

        print(
            "AI PIPELINE ERROR:"
        )

        print(
            error.stdout
        )

        print(
            error.stderr
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "AI analysis failed. "
                "Check the backend terminal."
            )
        )

    # --------------------------------------------------
    # Read JSON result
    # --------------------------------------------------

    result_file = (
        OUTPUT_FOLDER
        /
        "timeline.json"
    )

    if not result_file.exists():

        raise HTTPException(

            status_code=500,

            detail=(
                "timeline.json was not generated"
            )
        )

    with result_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        result_data = json.load(
            file
        )

    return {

        "message":
            "Video analyzed successfully",

        "filename":
            file.filename,

        "timeline":
            result_data.get(
                "timeline",
                []
            ),

        "events":
            result_data.get(
                "events",
                []
            ),

        "video_url":
            "/result-video"
    }


# --------------------------------------------------
# Get annotated video
# --------------------------------------------------

@app.get("/result-video")
def result_video():

    video_path = (
        OUTPUT_FOLDER
        /
        "annotated.mp4"
    )

    if not video_path.exists():

        raise HTTPException(

            status_code=404,

            detail="Annotated video not found"
        )

    return FileResponse(

        video_path,

        media_type="video/mp4",

        filename="annotated.mp4"
    )