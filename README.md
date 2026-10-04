Elder pose detection system - Execute instructions

- Download the zip file
- Create virtual environment (CMD: python -m venv venv)
- Activate virtual environment (CMD: venv/Scripts/activate)
- Install requires (pip install -e .)
- Create a .env file in project directry.
- Create a gemini api key and add it to the .env file and also add the gemini model
    (GEMINI_API_KEY="<ADD_API_KEY>")
    (GEMINI_MODEL = "<GEMINI_MODEL>")
- For running main file (CMD: python src/main.py)
- For running backend (CMD: uvicorn application:app --reload)
