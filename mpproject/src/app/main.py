from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True))


@app.get("/")
async def root():
    return {"message": "Hello World"}
