from fastapi import FastAPI
from pydantic import BaseModel
from pii_detector import detect_pii
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or restrict if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Request(BaseModel):
    text: str

@app.post("/detect")
def detect(request: Request):
    result = detect_pii(request.text)
    return {"pii": result}
