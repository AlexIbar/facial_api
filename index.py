import uuid

from verificaImagen import verificaImagenI
from typing import Union
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from os import getcwd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    nuevoNombre = str(uuid.uuid4())+".jpg"
    with open(getcwd() +"/Dataset_val/"+ nuevoNombre, "wb") as myfile:
        content = await file.read()
        myfile.write(content)
        myfile.close()
    return verificaImagenI(nuevoNombre)
