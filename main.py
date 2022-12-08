from typing import Optional, List

from fastapi import FastAPI, UploadFile
import numpy as np
from Sudoku import ImageRegconition, Solver
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

class SudokuBoard(BaseModel):
    board: List[List[int]]
    solution_limit: int = 1

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

imgReg = ImageRegconition(model_path='./models/digits_model.h5')
solver = Solver()



@app.post('/uploadfiles/')
async def upload_files(file: Optional[UploadFile] = None):
    if not file:
        return {"success": False, "message": "No upload file sent"}
    elif 'image' not in file.content_type:
        return {"success": False, "message": "Your uploaded file is not an image (.png, .jpg, .jpeg, .gif)"}
    else:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        #print(nparr)
        try:
            result = imgReg.detect_board(nparr)
            return {"success": True, "content": result.tolist()}
        except Exception as e:
            return {"success": False, "message": str(e)}
            

@app.post('/solve/')
async def solve_board(input: SudokuBoard = None):
    sudoku_board = np.array(input.board)
    if sudoku_board is None:
        return {"success": False, "message": "Sudoku board is null"}
    else:
        solver.solution_limit = input.solution_limit
        results = []
        try:
            for solution in solver.solve(sudoku_board):
                results.append(solution.tolist())

            return {"success": True, "content": results}    
        except Exception as e:
            return {"success": False, "message": str(e)}

@app.get('/')
def main():
    with open('index.html', 'r') as f:
        return HTMLResponse(content=f.read())