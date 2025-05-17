# Project structure refactor into OOP-style design

# Directory structure:
# /app
# ├── main.py                  <- FastAPI entry point
# ├── models.py                <- Pydantic request model
# ├── utils.py                 <- Helper functions
# ├── solvers/
# │   ├── base.py            <- Base OptimizationSolver class
# │   ├── graphical.py       <- GraphicalSolver class
# │   ├── simplex.py         <- SimplexSolver + TableauStep
# │   └── plot.py            <- PlotRenderer for matplotlib
# └── templates/index.html    <- Your existing HTML

# The refactored main.py would look like this:

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.models import OptimizationRequest
from app.solvers.graphical import GraphicalSolver
from app.solvers.simplex import SimplexSolver
from app.utils import convert_numpy_types
import numpy as np

app = FastAPI(title="Linear Programming Solver API")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/optimize")
async def optimize(request_data: OptimizationRequest):
    c = np.array(request_data.objective_coefficients)
    A = np.array(request_data.constraint_matrix)
    b = np.array(request_data.rhs_values)
    bounds = request_data.variable_bounds or [(0, None)] * len(c)

    if not request_data.maximize:
        c = -c

    if request_data.method.lower() == "graphique":
        if len(c) != 2:
            raise HTTPException(status_code=400, detail="Graphical method requires exactly 2 variables")
        solver = GraphicalSolver(c, A, b, bounds, request_data.maximize)
    elif request_data.method.lower() == "simplexe":
        solver = SimplexSolver(c, A, b, bounds, request_data.maximize)
    else:
        raise HTTPException(status_code=400, detail="Method must be either 'graphique' or 'simplexe'")

    result = solver.solve()
    return convert_numpy_types(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
