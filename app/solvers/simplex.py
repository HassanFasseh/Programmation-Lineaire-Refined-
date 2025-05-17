# solvers/simplex.py
import numpy as np
from app.solvers.base import OptimizationSolver

class TableauStep:
    def __init__(self, tableau, basic_vars, non_basic_vars, pivot_row=None, pivot_col=None):
        self.tableau = tableau.copy()
        self.basic_vars = basic_vars.copy()
        self.non_basic_vars = non_basic_vars.copy()
        self.pivot_row = pivot_row
        self.pivot_col = pivot_col

class SimplexSolver(OptimizationSolver):
    def solve(self):
        num_constraints = len(self.b)
        num_variables = len(self.c)

        A_std = np.hstack((self.A, np.eye(num_constraints)))
        c_std = np.hstack((self.c, np.zeros(num_constraints)))

        tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
        tableau[0, :-1] = -c_std if self.maximize else c_std
        tableau[1:, :-1] = A_std
        tableau[1:, -1] = self.b

        basic_vars = [num_variables + i for i in range(num_constraints)]
        non_basic_vars = list(range(num_variables))
        steps = [TableauStep(tableau, basic_vars, non_basic_vars)]

        max_iter = 100
        for _ in range(max_iter):
            pivot_col = self._choose_pivot_col(tableau)
            if pivot_col is None:
                break

            pivot_row = self._choose_pivot_row(tableau, pivot_col)
            if pivot_row is None:
                return {
                    "status": "unbounded",
                    "message": "The problem is unbounded",
                    "steps": [self._format_step(s) for s in steps]
                }

            steps.append(TableauStep(tableau, basic_vars, non_basic_vars, pivot_row, pivot_col))

            pivot_val = tableau[pivot_row, pivot_col]
            tableau[pivot_row] /= pivot_val
            for i in range(len(tableau)):
                if i != pivot_row:
                    tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

            basic_vars[pivot_row - 1], non_basic_vars[pivot_col] = (
                non_basic_vars[pivot_col], basic_vars[pivot_row - 1])

            steps.append(TableauStep(tableau, basic_vars, non_basic_vars))

        solution = np.zeros(num_variables + num_constraints)
        for i, var in enumerate(basic_vars):
            solution[var] = tableau[i + 1, -1]

        optimal_value = -tableau[0, -1] if self.maximize else tableau[0, -1]

        return {
            "status": "optimal",
            "solution": {
                "variables": solution[:num_variables].tolist(),
                "value": float(optimal_value)
            },
            "steps": [self._format_step(s) for s in steps]
        }

    def _choose_pivot_col(self, tableau):
        row = tableau[0, :-1]
        if self.maximize:
            idx = np.argmin(row)
            return idx if row[idx] < -1e-10 else None
        else:
            idx = np.argmax(row)
            return idx if row[idx] > 1e-10 else None

    def _choose_pivot_row(self, tableau, pivot_col):
        ratios = []
        for i in range(1, len(tableau)):
            if tableau[i, pivot_col] > 1e-10:
                ratios.append((i, tableau[i, -1] / tableau[i, pivot_col]))
        if not ratios:
            return None
        return min(ratios, key=lambda x: x[1])[0]

    def _format_step(self, step):
        tableau_data = step.tableau.tolist()
        basic_vars = step.basic_vars
        non_basic_vars = step.non_basic_vars
        num_rows = len(tableau_data)
        num_cols = len(tableau_data[0])

        basic_names = [f"x{v+1}" if v < num_cols - num_rows else f"s{v - (num_cols - num_rows) + 1}" for v in basic_vars]
        non_basic_names = [f"x{v+1}" if v < num_cols - num_rows else f"s{v - (num_cols - num_rows) + 1}" for v in non_basic_vars]

        formatted = [[""] + non_basic_names + ["RHS"]]
        formatted.append(["Z"] + [round(v, 4) for v in tableau_data[0]])
        for i in range(1, num_rows):
            formatted.append([basic_names[i - 1]] + [round(v, 4) for v in tableau_data[i]])

        return {
            "tableau": formatted,
            "basic_vars": basic_names,
            "non_basic_vars": non_basic_names,
            "pivot_row": step.pivot_row,
            "pivot_col": step.pivot_col
        }
