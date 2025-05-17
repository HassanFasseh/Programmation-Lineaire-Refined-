# solvers/graphical.py
import numpy as np
from fastapi import HTTPException
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import io, base64
from app.solvers.base import OptimizationSolver
from app.utils import sort_vertices_counterclockwise

class GraphicalSolver(OptimizationSolver):
    def solve(self):
        if len(self.c) != 2:
            raise HTTPException(status_code=400, detail="Graphical method requires exactly 2 variables")

        A, b, c, bounds = self.A, self.b, self.c, self.bounds
        
        # Create figure with better settings
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.set_aspect('equal', adjustable='box')
        
        # Determine appropriate bounds for the plot
        x_min, x_max, y_min, y_max = self._calculate_plot_bounds(A, b, bounds)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add grid and labels
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.set_title('Linear Programming Solution', fontsize=14)
        
        # Plot constraints and feasible region
        self._plot_constraints(ax, A, b, x_min, x_max)
        feasible_vertices = self._find_feasible_vertices(A, b, bounds)
        
        # Plot feasible region if it exists
        if feasible_vertices:
            feasible_polygon = Polygon(feasible_vertices, closed=True, 
                                     fill=True, color='lightblue', alpha=0.5)
            ax.add_patch(feasible_polygon)
            
            # Find and plot optimal solution
            result = self._find_optimal_solution(feasible_vertices, c)
            self._plot_optimal_point(ax, result['optimal_point'])
            
            # Plot all vertices
            self._plot_vertices(ax, feasible_vertices, result['optimal_point'])
        else:
            result = {
                "status": "infeasible",
                "plot": None,
                "optimal_point": {"x": None, "y": None},
                "optimal_value": None,
                "vertices": []
            }
        
        # Save plot to base64
        result["plot"] = self._save_plot_to_base64(fig)
        plt.close(fig)
        return result

    def _calculate_plot_bounds(self, A, b, bounds):
        """Calculate appropriate plot boundaries based on constraints and bounds"""
        x_min, x_max = 0, 10
        y_min, y_max = 0, 10
        
        # Adjust based on constraints
        for i in range(len(A)):
            if A[i, 1] != 0:  # Non-vertical line
                x_intercept = b[i] / A[i, 0] if A[i, 0] != 0 else x_max
                y_intercept = b[i] / A[i, 1] if A[i, 1] != 0 else y_max
                x_max = max(x_max, x_intercept * 1.2)
                y_max = max(y_max, y_intercept * 1.2)
            else:  # Vertical line
                x_intercept = b[i] / A[i, 0]
                x_max = max(x_max, x_intercept * 1.2)
        
        # Adjust based on variable bounds
        for i in range(2):
            if bounds[i][0] is not None:
                x_min = bounds[i][0] if i == 0 else x_min
                y_min = bounds[i][0] if i == 1 else y_min
            if bounds[i][1] is not None:
                x_max = bounds[i][1] if i == 0 else x_max
                y_max = bounds[i][1] if i == 1 else y_max
        
        # Ensure minimum range
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < 5:
            x_max = x_min + 5
        if y_range < 5:
            y_max = y_min + 5
            
        return x_min, x_max, y_min, y_max

    def _plot_constraints(self, ax, A, b, x_min, x_max):
        """Plot all constraints with labels"""
        x_vals = np.linspace(x_min, x_max, 500)
        
        for i in range(len(A)):
            if A[i, 1] != 0:  # Non-vertical line
                y = (b[i] - A[i, 0] * x_vals) / A[i, 1]
                line, = ax.plot(x_vals, y, linewidth=2.5, 
                               label=f'{A[i,0]:.2f}x1 + {A[i,1]:.2f}x2 ≤ {b[i]:.2f}')
                # Add constraint label
                label_x = x_vals[len(x_vals)//3]
                label_y = (b[i] - A[i, 0] * label_x) / A[i, 1]
                ax.text(label_x, label_y, f'C{i+1}', color=line.get_color(),
                       fontsize=10, ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            else:  # Vertical line
                x = b[i] / A[i, 0]
                line = ax.axvline(x=x, linewidth=2.5, 
                                 label=f'{A[i,0]:.2f}x1 ≤ {b[i]:.2f}')
                # Add constraint label
                label_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.7 + ax.get_ylim()[0]
                ax.text(x, label_y, f'C{i+1}', color=line.get_color(),
                       fontsize=10, ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add legend
        ax.legend(loc='upper right', fontsize=9)

    def _find_feasible_vertices(self, A, b, bounds):
        """Find all feasible vertices of the feasible region"""
        vertices = []
        
        # Check origin if feasible
        origin = np.array([0, 0])
        if self._is_feasible(origin):
            vertices.append(origin)
        
        # Check axis intercepts
        for i in range(2):
            for j in range(len(A)):
                if A[j, i] != 0:
                    point = np.zeros(2)
                    point[i] = b[j] / A[j, i]
                    if self._is_feasible(point):
                        vertices.append(point)
        
        # Check intersections of constraints
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                try:
                    sol = np.linalg.solve(A[[i,j]], b[[i,j]])
                    if self._is_feasible(sol):
                        vertices.append(sol)
                except np.linalg.LinAlgError:
                    continue
        
        # Filter unique vertices and sort them counterclockwise
        unique_vertices = self._filter_unique(vertices)
        return unique_vertices

    def _find_optimal_solution(self, vertices, c):
        """Find the optimal solution from the feasible vertices"""
        obj_vals = [np.dot(c, v) for v in vertices]
        opt_idx = np.argmax(obj_vals) if self.maximize else np.argmin(obj_vals)
        opt_point = vertices[opt_idx]
        opt_value = obj_vals[opt_idx]
        
        return {
            "status": "optimal",
            "optimal_point": {"x": float(opt_point[0]), "y": float(opt_point[1])},
            "optimal_value": float(opt_value),
            "vertices": [
                {
                    "x": float(v[0]),
                    "y": float(v[1]),
                    "objective_value": float(np.dot(c, v)),
                    "is_optimal": np.allclose(v, opt_point)
                } for v in vertices
            ]
        }

    def _plot_vertices(self, ax, vertices, opt_point=None):
        """Plot all vertices with optimal point highlighted"""
        for v in vertices:
            ax.scatter(v[0], v[1], color='blue', s=80, zorder=5)
            ax.text(v[0], v[1], f'({v[0]:.2f}, {v[1]:.2f})', 
                   fontsize=9, ha='right', va='bottom')
        
        if opt_point:
            opt_array = np.array([opt_point['x'], opt_point['y']])
            ax.scatter(opt_array[0], opt_array[1], color='red', s=120, 
                      marker='*', zorder=6, label='Optimal Solution')
            ax.text(opt_array[0], opt_array[1], f'Optimal\n({opt_array[0]:.2f}, {opt_array[1]:.2f})', 
                   fontsize=10, ha='left', va='bottom', color='red')

    def _plot_optimal_point(self, ax, opt_point):
        """Plot the optimal solution point"""
        opt_array = np.array([opt_point['x'], opt_point['y']])
        ax.scatter(opt_array[0], opt_array[1], color='red', s=120, 
                  marker='*', zorder=6, label='Optimal Solution')
        ax.text(opt_array[0], opt_array[1], f'Optimal\n({opt_array[0]:.2f}, {opt_array[1]:.2f})', 
               fontsize=10, ha='left', va='bottom', color='red')

    def _is_feasible(self, point):
        """Check if a point satisfies all constraints and bounds"""
        return (all(np.dot(self.A, point) <= self.b + 1e-10) and 
                all((self.bounds[k][0] is None or point[k] >= self.bounds[k][0] - 1e-10) and 
                    (self.bounds[k][1] is None or point[k] <= self.bounds[k][1] + 1e-10)
                    for k in range(2)))

    def _filter_unique(self, vertices):
        """Filter out duplicate vertices"""
        unique = []
        for v in vertices:
            if not any(np.allclose(v, u, atol=1e-8) for u in unique):
                unique.append(v)
        return sort_vertices_counterclockwise(unique)

    def _save_plot_to_base64(self, fig):
        """Save matplotlib figure to base64 encoded string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')