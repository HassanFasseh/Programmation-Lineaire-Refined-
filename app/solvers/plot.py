# solvers/plot.py
import matplotlib.pyplot as plt
import io
import base64

class PlotRenderer:
    def __init__(self, figsize=(10, 8), style='ggplot'):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        plt.style.use(style)

    def draw_constraints(self, A, b, x_vals):
        for i in range(len(A)):
            if A[i, 1] != 0:
                y = (b[i] - A[i, 0] * x_vals) / A[i, 1]
                self.ax.plot(x_vals, y, linewidth=2.5)
            else:
                x = b[i] / A[i, 0]
                self.ax.axvline(x=x, linewidth=2.5)

    def draw_vertices(self, vertices, opt_point=None):
        if vertices:
            self.ax.scatter(*zip(*vertices), color='black', s=60)
        if opt_point is not None:
            self.ax.scatter(opt_point[0], opt_point[1], color='red', s=120)

    def save_plot(self, dpi=150):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode()
