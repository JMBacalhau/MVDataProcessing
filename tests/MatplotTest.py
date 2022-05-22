# -*- coding: utf-8 -*-
"""
Created on Sat May 21 22:52:14 2022

@author: Bacalhau
"""

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
fig = plt.Figure()
ax = fig.add_subplot(111)
ax.plot(range(5))
canvas = FigureCanvas(fig)
canvas.print_figure('sample.png')