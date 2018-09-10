__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt
from figures.base import Base
from figures.settings import *

from binding.analysis.sweep import Grid


class PhaseDiagram(Base):
    """
    Object for plotting phase diagram of equilibrium binding site occupancies across a two-dimensional concentration grid.

    Attributes:
    element (binding.model.elements.Element) - binding element

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    def __init__(self, element):
        """
        Instantiate object for plotting binding site occupancy phase diagram.

        Args:
        element (binding.model.elements.Element) - binding element
        """

        # store element
        self.element = element

        # initialize figure
        self.fig = None

    def render(self,
                cmin=(0, 0),
                cmax=(100, 100),
                Nc=(25, 25),
                figsize=(2, 2),
                **kwargs):
        """
        Render equilibrium binding site occupancy phase diagram.

        Args:
        cmin, cmax (tuple) - concentration bounds for phase diagram
        Nc (tuple) - sampling density for each concentration axis
        figsize (tuple) - figure size
        kwargs: keyword arguments for binding model
        """

        # create figure
        self.fig = self.create_figure(figsize=figsize)

        # evaluate and plot phase diagram
        self.plot(cmin=cmin,
                  cmax=cmax,
                  Nc=Nc,
                  **kwargs)

    def plot(self,
            cmin=(0, 0),
            cmax=(100, 100),
            Nc=(25, 25),
            **kwargs):

        """
        Plot expression dynamics for each cell type.

        Args:
        cmin, cmax (tuple) - concentration bounds for phase diagram
        Nc (tuple) - sampling density for each concentration axis
        kwargs: keyword aguments for binding model
        """

        # construct concentration grid and evaluate binding site occupancies
        grid = Grid(cmin=cmin, cmax=cmax, Nc=Nc)
        occupancies = grid.get_occupancies(self.element, **kwargs)

        # plot phase diagram
        occupancies.plot_overall_occupancy('Pnt',
                                           cmap=plt.cm.PiYG,
                                           mask=True,
                                           ax=self.fig.axes[0])

    @staticmethod
    def _format_ax(ax):
        """
        Format axis.

        Args:
        ax (matplotlib.axes.AxesSubplot)
        """
        pass


class TitrationContours(PhaseDiagram):
    """
    Object for visualizing a titration contour. Figure shows equilibrium Pnt binding site occupancies for each binding site position as a function of Pnt protein concentration. Yan levels are held constant at a specified value.

    Attributes:
    element (binding.model.elements.Element) - binding element

    Inherited attributes:
    fig (matplotlib.figure.Figure)
    """

    def render(self,
                yan_level=50,
                cmin=0,
                cmax=100,
                Nc=25,
                figsize=(3, 2),
                **kwargs):
        """
        Render titration contours. Contours reflect equilibrium Pnt bidning site occupancy at each binding site position. Yan concentration is fixed at a specified value.

        Args:
        yan_level (float) - fixed yan concentration
        cmin, cmax (float) - concentration range for titration
        Nc (int) - number of samples
        kwargs: keyword aguments for binding model
        """

        # create figure
        self.fig = self.create_figure(figsize=figsize)

        # evaluate and plot phase diagram
        self.plot(yan_level=yan_level,
                  cmin=cmin,
                  cmax=cmax,
                  Nc=Nc,
                  **kwargs)

    def plot(self,
                yan_level=50,
                cmin=0,
                cmax=100,
                Nc=25,
                **kwargs):

        """
        Plot titration contours. Contours reflect equilibrium Pnt bidning site occupancy at each binding site position. Yan concentration is fixed at a specified value.

        Args:
        yan_level (float) - fixed yan concentration
        cmin, cmax (float) - concentration range for titration
        Nc (int) - number of samples
        kwargs: keyword aguments for binding model
        """

        # construct concentration grid and evaluate binding site occupancies
        grid = Grid(cmin=(cmin, yan_level), cmax=(cmax, yan_level), Nc=(Nc, 1))
        occupancies = grid.get_occupancies(self.element, **kwargs)

        # plot titration contours
        _ = occupancies.plot_contours(fixed=0, fig=self.fig)

