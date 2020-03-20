__author__ = 'Sebastian Bernasek'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial import distance_matrix

from .hill import HillModel
from binding.model.partitions import PartitionFunction


class Grid:
    """
    Class defines a 2D grid of pairwise transcriptipn factor concentrations.

    Attributes:
    cmin (int or tuple) - minimum concentration for each substrate, nM
    cmax (int or tuple) - maximum concentration for each substrate, nM
    Nc (int or tuple) - number of concentrations for each substrate
    names (dict) - {name: index} pairs for each substrate
    Cx, Cy (np.ndarray[float], length Nc) - unique concentrations for each substrate, nM
    concentrations (np.ndarray[float], shape 2 x (Nc^2) ) - unique substrate concentration pairs, nM
    """

    def __init__(self, cmin=0, cmax=100, Nc=25, names=None):

        """
        Args:
        cmin (int or tuple) - minimum concentration for each substrate, nM
        cmax (int or tuple) - maximum concentration for each substrate, nM
        Nc (int or tuple) - number of concentrations for each substrate
        names (dict) - {name: index} pairs for each substrate
        """

        # set names
        if names is None:
            names = dict(none=0, Yan=1, Pnt=2)
        self.names = names

        # define concentrations
        self.cmin = self.make_tuple(cmin)
        self.cmax = self.make_tuple(cmax)
        self.Nc = self.make_tuple(Nc)

        # set concentrations
        self.Cx = np.linspace(self.cmin[0], self.cmax[0], self.Nc[0]) * 1E-9
        self.Cy = np.linspace(self.cmin[1], self.cmax[1], self.Nc[1]) * 1E-9
        xx, yy = np.meshgrid(*(self.Cx, self.Cy), indexing='xy')

        self.concentrations = np.vstack((xx.ravel(), yy.ravel())).T

    @staticmethod
    def make_tuple(x):
        """ Convert standalone values to tuples. """
        if type(x) in (float, int, np.float64, np.int64):
            return (x, x)
        else:
            return x

    def run_binding_model(self, element, cut_depth=None):
        """
        Evaluate binding site occupancies for a particular binding element.

        Args:
        element (binding.model.elements.Element instance) - binding element
        cut_depth (int) - depth at which parallelization begins
        """
        return BindingModel(element, cut_depth,
                            cmin=self.cmin,
                            cmax=self.cmax,
                            Nc=self.Nc,
                            names=self.names)

    def run_simple_model(self):
        """
        Evaluate binding site occupancies using a simple analytical model for two competing transcription factors.
        """
        return SimpleModel(cmin=self.cmin,
                           cmax=self.cmax,
                           Nc=self.Nc,
                           names=self.names)


class SimpleModel(Grid):
    """
    Simple model describing the equilibrium surface coverage of sites bound by two competing transcription factor substrates. Analytical solution assumes no cooperativity exists within or between the two substrates.

    Attributes:
    theta_A, thetaB (np.ndarray[float], shape Nc x Nc) - surface coverage by substrates A and B
    theta (np.ndarray[float], shape Nc x Nc) - total surface coverage

    Inherited attributes:
    cmin (int or tuple) - minimum concentration for each substrate, nM
    cmax (int or tuple) - maximum concentration for each substrate, nM
    Nc (int or tuple) - number of concentrations for each substrate
    names (dict) - {name: index} pairs for each substrate
    Cx, Cy (np.ndarray[float], length Nc) - unique concentrations for each substrate, nM
    concentrations (np.ndarray[float], shape 2 x (Nc^2) ) - unique substrate concentration pairs, nM

    """
    def __init__(self, **kwargs):
        """
        Instantiate object for analytically evaluating equilibrium binding site occupancies for two competing transcription factors.

        Keyword arguments:
        cmin (int or tuple) - minimum concentration for each protein, nM
        cmax (int or tuple) - maximum concentration for each protein, nM
        Nc (int or tuple) - number of concentrations for each protein
        names (dict[str]=int) - name for each binding protein
        """

        Grid.__init__(self, **kwargs)
        A, B = np.meshgrid(*(self.Cx, self.Cy), indexing='xy')
        self.evaluate_occupancies(A*1e9, B*1e9)

    @staticmethod
    def solve(A0=1, B0=1, S0=1, KdA=1, KdB=1):
        """
        Analytically compute equilibrium surface coverages for a system of two substrates competing for common binding sites.

        Args:
        A0, B0 (float) - initial total concentrations of substrates A and B
        S0 (float) - initial total concentrations of binding sites
        KdA, KdB (float) - dissociation constants for substrates A and B

        Returns:
        theta_A, theta_B (float) - fractional equilibrium surface coverage by substrates A and B
        """

        # coefficients
        a = KdA + KdB + A0 + B0 - S0
        b = KdB*(A0 - S0) + KdA*(B0-S0)+KdA*KdB
        c = -KdA*KdB*S0

        # compute coverages
        theta = np.arccos((-2*(a**3)+9*a*b -27*c)/(2*np.sqrt(((a**2)-3*b)**3)))
        AS = A0 * ((2*np.sqrt((a**2)-3*b)*np.cos(theta/3))-a) / (3*KdA + (2*np.sqrt((a**2) - 3*b)*np.cos(theta/3)) - a)
        BS = B0 * ((2*np.sqrt((a**2) - 3*b)*np.cos(theta/3)) - a) / (3*KdB + (2*np.sqrt((a**2) - 3*b)*np.cos(theta/3)) - a)
        S = S0 - AS - BS
        return AS/S0, BS/S0

    def evaluate_occupancies(self, A, B):
        """
        Evaluate equilibrium surface coverage using analytical model.

        Args:
        A, B (np.ndarray[float] or float) - total substrate concentrations
        """
        theta_A, theta_B = self.solve(A, B)
        self.theta_A = theta_A
        self.theta_B = theta_B
        self.theta = theta_A + theta_B

    def plot_phase_diagram(self, species='pnt',  **kwargs):
        """
        Plot equilibrium surface coverage phase diagram.

        Args:
        species (str) - binding species
        kwargs: keyword arguments for visualization

        Returns:
        ax (matplotlib.axes.AxesSubplot)
        """

        # get fractional occupancies
        if species.lower() == 'total':
            coverage = self.theta
        elif species.lower() == 'pnt':
            coverage = self.theta_A
        elif species.lower() == 'yan':
            coverage = self.theta_B
        else:
            raise ValueError('Species not recognized.')

        # plot occupancies
        ax = self._plot_phase_diagram(coverage.T, self.theta, **kwargs)

        self._format_ax(ax)

        return ax

    @staticmethod
    def _plot_phase_diagram(substrate_coverage,
                            total_coverage=None,
                            cmap=plt.cm.PiYG,
                            vmin=0, vmax=1,
                            stretch=True,
                            bg_color=70,
                            ax=None):
        """
        Visualize equilibrium surface coverage phase diagram.

        Args:
        substrate_coverage (np.ndarray[float], Nc x Nc) - substrate coverage
        total_coverage (np.ndarray[float], Nc x Nc) - total coverage
        cmap (matplotlib.colors.ColorMap)
        vmin, vmax (float) - bounds for colormap
        stretch (bool) - if True, keep true aspect ratio
        bg_color (np.uint8) - background fill color, defaults to grey
        ax (matplotlib.axes.AxesSubplot)
        kwargs: keyword arguments for visualization

        Returns:
        ax (matplotlib.axes.AxesSubplot)
        """

        # create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))

        # apply colormap to create image
        norm = Normalize(vmin, vmax)
        im = cmap(norm(substrate_coverage))

        # mask total coverage using alpha channel (low areas are transparent)
        if total_coverage is not None:
            im[:, :, -1] = total_coverage
            bg = np.empty(substrate_coverage.shape + (3,), dtype=np.uint8)
            bg.fill(bg_color)
            ax.imshow(bg)

        # render image
        ax.imshow(im)

        return ax

    def _format_ax(self, ax, stretch=True):
        """
        Format axis labels, orientation, and tickmarks.

        Args:
        ax (matplotlib.axes.AxesSubplot)
        stretch (bool) - if True, keep true aspect ratio
        """

        # invert axes
        ax.invert_yaxis()

        # label axes
        index_to_name = {v:k for k,v in self.names.items()}
        ax.set_xlabel('{:s} concentration (nM)'.format(index_to_name[1]))
        ax.set_ylabel('{:s} concentration (nM)'.format(index_to_name[2]))

        # format xticks
        xtx = np.linspace(0, self.Cx.max(), 5)
        tick_positions = np.interp(xtx, self.Cx, np.arange(self.Cx.size))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(['{:.0f}'.format(x) for x in xtx*1e9])

        # format yticks
        ytx = np.linspace(0, self.Cy.max(), 5)
        tick_positions = np.interp(ytx, self.Cy, np.arange(self.Cy.size))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(['{:.0f}'.format(x) for x in ytx*1e9])

        # set aspect ratio
        if stretch:
            aspect = (self.Nc[0] / self.Nc[1])
        else:
            aspect = 1
        ax.set_aspect(aspect)


class BindingModel(SimpleModel):
    """
    Statistical mechanical model describing the equilibrium fractional occupancy of predefined regulatory elements bound by one or more competing transcription factor substrates. Numerical solution entails full enumeration of all microstates.

    This object contains equilibrium fractional occupancies of a binding element for all pairwise concentrations in a concentration grid.
    """

    def __init__(self, element, cut_depth=None, **kwargs):
        """
        Instantiate 2D phase diagram.

        Args:
        element (binding.model.elements.Element) - binding element
        cut_depth (int) - tree depth at which paralellization begins

        Keyword arguments:
        cmin (int or tuple) - minimum concentration for each protein, nM
        cmax (int or tuple) - maximum concentration for each protein, nM
        Nc (int or tuple) - number of concentrations for each protein
        names (dict[str]=int) - name for each binding protein
        """

        Grid.__init__(self, **kwargs)

        # evaluate occupancies
        self.evaluate_occupancies(element, cut_depth=cut_depth)

    def evaluate_occupancies(self, element, cut_depth=None):
        """
        Evaluate partition function to construct Ns x Nc x b occupancy array.

        Args:
        element (binding.model.elements.Element) - binding element
        cut_depth (int) - tree depth at which paralellization begins
        """

        # evaluate partition function
        pf = PartitionFunction(element, self.concentrations)
        occupancies = pf.c_get_occupancies(cut_depth)

        # store fraction occupancies
        self.occupancies = np.swapaxes(occupancies, 1, 2)
        self.total_occupancy = self.occupancies[:,:,1:].sum(axis=-1)

        # store overall surface coverage
        self.theta = 1-self.occupancies[:, :, 0].mean(axis=0).reshape(*self.Nc)

        # store binding element ETS site positions
        ets_sites = [i for i, x in enumerate(element.ets) if x == 1]
        self.ets = np.array(ets_sites).reshape(-1, 1)
        self.Ns = element.Ns

        # fit hill functional form to titration contours
        self.fit_hill()

    def plot_phase_diagram(self, species='Pnt', **kwargs):
        """
        Plot equilibrium surface coverage phase diagram.

        Args:
        species (str) - binding species
        kwargs: keyword arguments for visualization

        Returns:
        ax (matplotlib.axes.AxesSubplot)
        """

        # get substrate coverage
        if species.lower() == 'total':
            substrate_coverage = self.theta
        else:
            occupancies = self.occupancies[:, :, self.names[species]]
            substrate_coverage = occupancies.mean(axis=0).reshape(*self.Nc)

        # plot phase diagram
        ax = self._plot_phase_diagram(substrate_coverage, self.theta, **kwargs)

        # format axis
        self._format_ax(ax)

        return ax

    def plot_phase_map(self, func, **kwargs):
        """
        Visualizes function applied to phase diagram.

        Args:
        func (function(pnt, yan)) - function applied to pnt/yan occupancy
        kwargs: keyword arguments for phase diagram
        """

        # get occupancy by each species
        pnt = self.occupancies[:, :, self.names['Pnt']].mean(axis=0)
        pnt = pnt.reshape(*self.Nc)
        yan = self.occupancies[:, :, self.names['Yan']].mean(axis=0)
        yan = yan.reshape(*self.Nc)

        # plot function output
        ax = self._plot_phase_diagram(func(yan, pnt).T, **kwargs)

        return ax

    def plot_titration_contours(self,
                      species='Pnt',
                      variable='Pnt',
                      fixed=0,
                      cmap=plt.cm.viridis,
                      fig=None):
        """
        Plot titration contours.

        Args:
        species (str) - binding substrate whose surface coverage is shown
        variable (str) - titrated substrate
        fixed (int) - concentration index of non-titrated substrate
        cmap (matplotlib.colors.ColorMap)
        fig (matplotlib.figures.Figure)

        Returns:
        fig (matplotlib.figures.Figure)
        """

        # determine species indices
        species_dim = self.names[species]
        variable_dim = self.names[variable] - 1
        N = self.occupancies.shape[0]

        # get titration data
        if variable_dim == 1:
            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, fixed, :]
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[fixed, :] * 1e9
            fixed_concentration = self.concentrations[:, 0].reshape(*self.Nc)[fixed, 0] * 1e9
            fixed_species = 'Yan'

        elif variable_dim == 0:
            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, :, fixed]
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[:, fixed] * 1e9
            fixed_concentration = self.concentrations[:, 1].reshape(*self.Nc)[0, fixed] * 1e9
            fixed_species = 'Pnt'

        # create figure with vinding site positions
        fig = self.create_titration_figure(fig=fig, cmap=cmap)
        fig.suptitle('Fixed {:0.1f} nM {:s}'.format(fixed_concentration, fixed_species), fontsize=9)
        ax0, ax1 = fig.axes

        # calculate distance to nearest ets site
        prx = distance_matrix(np.arange(N).reshape(-1,1),self.ets).min(axis=1)

        # visualize binding sites colored by proximity to ets site
        ax0.imshow(prx.reshape(1,-1), cmap=cmap, aspect=1,interpolation='none')
        for site_index in self.ets:
            ax0.text(site_index, 0, 'E', color='w', ha='center', va='center')

        # format axis
        ax0.grid(color='k', linestyle='-', linewidth=2)
        ax0.set_yticks([])
        ax0.xaxis.set_ticks_position('top')
        ax0.set_xticks(np.arange(N)-0.5)
        ax0.set_xticklabels([])
        ax0.tick_params(length=0, width=0.5)

        # plot occupancy contours
        norm = Normalize(vmin=0, vmax=prx.max())
        for i, x in enumerate(occupancy):
            color = cmap(norm(prx[i]))
            ax1.plot(concentration, x, '-', color=color)

        ax1.set_ylim(0, 1)
        ax1.set_xlim(concentration.min(), concentration.max())
        ax1.set_ylabel('Occupancy ({:s})'.format(species), fontsize=8)
        ax1.set_xlabel('{:s} concentration (nM)'.format(variable), fontsize=8)

        return fig

    def create_titration_figure(self, fig=None, cmap=plt.cm.viridis):
        """
        Create titration figure.

        Args:
        fig (matplotlib.figures.Figure)
        cmap (matplotlib.colors.ColorMap)

        Returns:
        fig (matplotlib.figures.Figure)
        """

        # instantiate figure figure
        if fig is None:
            fig = plt.figure(figsize=(4, 4))

        # define grid specifications
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(1, 8), hspace=.15)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        return fig

    def plot_overall_titration_contour(self,
                      species='Pnt',
                      variable='Pnt',
                      fixed=0,
                      color='k',
                      ax=None):
        """
        Plot titration contour averaged across all binding sites.

        Args:
        species (str) - binding substrate whose surface coverage is shown
        variable (str) - titrated substrate
        fixed (int) - concentration index of non-titrated substrate
        color (str or tuple) - line color
        ax (matplotlib.axis instance)

        Returns:
        ax (matplotlib.axis instance)
        """

        # determine species indices
        species_dim = self.names[species]
        variable_dim = self.names[variable] - 1
        N = self.occupancies.shape[0]

        # get titration data
        if variable_dim == 1:
            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, fixed, :].mean(axis=0)
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[fixed, :] * 1e9
            fixed_concentration = self.concentrations[:, 0].reshape(*self.Nc)[fixed, 0] * 1e9
            fixed_species = 'Yan'

        elif variable_dim == 0:
            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, :, fixed].mean(axis=0)
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[:, fixed] * 1e9
            fixed_concentration = self.concentrations[:, 1].reshape(*self.Nc)[0, fixed] * 1e9
            fixed_species = 'Pnt'

        # create figure 
        if ax is None:
            fig, ax = plt.subplots()

        # plot occupancy contours
        ax.plot(concentration, occupancy, '-', color=color)
        
        # format axis
        ax.set_ylim(0, 1)
        ax.set_xlim(concentration.min(), concentration.max())
        ax.set_ylabel('Occupancy ({:s})'.format(species), fontsize=8)
        ax.set_xlabel('{:s} concentration (nM)'.format(variable), fontsize=8)
        ax.set_title('Fixed {:0.1f} nM {:s}'.format(fixed_concentration, fixed_species), fontsize=9)

        return ax

    def fit_hill(self, **kwargs):
        """ Fit HillModel to occupancies. """
        occupancies = self.occupancies.mean(axis=0)[:, 1:]
        concentrations = self.concentrations * 1e9
        self.model = HillModel(concentrations, occupancies, **kwargs)

    def plot_hill_fit(self, 
                      species='Pnt', 
                      variable='Pnt', 
                      cmap=plt.cm.plasma, 
                      figsize=(3, 2)):
        """
        Visualize Hill functional fit to all titration contours.

        Args:
        species (str) - binding substrate whose surface coverage is shown
        variable (str) - titrated substrate
        cmap (matplotlib.colors.ColorMap) - contour colors
        figsize (tuple) - figure size

        Returns:
        fig (matplotlib.figures.Figure)
        """
        
        # determine species indices
        species_dim = self.names[species]
        variable_dim = self.names[variable] - 1
        
        # determine number of concentrations
        num_concentrations = self.Nc[not variable_dim]    
        norm = Normalize(0, num_concentrations)
        
        # create figure
        fig, ax = plt.subplots(figsize=figsize)

        # iterate across yan concentrations
        for y in range(0, num_concentrations):

            # get concentrations
            x = self.model.x[y::num_concentrations, variable_dim]

            # plot data and prediction
            data = self.model.y[y::num_concentrations, species_dim-1]
            prediction = self.model.yp[y::num_concentrations, species_dim-1]

            color = cmap(norm(y))
            ax.plot(x, prediction, '-', color=color, linewidth=2)
            ax.scatter(x, data, c=color, s=50, marker=r'$\diamond$')

        _ = ax.set_ylabel('{:s} Occupancy'.format(species), fontsize=8)
        _ = ax.set_xlabel('{:s} concentration (nM)'.format(variable), fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(self.model.x[:, variable_dim].min(), self.model.x[:, variable_dim].max())
        return fig
