import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial import distance_matrix

from partitions import PartitionFunction
from hill import HillModel


tickpad=2
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.labelcolor'] = 'k'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelpad'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 1
plt.rcParams['xtick.major.pad'] = tickpad
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 1
plt.rcParams['ytick.major.pad'] = tickpad
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['ytick.direction'] = 'in'



class ConcentrationSweep:
    """ Class defines a set of pairwise protein concentrations. """

    def __init__(self, cmin=0, cmax=100, Nc=25, names=None):

        """
        Args:
        cmin (int or tuple) - minimum concentration for each protein, nM
        cmax (int or tuple) - maximum concentration for each protein, nM
        Nc (int or tuple) - number of concentrations for each protein
        names (dict[str]=int) - name for each binding protein

        """

        # set names
        if names is None:
            names = dict(none=0, Pnt=1, Yan=2)
        self.names = names

        # define concentrations
        self.cmin = self.get_tuple(cmin)
        self.cmax = self.get_tuple(cmax)
        self.Nc = self.get_tuple(Nc)

        # set concentrations
        self.Cx = np.linspace(self.cmin[0], self.cmax[0], self.Nc[0]) * 1E-9
        self.Cy = np.linspace(self.cmin[1], self.cmax[1], self.Nc[1]) * 1E-9
        xx, yy = np.meshgrid(*(self.Cx, self.Cy), indexing='xy')
        self.concentrations = np.stack((xx.T, yy.T)).reshape(2, -1).T

    @staticmethod
    def get_tuple(x):
        if type(x) in (float, int, np.float64, np.int64):
            return (x, x)
        else:
            return x

    def get_occupancies(self, element, cut_depth=None):
        """
        Evaluate binding site occupancies.

        Args:
        element (Element instance) - binding element
        cut_depth (int) - depth at which parallel evaluation of subtrees occurs
        """
        return Occupancies(element, cut_depth, cmin=self.cmin, cmax=self.cmax, Nc=self.Nc, names=self.names)


class Occupancies(ConcentrationSweep):
    """ Defines occupancies of an element under a set of concentrations. """

    def __init__(self, element, cut_depth=None, **kwargs):
        ConcentrationSweep.__init__(self, **kwargs)

        # set occupancies
        self.set_occupancies(element, cut_depth=cut_depth)

    def set_occupancies(self, element, cut_depth=None):
        """ Get Ns x Nc x b occupancy array. """
        pf = PartitionFunction(element, self.concentrations)
        occupancies = pf.c_get_occupancies(cut_depth)
        self.occupancies = np.swapaxes(occupancies, 1, 2)
        self.total_occupancy = self.occupancies[:,:,1:].sum(axis=-1)
        self.ets = np.array([i for i, x in enumerate(element.ets) if x == 1]).reshape(-1, 1)
        self.Ns = element.Ns
        self.fit_model()

    def plot_occupancy(self, site=0, species='Pnt', **kwargs):
        """ Plot occupancy for an individual binding site. """
        if species.lower() == 'total':
            zz = 1 - self.occupancies[site, :, 0].reshape(*self.Nc)
        else:
            zz = self.occupancies[site, :, self.names[species]].reshape(*self.Nc)

        fig = self.show(zz.T, species, **kwargs)
        fig.suptitle('Site N={:d}'.format(site), fontsize=9)
        return fig

    def plot_overall_occupancy(self, species='Pnt', mask=None, title=False, **kwargs):
        """ Plot overall occupancy across the entire element. """

        # get total occupancy
        total = 1 - self.occupancies[:, :, 0].mean(axis=0).reshape(*self.Nc)
        if mask is not None:
            mask = total.T

        if species.lower() == 'total':
            zz = total
            #zz = np.ones_like(zz) - zz
        else:
            zz = self.occupancies[:, :, self.names[species]].mean(axis=0).reshape(*self.Nc)

        fig = self.show(zz.T, species, mask=mask, **kwargs)
        if title:
            fig.suptitle('Over all sites', fontsize=9)
        return fig

    def plot_colorbar(figsize=(5, 1), vmin=0, vmax=1, cmap=plt.cm.plasma):

        fig, ax_cbar = plt.subplots(figsize=figsize)
        norm = Normalize(vmin, vmax)
        cbar = ColorbarBase(ax_cbar, cmap=cmap, norm=norm, orientation='horizontal')
        ax_cbar.xaxis.set_ticks_position('top')
        cbar.set_ticks(np.arange(0, 1.1, .2))
        ax_cbar.tick_params(labelsize=7, pad=1)
        label = 'Fractional Occupancy'
        if name is not None:
            label = label + ' ({:s})'.format(name)
        cbar.set_label(label, fontsize=8, labelpad=5)
        #ax_cbar.xaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
        return fig

    def show(self, zz, name=None, mask=None, cmap=plt.cm.plasma, vmin=0, vmax=1, stretch=True, bg_color=70, figsize=(4, 4)):

        # create figure
        fig, ax = plt.subplots(figsize=figsize)

        # create image
        norm = Normalize(vmin, vmax)
        im = cmap(norm(zz))


        # apply mask as transparency
        if mask is not None:
            im[:, :, -1] = mask
            bg = np.empty(zz.shape + (3,), dtype=np.uint8)
            bg.fill(bg_color)
            ax.imshow(bg)

        # visualize occupancy
        ax.imshow(im)
        #ax.imshow(zz, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel('Pnt concentration (nM)')
        ax.set_ylabel('Yan concentration (nM)')

        # format ticks
        self.format_ticks(ax, stretch=stretch)

        return fig

    def format_ticks(self, ax, format_x=True, format_y=True, stretch=True):

        if format_x:
            xtx = np.linspace(0, self.Cx.max(), 5)
            tick_positions = np.interp(xtx, self.Cx, np.arange(self.Cx.size))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(['{:.0f}'.format(x) for x in xtx*1e9])
        if format_y:
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

    def plot_contours(self, species='Pnt', variable='Pnt', fixed=0, cmap=plt.cm.viridis, figsize=(4, 4)):

        # get data
        species_dim = self.names[species]
        variable_dim = self.names[variable]-1
        N = self.occupancies.shape[0]
        if variable_dim == 0:

            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, :, fixed]
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[:, fixed] * 1e9
            fixed_concentration = self.concentrations[:, 1].reshape(*self.Nc)[0, fixed] * 1e9
            fixed_species = 'Yan'
        elif variable_dim == 1:
            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, fixed, :]
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[fixed, :] * 1e9
            fixed_concentration = self.concentrations[:, 0].reshape(*self.Nc)[fixed, 0] * 1e9
            fixed_species = 'Pnt'

        # create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(1, 8), hspace=0.15)
        fig.suptitle('Fixed {:0.1f} nM {:s}'.format(fixed_concentration, fixed_species), fontsize=9)

        # calculate distance to nearest ets site
        proximity = distance_matrix(np.arange(N).reshape(-1, 1), self.ets).min(axis=1)

        # visualize binding sites, colored by proximity to ets site
        ax0 = plt.subplot(gs[0])
        ax0.imshow(proximity.reshape(1,-1), cmap=cmap, aspect=1, interpolation='none')
        ax0.grid(color='k', linestyle='-', linewidth=2)
        ax0.set_yticks([])
        ax0.xaxis.set_ticks_position('top')
        ax0.set_xticks(np.arange(N)-0.5)
        ax0.set_xticklabels([])
        ax0.tick_params(labelsize=8, pad=0)

        # add ETS sites
        for site_index in self.ets:
            ax0.text(site_index, 0, 'E', fontsize=8, color='w', ha='center', va='center')

        # plot occupancy contours
        ax1 = plt.subplot(gs[1])
        norm = Normalize(vmin=0, vmax=proximity.max())
        for i, x in enumerate(occupancy):
            color = cmap(norm(proximity[i]))
            ax1.plot(concentration, x, '-', color=color)

        ax1.set_ylim(0, 1)
        ax1.set_xlim(concentration.min(), concentration.max())
        ax1.set_ylabel('Fractional occupancy ({:s})'.format(species), fontsize=8)
        ax1.set_xlabel('{:s} concentration (nM)'.format(variable), fontsize=8)

        return fig

    def create_figure(self, variable='Pnt', fixed=0, cmap=plt.cm.viridis, figsize=(4, 4)):

        # create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=(1, 8), hspace=0.15)

        # label fixed species
        variable_dim = self.names[variable]-1
        if variable_dim == 0:
            fixed_concentration = self.concentrations[:, 1].reshape(*self.Nc)[0, fixed] * 1e9
            fixed_species = 'Yan'
        elif variable_dim == 1:
            fixed_concentration = self.concentrations[:, 0].reshape(*self.Nc)[fixed, 0] * 1e9
            fixed_species = 'Pnt'
        fig.suptitle('Fixed {:0.1f} nM {:s}'.format(fixed_concentration, fixed_species), fontsize=9)

        # calculate distance to nearest ets site
        sites = np.arange(self.Ns)
        proximity = distance_matrix(sites.reshape(-1, 1), self.ets).min(axis=1)
        norm = Normalize(vmin=0, vmax=proximity.max())

        # visualize binding sites, colored by proximity to ets site
        ax0 = plt.subplot(gs[0])
        ax0.imshow(proximity.reshape(1,-1), cmap=cmap, aspect=1, interpolation='none')
        ax0.grid(color='k', linestyle='-', linewidth=2)
        ax0.set_yticks([])
        ax0.xaxis.set_ticks_position('top')
        ax0.set_xticks(sites-0.5)
        ax0.set_xticklabels([])
        ax0.tick_params(labelsize=8, pad=0)

        # add ETS sites
        for site_index in self.ets:
            ax0.text(site_index, 0, 'E', fontsize=8, color='w', ha='center', va='center')

        # create main axis
        ax1 = plt.subplot(gs[1])
        ax1.set_ylim(0, 1)
        if variable_dim == 0:
            ax1.set_xlim(self.Cx.min()*1e9, self.Cx.max()*1e9)
        else:
            ax1.set_xlim(self.Cy.min()*1e9, self.Cy.max()*1e9)

        ax1.set_ylabel('Fractional occupancy ({:s})'.format(variable), fontsize=8)
        ax1.set_xlabel('{:s} concentration (nM)'.format(variable), fontsize=8)

        return fig, cmap, norm

    def add_contour(self, ax, species='Pnt', variable='Pnt', fixed=0, overall=True, color='r', **kwargs):

        # get data
        species_dim = self.names[species]
        variable_dim = self.names[variable]-1
        N = self.occupancies.shape[0]
        if variable_dim == 0:
            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, :, fixed]
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[:, fixed] * 1e9
        elif variable_dim == 1:
            occupancy = self.occupancies[:, :, species_dim].reshape(N, *self.Nc)[:, fixed, :]
            concentration = self.concentrations[:, variable_dim].reshape(*self.Nc)[fixed, :] * 1e9

        # plot occupancy contour(s)
        if overall:
            ax.plot(concentration, occupancy.mean(axis=0), '-', color=color)
        else:
            for i, x in enumerate(occupancy):
                ax1.plot(concentration, x, '-', color=color)

        return ax

    def fit_model(self, **kwargs):
        """ Fit HillModel to occupancies. """
        occupancies = self.occupancies.mean(axis=0)[:, 1:]
        concentrations = self.concentrations * 1e9
        self.model = HillModel(concentrations, occupancies, **kwargs)

    def show_model(self, cmap=plt.cm.plasma, figsize=(4, 3)):
        """ Visualize HillModel fit. """

        norm = Normalize(0, self.Nc[1])

        fig, ax = plt.subplots(figsize=figsize)

        # iterate across yan concentrations
        for y in range(0, self.Nc[1]):

            # get concentrations
            x = self.model.x[y::self.Nc[1], 0]

            # plot data and prediction
            data = self.model.y[y::self.Nc[1], 0]
            prediction = self.model.yp[y::self.Nc[1], 0]

            color = cmap(norm(y))
            ax.plot(x, prediction, '-', color=color, linewidth=2)
            ax.scatter(x, data, c=color, s=50, marker=r'$\diamond$')

        _ = ax.set_ylabel('Fractional occupancy (Pnt)', fontsize=8)
        _ = ax.set_xlabel('Pnt concentration (nM)', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(self.model.x[:, 0].min(), self.model.x[:, 0].max())
        return fig

