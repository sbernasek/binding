__author__ = 'Sebastian Bernasek'

import os
import matplotlib.pyplot as plt


class Base:
    """
    Base class for figures providing some common methods.

    Attributes:
    df (pd.DataFrame) - data
    fig (matplotlib.figure.Figure)
    """

    # set default directory as class attribute
    directory = 'graphics'

    def __init__(self, df):
        """
        Instantiate Figure.

        Args:
        df (pd.DataFrame) - data for figure
        """
        self.df = df
        self.fig = None

    def save(self,
             name='figure',
             directory=None,
             fmt='pdf',
             dpi=300,
             rasterized=False):
        """
        Save figure.

        Args:
        name (str) - filename
        directory (str) - target directory
        fmt (str) - file format
        dpi (int) - resolution
        rasterized (bool) - if True, save rasterized version
        """

        # use class default path if none provided
        if directory is None:
            directory = self.directory

        # construct filepath
        filepath = os.path.join(directory, name+'.'+fmt)

        # save figure
        self.fig.savefig(filepath,
                         dpi=dpi,
                         format=fmt,
                         transparent=True,
                         rasterized=rasterized)

    def show(self):
        """ Display figure. """
        plt.show(self.fig)

    @staticmethod
    def create_figure(figsize=(1, 1.5), nrows=1, ncols=1):
        """
        Create figure.

        Args:
        figsize (tuple) - figure size
        nrows (int) - number of rows
        ncols (int) - number of columns

        Returns:
        fig (matplotlib.figure.Figure)
        """
        fig, _ = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        return fig


class ColorBar(Base):
    """
    Standalone colorbar.
    """

    def __init__(self, figsize=(5, 1), **kwargs):
        """
        Instantiate standalone colorbar.

        Args:
        figsize (tuple) - dimensions
        vmin, vmax (float) - bounds for colorscale
        cmap (matplotlib.colors.ColorMap) - color map
        """
        self.fig = self.create_figure(figsize=figsize)
        self.render(**kwargs)

    def render(self,
                vmin=0, vmax=1,
                cmap=plt.cm.plasma,
                label=None):
        """
        Plot standalone colorbar.

        Args:
        vmin, vmax (float) - bounds for colorscale
        cmap (matplotlib.colors.ColorMap) - color map
        """

        # get axis
        ax = self.fig.axes[0]

        # plot colorbar
        cbar = ColorbarBase(ax,
                            cmap=cmap,
                            norm=Normalize(vmin, vmax),
                            orientation='horizontal')

        # format ticks
        ax.xaxis.set_ticks_position('top')
        cbar.set_ticks(np.arange(0, 1.1, .2))
        ax.tick_params(pad=1)

        # add label
        if label is not None:
            cbar.set_label(label, labelpad=5)

