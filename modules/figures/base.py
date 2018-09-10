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

