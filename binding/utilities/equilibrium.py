__author__ = 'Sebastian Bernasek'

import numpy as np


class Equilibrium:
    """
    Utility for converting between equilibrium dissociation constants and Gibbs free energies.

    Attributes:
    RT (float) - specific energy
    """

    def __init__(self, T=300, R=1.987204118*1E-3):
        """
        Instantiate conversion utility.

        Args:
        T (float) - temperature
        R (float) - gas constant
        """
        self.RT = R*T

    def KD_to_dG(self, KD):
        """ Convert dissociation constant to free energy. """
        return self.RT*np.log(KD)

    def dG_to_KD(self, dG):
        """ Convert free energy to dissociation constant. """
        return np.exp(-dG/(self.RT))
