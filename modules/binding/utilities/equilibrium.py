import numpy as np


class Equilibrium:

    def __init__(self, T=300, R=1.987204118*1E-3):
        self.RT = R*T

    def get_dG(self, KD):
        return self.RT*np.log(KD)

    def get_K(self, dG):
        return np.exp(-dG/(self.RT))
