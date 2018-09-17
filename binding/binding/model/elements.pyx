__author__ = 'Sebastian Bernasek'

# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array, clone
from libc.math cimport exp

from bits cimport get_ternary_repr
from binding.model.bits cimport get_ternary_repr, c_bits_to_int
from elements cimport cElement


cdef class cElement:
    """
    Object represents a binding element.

    Attributes:
    R (float) - gas constant
    T (float) - temperature
    b (int) - number of binding states (n+1)
    n (int) - number of binding species
    Ns (int) - number of binding sites
    Nm (int) - number of microstates
    alpha (array, length n) - binding energies to strong sites
    beta (array, length n) - binding energies to weak sites
    gamma (array, length n) -  polymerization energies
    ets (array, length Ns) - boolean array denoting strong sites
    """

    def __init__(self,
                 int Ns,
                 int N_species,
                 dict params,
                 array ets,
                 double R = 1.987204118*1E-3,
                 double T = 300,
                 int scale = 1):

        # set conditions
        self.R = R
        self.T = T

        # set system size
        self.b = N_species + 1
        self.n = N_species
        self.Ns = Ns
        self.Nm = self.b**Ns

        # set parameters
        self.alpha = clone(array('d'), N_species, False)
        self.beta = clone(array('d'), N_species, False)
        self.gamma = clone(array('d'), N_species, False)
        self.set_params(params, scale)

        # define ETS sites
        self.ets = ets

    cdef void set_params(self, dict params, int scale):
        """ Set binding energies. """
        cdef int index
        cdef double nRT = -1/(self.R*self.T)

        # get scaling
        if scale == 0:
            nRT = 1

        # set parameters
        for index in xrange(self.b-1):
            self.alpha.data.as_doubles[index] = params['alpha'][index] * nRT
            self.beta.data.as_doubles[index] = params['beta'][index] * nRT
            self.gamma.data.as_doubles[index] = params['gamma'][index] * nRT

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_binding_energy(self,
                                   int site_index,
                                   int site_state) with gil:
        """ Get binding energy for single site. """
        cdef double energy
        if self.ets.data.as_longs[site_index] == 1:
            energy = self.alpha.data.as_doubles[site_state-1]
        else:
            energy = self.beta.data.as_doubles[site_state-1]
        return energy

    cdef cElement truncate(self, int cut_point):
        """ Shortens element by removing sites before specified cut point. """

        cdef int Ns = self.Ns - cut_point
        cdef int index
        cdef dict params
        cdef array ets

        # get params
        params = dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma)

        # truncate ets sites
        ets = clone(array('l'), Ns, True)
        for index in xrange(Ns):
            ets.data.as_longs[index] = self.ets.data.as_longs[cut_point+index]

        return cElement(Ns, self.n, params, ets, self.R, self.T, 0)


cdef class cRecursiveElement(cElement):
    """
    Equivalent version in which arrays are ordered by L/R traversal of binding site positions.
    """

    def __init__(self, *args):
        cElement.__init__(self, *args)
        self.reset()
        self.set_energies()

    cpdef np.ndarray get_E(self):
        """ Returns microstate energies as np array. """
        return np.array(self.E, dtype=np.float64).reshape(self.Nm)

    cdef void reset(self):
        """ Initialize binding energies. """
        self.E = clone(array('d'), self.Nm, True)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_energies(self) with gil:
        cdef int site_state
        cdef int index
        cdef double energy

        # run recursion
        self.index = 1 # first microstate gets skipped
        for site_state in xrange(self.b):
            self.set_energy(0, site_state, 0, 0)

        # exponentiate
        for index in xrange(self.Nm):
            energy = exp(-self.E.data.as_doubles[index] / (self.R*self.T))
            self.E.data.as_doubles[index] = energy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_energy(self,
                          int site_index,
                          int site_state,
                          int neighbor_state,
                          double E) with gil:
        """ Recursive set_energy function. """
        cdef int new_state

        # if site is occupied, set energy value
        if site_state != 0:
            E += self.get_binding_energy(site_index, site_state)
            if site_state == neighbor_state:
                E += self.gamma.data.as_doubles[site_state-1]

        # recurse
        if site_index < (self.Ns - 1):
            for new_state in xrange(self.b):
                self.set_energy(site_index+1, new_state, site_state, E)

        # store values and increment index
        if site_state != 0:
            self.E.data.as_doubles[self.index] = E
            self.index += 1


cdef class cIterativeElement(cElement):
    """
    Equivalent version in which microstates are ordered by their base-b representation.
    """

    def __init__(self, *args):
        cElement.__init__(self, *args)
        self.reset()
        self.set_energies()

    cpdef np.ndarray get_E(self):
        """ Returns microstate energies as np array. """
        return np.array(self.E, dtype=np.float64).reshape(self.Nm)

    cpdef np.ndarray get_a(self):
        """ Returns microstate TF occupancy counts as np array. """
        return np.array(self.a, dtype=np.int64).reshape((self.b-1, self.Nm))

    cdef void reset(self):
        """ Initialize binding energies and occupancy counts. """
        self.E = clone(array('d'), self.Nm, True)
        self.a = clone(array('l'), self.Nm*self.n, True)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_energies(self) with gil:
        cdef int site_state
        cdef int index
        cdef double energy

        # run recursion
        for site_state in xrange(self.b):
            self.set_energy(0, site_state, 0, 0, 0, 0, 0)

        for index in xrange(self.Nm):
            energy = exp(-self.E.data.as_doubles[index] / (self.R*self.T))
            self.E.data.as_doubles[index] = energy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_energy(self,
                          int site_index,
                          int site_state,
                          long long neighbor_microstate,
                          int neighbor_state,
                          int a1, int a2, double E) with gil:
        """ Recursive set_energy function. """

        cdef long long microstate
        cdef int new_state

        # get microstate
        microstate = neighbor_microstate + site_state*(self.b**site_index)

        # if site is unoccupied, skip it
        if site_state != 0:

            # set binding energy
            E += self.get_binding_energy(site_index, site_state)
            if site_state == neighbor_state:
                E += self.gamma.data.as_doubles[site_state-1]
            self.E.data.as_doubles[microstate] = E

            # set occupancy
            if site_state == 1:
                a1 += 1
            else:
                a2 += 1
            self.a.data.as_longs[microstate] = a1
            self.a.data.as_longs[self.Nm + microstate] = a2

        # recurse
        if site_index < (self.Ns - 1):
            for new_state in xrange(self.b):
                self.set_energy(site_index+1, new_state, microstate, site_state, a1, a2, E)


class Element:
    """
    Class defines a single element bound by one or more proteins.

    Attributes:
    Ns (int) - number of binding sites
    b (int) - number of binding states (n+1)
    n (int) - number of binding species
    Nm (int) - number of microstates
    params (dict[param]=tuple) - parameter values, e.g. alpha, beta, gamma
    ets (array, length Ns) - boolean array denoting strong sites
    """

    def __init__(self, Ns, N_species=2, params=None, ets=(0,)):
        """
        Instantiate binding element.

        Args:
        Ns (int) - number of binding sites
        N_species (int) - number of binding species
        params (dict[param]=tuple) - parameter values, e.g. alpha, beta, gamma
        ets (tuple) - positional indices of strong binding sites
        """

        # set system size
        self.Ns = Ns
        self.b = N_species + 1
        self.n = N_species
        self.Nm = self.b**Ns
        self.params = params

        # set ets sites
        site_indices = np.zeros(int(Ns), dtype=np.int32)
        for i in ets:
            site_indices[int(i)] = 1
        self.ets = array('l', site_indices)

    def get_c_element(self, element_type='base', **kwargs):
        """ Get cMicrostates object. """
        params = (self.Ns, self.b-1, self.params, self.ets)
        if element_type == 'iterative':
            element = cIterativeElement(*params, scale=1, **kwargs)
        elif element_type == 'recursive':
            element = cRecursiveElement(*params, scale=1, **kwargs)
        else:
            element = cElement(*params, scale=1, **kwargs)
        return element

    def get_mask(self, v, p, indices):
        return ((indices//(self.b**p)) % self.b) == v

    def get_masks(self):
        """ Returns boolean masks in Nm x b x Ns array. """
        indices = np.arange(self.Nm)
        masks = np.zeros((self.Nm, self.b, self.Ns), dtype=bool)
        for k in range(self.Ns):
            for j in range(self.b):
                masks[:, j, k] = self.get_mask(j, k, indices)
        return masks
