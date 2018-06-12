# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

import cython
from libc.math cimport exp
import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array, clone
from bits cimport get_ternary_repr, c_bits_to_int


cdef class cMicrostates:

    def __init__(self, unsigned int Ns,
                 unsigned int N_species,
                 dict params,
                 array ets,
                 double R = 1.987204118*1E-3, double T = 300):

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
        self.set_params(params)

        # define ETS sites
        self.ets = ets

        # enumerate microstate energies
        self.reset()
        self.set_energies()

    cpdef np.ndarray get_E(self):
        """ Returns microstate energies as np array. """
        return np.array(self.E, dtype=np.float64).reshape(self.Nm)

    cdef void set_params(self, dict params):
        cdef unsigned int index
        for index in xrange(self.b-1):
            self.alpha.data.as_doubles[index] = params['alpha'][index]
            self.beta.data.as_doubles[index] = params['beta'][index]
            self.gamma.data.as_doubles[index] = params['gamma'][index]

    cdef void reset(self):
        """ Initialize binding energies. """
        self.E = clone(array('d'), self.Nm, True)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_energies(self) nogil:
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_binding_energy(self, unsigned int site_index, unsigned int site_state) nogil:
        """ Get binding energy for single site. """
        cdef double energy
        if self.ets.data.as_uints[site_index] == 1:
            energy = self.alpha.data.as_doubles[site_state-1]
        else:
            energy = self.beta.data.as_doubles[site_state-1]
        return energy


cdef class cRecursiveMicrostates(cMicrostates):
    """ Equivalent version in which arrays are ordered by L/R traversal. """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_energies(self) nogil:
        cdef unsigned int site_state
        cdef unsigned int index
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
                          unsigned int site_index,
                          unsigned int site_state,
                          unsigned int neighbor_state,
                          double E) nogil:
        """ Recursive set_energy function. """
        cdef unsigned int new_state

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


cdef class cIterativeMicrostates(cMicrostates):

    cdef void reset(self):
        """ Initialize binding energies and occupancy counts. """
        self.E = clone(array('d'), self.Nm, True)
        self.a = clone(array('I'), self.Nm*self.n, True)

    cpdef np.ndarray get_a(self):
        """ Returns microstate TF occupancy counts as np array. """
        return np.array(self.a, dtype=np.int64).reshape((self.b-1, self.Nm))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_energies(self) nogil:
        cdef unsigned int site_state
        cdef unsigned int index
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
                          unsigned int site_index,
                          unsigned int site_state,
                          unsigned int neighbor_microstate,
                          unsigned int neighbor_state,
                          unsigned int a1, unsigned int a2, double E) nogil:
        """ Recursive set_energy function. """

        cdef unsigned int microstate
        cdef unsigned int new_state

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
            self.a.data.as_uints[microstate] = a1
            self.a.data.as_uints[self.Nm + microstate] = a2

        # recurse
        if site_index < (self.Ns - 1):
            for new_state in xrange(self.b):
                self.set_energy(site_index+1, new_state, microstate, site_state, a1, a2, E)


    # cpdef tuple get_energy_contributions(self):
    #     """ Returns a/b/g energy contributions to each microstate. """

    #     cdef array alpha, beta, gamma
    #     cdef unsigned int index
    #     cdef double energy
    #     cdef unsigned int j, k
    #     cdef array microstate
    #     cdef unsigned int n, site_state, neighbor

    #     alpha = clone(array('d'), self.Nm*2, True)
    #     beta = clone(array('d'), self.Nm*2, True)
    #     gamma = clone(array('d'), self.Nm*2, True)

    #     # SET GROUND STATES
    #     for index in xrange(1, self.b):
    #         if self.ets.data.as_uints[0] == 1:
    #             energy = self.alpha.data.as_doubles[index-1]
    #             alpha.data.as_doubles[(index-1)*self.Nm + 0] = energy
    #         else:
    #             energy = self.beta.data.as_doubles[index-1]
    #             beta.data.as_doubles[(index-1)*self.Nm + 0] = energy

    #     # SET ALL OTHER STATES
    #     for k in xrange(self.b, self.Nm):

    #         # get microstate
    #         n, microstate = get_ternary_repr(k) # note there are n+1 digits
    #         site_state = microstate.data.as_uints[n]

    #         # add N-1 neighbor's energy
    #         neighbor = c_bits_to_int(microstate, n, self.b)
    #         for j in xrange(self.b-1):
    #             alpha.data.as_doubles[j*self.Nm + k] += alpha.data.as_doubles[j*self.Nm + neighbor]
    #             beta.data.as_doubles[j*self.Nm + k] += beta.data.as_doubles[j*self.Nm + neighbor]
    #             gamma.data.as_doubles[j*self.Nm + k] += gamma.data.as_doubles[j*self.Nm + neighbor]

    #         # add binding energy
    #         if self.ets.data.as_uints[n] == 1:
    #             energy = self.alpha.data.as_doubles[site_state-1]
    #             alpha.data.as_doubles[(site_state-1)*self.Nm + k] += energy
    #         else:
    #             energy = self.beta.data.as_doubles[site_state-1]
    #             beta.data.as_doubles[(site_state-1)*self.Nm + k] += energy

    #         # if N-1 neighbor shares same occupant, add gamma
    #         if site_state == microstate.data.as_uints[n-1] and site_state != 0:
    #             gamma.data.as_doubles[(site_state-1)*self.Nm + k] += self.gamma.data.as_doubles[site_state-1]

    #     a = np.array(alpha, dtype=np.float64).reshape((self.b-1, self.Nm))
    #     b = np.array(beta, dtype=np.float64).reshape((self.b-1, self.Nm))
    #     g = np.array(gamma, dtype=np.float64).reshape((self.b-1, self.Nm))

    #     return (a, b, g)





class Microstates:
    def __init__(self, Ns, N_species=2, params=None, ets=(0,)):

        # set system size
        self.Ns = Ns
        self.b = N_species + 1
        self.Nm = self.b**Ns
        self.params = params

        # set ets sites
        site_indices = np.zeros(int(Ns), dtype=np.uint8)
        for i in ets:
            site_indices[int(i)] = 1
        self.ets = array('I', site_indices)

    def get_c_microstates(self, recursive=True, **kwargs):
        """ Get cMicrostates object. """
        params = (self.Ns, self.b-1, self.params, self.ets)
        if recursive:
            ms = cRecursiveMicrostates(*params, **kwargs)
        else:
            ms = cIterativeMicrostates(*params, **kwargs)
        return ms

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
