# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array, clone
from libc.math cimport exp

# import microstates
from elements cimport cElement


cdef class cPF:

    # attributes
    cdef int Ns
    cdef long long Nm
    cdef int b
    cdef int n
    cdef array C
    cdef array occupancies
    cdef double Z
    cdef cElement element

    def __init__(self, element, concentrations):
        self.Ns = element.Ns
        self.Nm = element.Nm
        self.b = element.b
        self.n = element.b-1
        self.C = array('d', concentrations)
        self.set_element(element)

    cpdef array get_occupancies(self):
        """ Get flattened b x Ns occupancy array. """
        self.reset()
        self.set_occupancies()
        return self.occupancies

    cdef void set_element(self, element):
        """ Set microstate energies. """
        self.element = element.get_c_element(element_type='base')

    cdef void reset(self):
        """ Initialize probabilities and occupancies with zeros. """
        self.occupancies = clone(array('d'), self.n*self.Ns, True)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_occupancies(self) nogil:
        cdef int state
        cdef double p

        # initialize partition function (define ground state)
        self.Z = 1.

        # run recursion
        for state in xrange(self.b):
            p = self.set_occupancy(0, state, 0, 0., 1.)

        # apply normalization
        self.normalize()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double set_occupancy(self,
                            int site,
                            int state,
                            int neighbor_state,
                            double deltaG,
                            double degeneracy,) nogil:

        cdef int index
        cdef int new_state
        cdef double boltzmann_factor = 0
        cdef double p = 0
        cdef double occupancy

        # evaluate binding strength of current microstate
        if state != 0:

            # update microstate free energy
            deltaG += self.element.get_binding_energy(site, state)
            if state == neighbor_state:
                deltaG += self.element.gamma.data.as_doubles[state-1]

            # update microstate degeneracy
            degeneracy *= self.C.data.as_doubles[state-1]
            if degeneracy == 0:
                return 0

        # recurse (evaluate derivative microstates)
        if site < (self.Ns - 1):
            for new_state in xrange(self.b):
                p += self.set_occupancy(site+1, new_state, state, deltaG, degeneracy)

        # if site is unoccupied, don't make an assignment but continue
        if state != 0:

            # evaluate (non-normalized) partition
            boltzmann_factor = exp(-deltaG/(self.element.R*self.element.T))*degeneracy
            p += boltzmann_factor

            # assign probabilities to current site/occupant pair
            index = (state-1)*self.Ns + site
            occupancy = self.occupancies.data.as_doubles[index]
            self.occupancies.data.as_doubles[index] = occupancy + p

            # update partition function and microstate index
            self.Z += boltzmann_factor

        return p

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void normalize(self) nogil:
        """ Normalize occupancies by partition function value. """
        cdef int i
        cdef double occupancy
        for i in xrange(self.n*self.Ns):
            occupancy = self.occupancies.data.as_doubles[i]
            self.occupancies.data.as_doubles[i] = occupancy/self.Z


cdef class cPF_test(cPF):
    cdef dict cache


class PartitionFunction:
    def __init__(self, element, concentrations):

        # get system dimensions
        self.Ns = element.Ns
        self.b = element.b
        self.n = element.b - 1

        # extra for python version
        self.Nm = element.Nm
        self.element = element
        self.concentrations = concentrations

    def c_get_occupancies(self, method='base'):
        """ Get Nc x b x Ns occupancy array. """

        # instantiate partition function
        if method == 'test':
            c_pf = cPF_test(self.element, self.concentrations)
        else:
            c_pf = cPF(self.element, self.concentrations)

        # convert array to ndarray
        shape = (self.b-1, self.Ns)
        c_occupancies = c_pf.get_occupancies()
        occupancies = np.array(c_occupancies, dtype=np.float64).reshape(*shape)

        # append balance
        balance = (1 - occupancies.sum(axis=0)).reshape(1, self.Ns)
        occupancies = np.append(balance, occupancies, axis=0)

        return occupancies
