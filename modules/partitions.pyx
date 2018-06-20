# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array, clone
from libc.math cimport exp

# import elements and recursion structure
from elements cimport cElement
from trees cimport cTree


cdef class cPF:

    # attributes
    cdef int Ns
    cdef int b
    cdef int n
    cdef array C
    cdef array occupancies
    cdef double Z
    cdef cElement element

    def __init__(self, element, concentrations):
        self.Ns = element.Ns
        self.b = element.b
        self.n = element.b-1
        self.C = array('d', concentrations.flatten())
        self.set_element(element)

    cpdef array get_occupancies(self):
        """ Get flattened b x Ns occupancy array. """
        self.reset()
        self.evaluate()
        return self.occupancies

    cdef void set_element(self, element):
        """ Set binding element. """
        self.element = element.get_c_element(element_type='base')

    cdef void reset(self):
        """ Initialize occupancies with zeros. """
        self.occupancies = clone(array('d'), self.n*self.Ns, True)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void evaluate(self) with gil:
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
                            double degeneracy,) with gil:

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
            boltzmann_factor = exp(deltaG)*degeneracy
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
    cdef void normalize(self) with gil:
        """ Normalize occupancies by partition function value. """
        cdef int i
        cdef double occupancy
        for i in xrange(self.n*self.Ns):
            occupancy = self.occupancies.data.as_doubles[i]
            self.occupancies.data.as_doubles[i] = occupancy/self.Z


cdef class cPF_test(cPF):
    cdef dict cache


cdef class cParallelPF(cTree):

    def __init__(self, element, concentrations, cut=None):
        cdef cElement c_element = element.get_c_element()
        cdef int Nc = concentrations.shape[0]

        # set cut point for parallelization (splits branches after cut_point)
        cdef int cut_point
        if cut is None:
            cut_point = 0
        else:
            cut_point = cut

        cdef array C = array('d', concentrations.T.flatten())
        cTree.__init__(self, c_element, Nc, C, cut_point)

    cpdef array get_occupancies(self):
        """ Get flattened b x Ns occupancy array. """
        self.traverse()
        self.normalize()
        return self.occupancies

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void normalize(self) with gil:
        """ Normalize occupancies by partition function value. """
        cdef int i, j
        cdef double occupancy, Z
        cdef int shift
        for i in xrange(self.element.Ns*self.element.n):
            shift = i*self.Nc
            for j in xrange(self.Nc):
                Z = self.Z.data.as_doubles[j]
                occupancy = self.occupancies.data.as_doubles[shift+j] / Z
                self.occupancies.data.as_doubles[shift+j] = occupancy


class PartitionFunction:
    """ Defines a partition function that enumerates probabilities of all element microstates for a given set of protein concentrations. """

    def __init__(self, element, concentrations):

        # get system dimensions
        self.element = element
        self.concentrations = concentrations
        self.Nc = concentrations.shape[0]

    def c_get_occupancies_parallel(self, cut=None):
        """ Get Nc x b x Ns occupancy array. """

        # instantiate partition function
        c_pf = cParallelPF(self.element, self.concentrations, cut)

        # convert array to ndarray
        shape = (self.element.Ns, self.element.n, self.Nc)
        c_occupancies = c_pf.get_occupancies()
        occupancies = np.array(c_occupancies, dtype=np.float64).reshape(*shape)

        # append balance
        balance = (1 - occupancies.sum(axis=1)).reshape(self.element.Ns, 1, self.Nc)
        occupancies = np.append(balance, occupancies, axis=1)

        return occupancies

    def c_get_occupancies(self, method='base'):
        """ Get Nc x b x Ns occupancy array. """

        # instantiate partition function
        if method == 'test':
            c_pf = cPF_test(self.element, self.concentrations)
        else:
            c_pf = cPF(self.element, self.concentrations)

        # convert array to ndarray
        shape = (self.element.n, self.element.Ns)
        c_occupancies = c_pf.get_occupancies()
        occupancies = np.array(c_occupancies, dtype=np.float64).reshape(*shape)

        # append balance
        balance = (1 - occupancies.sum(axis=0)).reshape(1, self.element.Ns)
        occupancies = np.append(balance, occupancies, axis=0)

        return occupancies
