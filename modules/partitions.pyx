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
    """
    Defines a partition function for a given binding element.

    Attributes:
        element (cElement instance) - binding site element of length Ns
        Nc (int) - number of unique protein concentration pairs
        C (double*) - protein concentrations, flattened N x Nc
        weights (double*) - boltzmann weights, flattened array
        degeneracy (double*) - degeneracy terms, flattened (Ns+1) x Nc
        occupancies (double*) - binding site occupancies, flattened Ns x N x Nc
        Z (double*) - partition function values, 1 x Nc array
    """

    @staticmethod
    def from_python(element, concentrations, cut_depth=None):
        """
        Initialize partition function by constructing cTree instance.

        Args:
        element (Element instance) - binding element
        concentrations (np.ndarray) protein concentrations, Nc x 2 doubles
        cut_depth (None or int) - depth at which parallel branching occurs
        """
        if cut_depth is None:
            cut_depth = 0

        c_element = element.get_c_element()
        Nc = concentrations.shape[0]
        C = array('d', concentrations.T.flatten())
        return cParallelPF(c_element, Nc, C, 0, cut_depth=cut_depth)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void evaluate_occupancies(self):
        """ Traverse tree and evaluate occupancies. """
        self.traverse()
        self.normalize()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray get_occupancies(self):
        """ Get site occupancy array (Ns x n x Nc) """

        # define shape
        shape = (self.max_depth, self.element.n, self.Nc)

        # construct occupancies ndarray from occupancies ptr
        cdef int i
        cdef int size = shape[0] * shape[1] * shape[2]
        cdef np.ndarray[double,ndim=1] occupancies = np.zeros(size, np.float64)
        for i in xrange(size):
            occupancies[i] = self.occupancies[i]

        return occupancies.reshape(*shape)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void normalize(self) nogil:
        """ Normalize binding site occupancies by partition function. """
        cdef int i, j, shift
        cdef double occupancy, Z

        # add one to partition function for empty probability
        for i in xrange(self.Nc):
            self.Z[i] += 1

        # normalize occupancies
        for i in xrange(self.max_depth*self.element.n):
            shift = i*self.Nc
            for j in xrange(self.Nc):
                occupancy = self.occupancies[shift+j] / self.Z[j]
                self.occupancies[shift+j] = occupancy


class PartitionFunction:
    """ Defines a partition function that enumerates probabilities of all element microstates for a given set of protein concentrations. """

    def __init__(self, element, concentrations):

        # get system dimensions
        self.element = element
        self.concentrations = concentrations
        self.Nc = concentrations.shape[0]

    def c_get_occupancies_parallel(self, cut_depth=None):
        """ Get Ns x b x Nc occupancy array. """

        # instantiate partition function
        c_pf = cParallelPF.from_python(self.element, self.concentrations, cut_depth)

        # evaluate partition function and get occupancies
        c_pf.evaluate_occupancies()
        occupancies = c_pf.get_occupancies()

        # append balance
        shape = (self.element.Ns, 1, self.Nc)
        balance = (1-occupancies.sum(axis=1)).reshape(*shape)
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
