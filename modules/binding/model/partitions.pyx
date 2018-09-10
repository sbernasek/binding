# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
import numpy as np
cimport numpy as np
from array import array
from cpython.array cimport array

# import elements and recursion structure
from elements cimport cElement
from trees cimport cRoot


cdef class cPF(cRoot):
    """
    Defines a partition function for a given binding element.

    Attributes:
        element (cElement instance) - binding site element of length Ns
        max_depth (int) - maximum depth of tree (Ns)
        Nc (int) - number of unique protein concentration pairs
        cut_depth (int) - depth of parallel evaluation of subtrees
    """

    @staticmethod
    def from_python(element, concentrations, cut_depth=None):
        """
        Initialize partition function by constructing cTree instance.

        Args:
        element (Element instance) - binding element
        concentrations (np.ndarray) protein concentrations, Nc x 2 doubles
        cut_depth (None or int) - depth of parallel evaluation of subtrees
        """
        if cut_depth is None:
            cut_depth = 0

        c_element = element.get_c_element()
        Nc = concentrations.shape[0]
        C = array('d', concentrations.T.flatten())
        return cPF(c_element, Nc, C, -10, cut_depth=cut_depth)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void evaluate_occupancies(self):
        """ Traverse tree and evaluate occupancies. """
        self.evaluate()
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

    def c_get_occupancies(self, cut_depth=None):
        """ Get Ns x b x Nc occupancy array. """

        # instantiate partition function
        c_pf = cPF.from_python(self.element, self.concentrations, cut_depth)

        # evaluate partition function and get occupancies
        c_pf.evaluate_occupancies()
        occupancies = c_pf.get_occupancies()

        # append balance
        shape = (self.element.Ns, 1, self.Nc)
        balance = (1-occupancies.sum(axis=1)).reshape(*shape)
        occupancies = np.append(balance, occupancies, axis=1)

        return occupancies
