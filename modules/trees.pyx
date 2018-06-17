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


cdef class cTree:

    def __init__(self,
                 cElement element,
                 int Nc,
                 array C,
                 int cut_point):

        # set tree properties
        self.element = element
        self.Ns = element.Ns
        self.cut_point = cut_point
        self.b = element.b
        self.n = element.n

        # set concentrations
        self.Nc = Nc
        self.C = C

        # initialize occupancies as zero
        self.initialize()

    cdef void initialize(self):

        # truncate weights + degeneracy arrays to cut point
        cdef int Ns = self.cut_point + 2

        """ Initialize all arrays with zeros. """
        cdef np.ndarray w = np.zeros(Ns*self.Nc, dtype=np.float64)
        cdef np.ndarray d = np.ones((Ns+1)*self.Nc, dtype=np.float64)

        #cdef np.ndarray w = np.zeros(self.Ns*self.Nc, dtype=np.float64)
        #cdef np.ndarray d = np.ones((self.Ns+1)*self.Nc, dtype=np.float64)
        self.weights = array('d', w)
        self.degeneracy = array('d', d)
        self.occupancies = clone(array('d'), self.Ns*self.n*self.Nc, True)
        self.Z = array('d', np.ones(self.Nc, dtype=np.float64))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void traverse(self):
        cdef int state

        # run recursion
        for state in xrange(self.b):
            self.walk(0, state, 0, 0.)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void walk(self,
                   int site,
                   int state,
                   int neighbor_state,
                   double deltaG) with gil:
        """ Visit all children. """

        # instantiate and traverse subtree
        if site > self.cut_point:
            self.branch(site, state, neighbor_state, deltaG)
        else:
            self.step(site, state, neighbor_state, deltaG)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void branch(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) with gil:
        """ Instantiate and traverse independent child tree. """

        cdef int row = site * self.Nc
        cdef int crow
        cdef int i, j, k
        cdef int index, child_index

        cdef double weight
        cdef double z
        cdef double occupancy

        # instantiate and traverse subtree
        cdef cSubTree child = cSubTree(self, site, state)
        child.walk(0, state, neighbor_state, deltaG)

        # inherit weights, occupancies, and partition function
        for c_index in xrange(self.Nc):

            # update weights (add top level)
            weight = child.weights.data.as_doubles[c_index]
            self.weights.data.as_doubles[row+c_index] = weight

            # update partition function (add total)
            z = self.Z.data.as_doubles[c_index] + child.Z.data.as_doubles[c_index]
            self.Z.data.as_doubles[c_index] = z

        # update occupancies (add all)
        for i in xrange(child.Ns):
            crow = i*self.n*self.Nc
            row = (i+site)*self.n*self.Nc
            for j in xrange(self.n):
                col = j*self.Nc
                for k in xrange(self.Nc):
                    child_index = crow+col+k
                    index = row+col+k
                    occupancy = self.occupancies.data.as_doubles[index] + child.occupancies.data.as_doubles[child_index]
                    self.occupancies.data.as_doubles[index] = occupancy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void step(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) with gil:
        """ Visit child. """

        # indices
        cdef int index
        cdef int c_index
        cdef int new_state

        # microstate energies
        cdef double boltzmann
        cdef double exponential

        # degeneracy
        cdef double C
        cdef double degeneracy

        # partition values
        cdef double weight
        cdef double z
        cdef double occupancy

        # indices (note degeneracy is shifted 1 place)
        cdef int row = site * self.Nc
        cdef int crow = row + self.Nc

        # initialize the weights for current site as zero
        for c_index in xrange(self.Nc):
            self.weights.data.as_doubles[row+c_index] = 0

        # evaluate binding strength of current microstate
        if state != 0:

            # update microstate free energy
            deltaG += self.element.get_binding_energy(site, state)
            if state == neighbor_state:
                deltaG += self.element.gamma.data.as_doubles[state-1]

            # increment degeneracy
            index = (state-1)*self.Nc
            for c_index in xrange(self.Nc):
                C = self.C.data.as_doubles[index+c_index]
                degeneracy = self.degeneracy.data.as_doubles[row+c_index] * C
                self.degeneracy.data.as_doubles[crow+c_index] = degeneracy

        else:
            # copy degeneracy
            for c_index in xrange(self.Nc):
                degeneracy = self.degeneracy.data.as_doubles[row+c_index]
                self.degeneracy.data.as_doubles[crow+c_index] = degeneracy

        # recursion (traverse child microstates)
        if site < (self.Ns - 1):

            for new_state in xrange(self.b):

                # evaluate children's weights
                self.walk(site+1, new_state, state, deltaG)

                # add children's weights to parents
                for c_index in xrange(self.Nc):
                    weight = self.weights.data.as_doubles[row+c_index] + self.weights.data.as_doubles[crow+c_index]
                    self.weights.data.as_doubles[row+c_index] = weight

        # if site is unoccupied, don't make an assignment but continue
        if state != 0:

            # evaluate partition weight
            exponential = exp(deltaG)

            for c_index in xrange(self.Nc):

                # compute boltzmann factor
                degeneracy = self.degeneracy.data.as_doubles[crow+c_index]
                boltzmann = exponential * degeneracy

                # update weights
                weight = self.weights.data.as_doubles[row+c_index] + boltzmann
                self.weights.data.as_doubles[row+c_index] = weight

                # assign weights to current site/occupant pair
                index = site*self.n*self.Nc + (state-1)*self.Nc + c_index
                occupancy = self.occupancies.data.as_doubles[index]
                self.occupancies.data.as_doubles[index] = occupancy + weight

                # update partition function
                z = self.Z.data.as_doubles[c_index] + boltzmann
                self.Z.data.as_doubles[c_index] = z


cdef class cSubTree(cTree):

    # methods
    def __init__(self,
                 cTree tree,
                 int cut_point,
                 int branch):
        cdef int c_index
        cdef int index = (branch-1)*self.Nc
        cdef int row = cut_point * tree.Nc
        cdef double degeneracy
        cdef cElement element = tree.element.truncate(cut_point)
        cTree.__init__(self, element, tree.Nc, tree.C, element.Ns)

        # copy parent degeneracy to bottom level and initialize Z
        for c_index in xrange(self.Nc):
            degeneracy = tree.degeneracy.data.as_doubles[row+c_index]
            self.degeneracy.data.as_doubles[c_index] = degeneracy
            self.Z.data.as_doubles[c_index] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void walk(self,
                   int site,
                   int state,
                   int neighbor_state,
                   double deltaG) with gil:
        """ Visit all children (cannot branch). """
        self.step(site, state, neighbor_state, deltaG)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void step(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) with gil:
        """ Visit child. """

        # indices
        cdef int index
        cdef int c_index
        cdef int new_state

        # microstate energies
        cdef double boltzmann
        cdef double exponential

        # degeneracy
        cdef double C
        cdef double degeneracy

        # partition values
        cdef double weight
        cdef double z
        cdef double occupancy

        # indices (note degeneracy is shifted 1 place)
        cdef int row = site * self.Nc
        cdef int crow = row + self.Nc

        # initialize the weights for current site as zero
        for c_index in xrange(self.Nc):
            self.weights.data.as_doubles[row+c_index] = 0

        # evaluate binding strength of current microstate
        if state != 0:

            # update microstate free energy
            deltaG += self.element.get_binding_energy(site, state)
            if state == neighbor_state:
                deltaG += self.element.gamma.data.as_doubles[state-1]

            # update degeneracy
            index = (state-1)*self.Nc
            for c_index in xrange(self.Nc):
                C = self.C.data.as_doubles[index+c_index]
                degeneracy = self.degeneracy.data.as_doubles[row+c_index] * C
                self.degeneracy.data.as_doubles[crow+c_index] = degeneracy

        else:
            # copy degeneracy
            for c_index in xrange(self.Nc):
                degeneracy = self.degeneracy.data.as_doubles[row+c_index]
                self.degeneracy.data.as_doubles[crow+c_index] = degeneracy

        # recursion (traverse child microstates)
        if site < (self.Ns - 1):

            for new_state in xrange(self.b):

                # evaluate children's weights
                self.walk(site+1, new_state, state, deltaG)

                # add children's weights to parents
                for c_index in xrange(self.Nc):
                    weight = self.weights.data.as_doubles[row+c_index] + self.weights.data.as_doubles[crow+c_index]
                    self.weights.data.as_doubles[row+c_index] = weight

        # if site is unoccupied, don't make an assignment but continue
        if state != 0:

            # evaluate partition weight
            exponential = exp(deltaG)

            for c_index in xrange(self.Nc):

                # compute boltzmann factor
                degeneracy = self.degeneracy.data.as_doubles[crow+c_index]
                boltzmann = exponential * degeneracy

                # update weights
                weight = self.weights.data.as_doubles[row+c_index] + boltzmann
                self.weights.data.as_doubles[row+c_index] = weight

                # assign weights to current site/occupant pair
                index = site*self.n*self.Nc + (state-1)*self.Nc + c_index
                occupancy = self.occupancies.data.as_doubles[index]
                self.occupancies.data.as_doubles[index] = occupancy + weight

                # update partition function
                z = self.Z.data.as_doubles[c_index] + boltzmann
                self.Z.data.as_doubles[c_index] = z






