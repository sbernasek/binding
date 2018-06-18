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
    """
    Defines a ternary tree that is traversed sequentially until a pre-defined cut point at which parallel branching occurs.

    Attributes:
    element (cElement instance) - binding site element
    cut_point (long) - binding site before parallel branching
    Nc (int) - number of unique protein concentration pairs
    C (array) - protein concentrations, flattened 2 x Nc
    weights (array) - boltzmann weights, flattened array
    degeneracy (array) - degeneracy terms, flattened (Ns+1) x Nc
    occupancies (array) - binding site occupancies, flattened Ns x N x Nc
    Z (array) - partition function values, 1 x Nc array
    shift (long) - index for current node
    cshift (long) - index for subsequent node

    Notes:
    - degeneracy array is used for passing degeneracy terms down the tree
    - weights array is used for passing boltzmann weights up the tree
    - weights/occupancies are not normalized during recursion

    """

    def __init__(self,
                 cElement element,
                 int Nc,
                 array C,
                 int cut_point):
        """
        Args:
        element (cElement) - binding element instance
        Nc (long) - number of unique protein concentration pairs
        C (array) - protein concentrations, flattened 2 x Nc array
        cut_point (long) - binding site before parallel branching
        """

        # set tree properties
        self.element = element
        self.cut_point = cut_point

        # set concentrations
        self.Nc = Nc
        self.C = C

        # initialize occupancies as zero
        self.initialize()

    cdef void initialize(self):
        """ Initialize all arrays with zeros. """

        # truncate weights + degeneracy arrays to cut point
        cdef int Ns = self.cut_point + 2
        cdef np.ndarray w = np.zeros(Ns*self.Nc, dtype=np.float64)
        cdef np.ndarray d = np.ones((Ns+1)*self.Nc, dtype=np.float64)

        self.weights = array('d', w)
        self.degeneracy = array('d', d)
        self.occupancies = clone(array('d'), self.element.Ns*self.element.n*self.Nc, True)
        self.Z = array('d', np.ones(self.Nc, dtype=np.float64))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void traverse(self):
        """ Traverse all nodes. """
        cdef int state
        for state in xrange(self.element.b):
            self.update_branch(0, state, 0, 0.)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_branch(self,
                   int site,
                   int state,
                   int neighbor_state,
                   double deltaG):
        """ Update branch of tree. """

        # instantiate and traverse branch
        if site > self.cut_point:
            self.create_subtree(site, state, neighbor_state, deltaG)
        else:
            self.update_node(site, state, neighbor_state, deltaG)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_subtree(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG):
        """ Instantiate and traverse independent branch. """

        cdef int i
        cdef int zshift = site * self.Nc
        cdef int oshift = site * self.element.n * self.Nc
        cdef double z
        cdef double occupancy

        # instantiate and traverse branch
        cdef cBranch child = cBranch(self, site, state)
        child.update_node_nogil(0, state, neighbor_state, deltaG)

        # inherit weights, occupancies, and partition function
        for i in xrange(self.Nc):

            # inherit weights (copy top level)
            self.weights.data.as_doubles[zshift+i] = child.weights.data.as_doubles[i]

            # increment partition function (add total)
            z = self.Z.data.as_doubles[i] + child.Z.data.as_doubles[i]
            self.Z.data.as_doubles[i] = z

        # update occupancies
        for i in xrange(child.element.Ns*self.element.n*self.Nc):
            occupancy = self.occupancies.data.as_doubles[oshift+i] + child.occupancies.data.as_doubles[i]
            self.occupancies.data.as_doubles[oshift+i] = occupancy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_node(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG):
        """
        Update current node.

        Args:
        site (long) - index of current node
        state (long) - index of current branch
        neighbor_state (long) - index of parent branch
        deltaG (double) - free energy of root branch
        """

        # get node index shifts (* note degeneracy is shifted 1 place)
        cdef int shift = site * self.Nc
        cdef int cshift = shift + self.Nc
        self.shift = shift
        self.cshift = cshift

        # initialize the weights for current branch
        self.initialize_weights()

        # get free energy and update degeneracies
        deltaG = self.get_free_energy(site, state, neighbor_state, deltaG)
        self.update_degeneracy(state)

        # update all branches (run recursion)
        self.update_branches(site, state, deltaG, shift, cshift)

        # update partition for current node
        self.update_partition(site, state, deltaG, shift, cshift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_weights(self) nogil:
        """
        Initialize boltzmann weights for current branch.
        """
        cdef int i
        for i in xrange(self.Nc):
            self.weights.data.as_doubles[self.shift+i] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_free_energy(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) nogil:
        """ Get microstate free energy. """
        if state != 0:
            deltaG += self.element.get_binding_energy(site, state)
            if state == neighbor_state:
                deltaG += self.element.gamma.data.as_doubles[state-1]
        return deltaG

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_degeneracy(self, int state) nogil:
        """
        Update degeneracies for current site and pass to children.

        Args:
        site (long) - index of current binding site
        state (long) - index of current branch
        """

        cdef int i
        cdef double C, d
        cdef int bshift

        # if site is occupied, increment degeneracies and pass to children
        if state != 0:
            bshift = (state-1)*self.Nc
            for i in xrange(self.Nc):
                C = self.C.data.as_doubles[bshift+i]
                d = self.degeneracy.data.as_doubles[self.shift+i] * C
                self.degeneracy.data.as_doubles[self.cshift+i] = d

        # if site is unoccupied, pass existing degeneracies to children
        else:
            for i in xrange(self.Nc):
                self.degeneracy.data.as_doubles[self.cshift+i] = self.degeneracy.data.as_doubles[self.shift+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_branches(self,
                    int site,
                    int state,
                    double deltaG,
                    int shift,
                    int cshift):
        """
        Update subsequent branches.

        Args:
        site (long) - index of current binding site
        state (long) - index of current branch
        deltaG (double) - free energy of root branch
        shift (long) -
        cshift (long) -
        """

        cdef int i, new_state
        cdef double weight

        # if tree has not reached target depth, update all branches
        if site < (self.element.Ns - 1):

            # update each branch
            for new_state in xrange(self.element.b):
                self.update_branch(site+1, new_state, state, deltaG)

                # add branch's weights to current node
                for i in xrange(self.Nc):
                    weight = self.weights.data.as_doubles[shift+i] + self.weights.data.as_doubles[cshift+i]
                    self.weights.data.as_doubles[shift+i] = weight

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_partition(self,
                    int site,
                    int state,
                    double deltaG,
                    int shift,
                    int cshift) nogil:
        """
        Update subsequent branches.

        Args:
        site (long) - index of current binding site
        state (long) - index of current branch
        deltaG (double) - free energy of root branch
        shift (long) -
        cshift (long) -
        """

        cdef int i, index
        cdef int oshift
        cdef double exponential, boltzmann
        cdef double degeneracy, weight, occupancy, z

        # if site is unoccupied, don't make any assignments
        if state != 0:

            # evaluate partition weight
            exponential = exp(deltaG)

            # get shift index for occupancy array
            oshift = (site*self.element.n*self.Nc) + ((state-1)*self.Nc)

            # update weights, occupancies, and partitions for each unique conc.
            for i in xrange(self.Nc):

                # compute boltzmann factor
                degeneracy = self.degeneracy.data.as_doubles[cshift+i]
                boltzmann = exponential * degeneracy

                # update weights
                index = shift+i
                weight = self.weights.data.as_doubles[index] + boltzmann
                self.weights.data.as_doubles[index] = weight

                # assign weights to current site/occupant pair
                index = oshift + i
                occupancy = self.occupancies.data.as_doubles[index]
                self.occupancies.data.as_doubles[index] = occupancy + weight

                # update partition function
                z = self.Z.data.as_doubles[i] + boltzmann
                self.Z.data.as_doubles[i] = z


cdef class cBranch(cTree):
    """
    Defines a ternary tree that may only be traversed sequentially.

    Addtl. attributes:
    root_id (long) - node depth before parallel branching
    branch_id (long) - index of branch
    """

    # methods
    def __init__(self,
                 cTree tree,
                 int cut_point,
                 int branch_id):
        """
        Args:
        tree (cTree) - parent tree from which branch is initiated
        cut_point (long) - node depth before parallel branching
        branch_id (long) - index of branch (not used)
        """

        cdef cElement element = tree.element.truncate(cut_point)
        cTree.__init__(self, element, tree.Nc, tree.C, element.Ns)
        self.branch_id = branch_id
        self.root_id = cut_point
        self.initialize_branch(tree)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_branch(self, cTree tree) nogil:
        """
        Initialize current branch.

        Args:
        tree (cTree) - parent tree from which branch is initiated
        """
        cdef int i
        cdef double degeneracy
        cdef int shift = self.root_id * tree.Nc

        for i in xrange(self.Nc):

            # copy root node degeneracy to top level of current branch
            degeneracy = tree.degeneracy.data.as_doubles[shift+i]
            self.degeneracy.data.as_doubles[i] = degeneracy

            # initialize partitions as zero
            self.Z.data.as_doubles[i] = 0.

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_node_nogil(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) nogil:
        """
        Update current node.

        Args:
        site (long) - index of current node
        state (long) - index of current branch
        neighbor_state (long) - index of parent branch
        deltaG (double) - free energy of root branch
        """

        # get node index shifts (* note degeneracy is shifted 1 place)
        cdef int shift = site * self.Nc
        cdef int cshift = shift + self.Nc
        self.shift = shift
        self.cshift = cshift

        # initialize the weights for current branch
        self.initialize_weights()

        # get free energy and update degeneracies
        deltaG = self.get_free_energy(site, state, neighbor_state, deltaG)
        self.update_degeneracy(state)

        # update all branches (run recursion)
        self.update_branches_nogil(site, state, deltaG, shift, cshift)

        # update partition for current node
        self.update_partition(site, state, deltaG, shift, cshift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_branches_nogil(self,
                    int site,
                    int state,
                    double deltaG,
                    int shift,
                    int cshift) nogil:
        """
        Update subsequent branches.

        Args:
        site (long) - index of current binding site
        state (long) - index of current branch
        deltaG (double) - free energy of root branch
        shift (long) -
        cshift (long) -
        """

        cdef int i, new_state

        # if tree has not reached target depth, update all branches
        if site < (self.element.Ns - 1):

            # update each branch
            for new_state in xrange(self.element.b):
                self.update_node_nogil(site+1, new_state, state, deltaG)

                # add branch's weights to current node
                for i in xrange(self.Nc):
                    weight = self.weights.data.as_doubles[shift+i] + self.weights.data.as_doubles[cshift+i]
                    self.weights.data.as_doubles[shift+i] = weight
