# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
from cython.parallel import prange

from cpython.array cimport array, clone
from libc.math cimport exp
from elements cimport cElement

from cpython.mem cimport PyMem_Malloc, PyMem_Free











cdef class cTree:
    """
    Defines a ternary tree that is traversed sequentially.

    Attributes:
    element (cElement instance) - binding site element of length Ns
    Nc (int) - number of unique protein concentration pairs
    C (double*) - protein concentrations, flattened N x Nc
    weights (double*) - boltzmann weights, flattened array
    degeneracy (double*) - degeneracy terms, flattened (Ns+1) x Nc
    occupancies (double*) - binding site occupancies, flattened Ns x N x Nc
    Z (double*) - partition function values, 1 x Nc array

    Notes:
    - all memory blocks are allocated upon instantiation
    - degeneracy ptr is used for passing degeneracy terms down the tree
    - weights ptr is used for passing boltzmann weights up the tree
    - weights/occupancies are not normalized during recursion

    """

    def __cinit__(self,
                 cElement element,
                 int Nc,
                 array C,
                 int root = 0,
                 int branch = 0):
        """
        Args:
        element (cElement) - binding element instance
        Nc (long) - number of unique protein concentration pairs
        C (array) - protein concentrations, flattened 2 x Nc array
        root (int) - root node for current branch
        branch (int) - state of root's parent node
        """

        # set tree properties
        self.element = element.truncate(root)
        self.max_depth = self.element.Ns

        # set concentrations
        self.Nc = Nc
        self.C = C.data.as_doubles

        # set initial conditions
        self.root = root
        self.branch = branch

        # initialize occupancies as zero
        self.initialize()

    def __dealloc__(self):
        """ Deallocated memory blocks. """
        PyMem_Free(self.degeneracy)
        PyMem_Free(self.weights)
        PyMem_Free(self.occupancies)
        PyMem_Free(self.Z)

    cdef void initialize(self):
        """ Initialize all arrays (requires GIL for memory allocation) """

        cdef int i

        # get dimensions
        cdef int d_shape = (self.max_depth+1)*self.Nc
        cdef int w_shape = self.max_depth*self.Nc
        cdef int o_shape = self.max_depth*self.element.n*self.Nc
        cdef int z_shape = self.Nc

        # allocate memory for degeneracy
        self.degeneracy = <double*> PyMem_Malloc(d_shape * sizeof(double))
        for i in xrange(d_shape):
            self.degeneracy[i] = 1
        if not self.degeneracy:
            raise MemoryError('Degeneracy memory block not allocated.')

        # allocate memory for weights
        self.weights = <double*> PyMem_Malloc(w_shape * sizeof(double))
        for i in xrange(w_shape):
            self.weights[i] = 0
        if not self.weights:
            raise MemoryError('Weights memory block not allocated.')

        # allocate memory for occupancies
        self.occupancies = <double*> PyMem_Malloc(o_shape * sizeof(double))
        for i in xrange(o_shape):
            self.occupancies[i] = 0
        if not self.occupancies:
            raise MemoryError('Occupancies memory block not allocated.')

        # allocate memory for partition function
        self.Z = <double*> PyMem_Malloc(z_shape * sizeof(double))
        for i in xrange(z_shape):
            self.Z[i] = 0
        if not self.Z:
            raise MemoryError('Z memory block not allocated.')

        # allocate memory
        # self.degeneracy = array('d', d_shape*[1]).data.as_doubles
        # self.weights = clone(array('d'), w_shape, True).data.as_doubles
        # self.occupancies = clone(array('d'), o_shape, True).data.as_doubles
        # self.Z = array('d', z_shape*[1]).data.as_doubles

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_root(self,
                       double deltaG,
                       double[:] degeneracy) nogil:
        """
        Set root node properties.

        Args:
        deltaG (double) - root node free energy
        degeneracy (double[:]) - root node degeneracies
        """
        cdef int i

        # set root delta G
        self.root_deltaG = deltaG

        # copy root degeneracy to top level of degeneracy array
        for i in xrange(self.Nc):
            self.degeneracy[i] = degeneracy[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_weights(self,
                                 int shift) nogil:
        """
        Initialize boltzmann weights for current branch.

        Args:
        shift (long) - index for current node
        """
        cdef int i
        for i in xrange(self.Nc):
            self.weights[shift+i] = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void traverse(self) nogil:
        """ Traverse all nodes. """
        cdef int branch
        for branch in xrange(self.element.b):
            self.update_node(0, branch, 0, self.root_deltaG)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_node(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG) nogil:
        """
        Update current node.

        Args:
        depth (long) - depth of current node
        branch (long) - state of current node
        parent_branch (long) - state of parent node
        deltaG (double) - free energy of parent node
        """

        # get node index shifts (* note degeneracy is shifted 1 place)
        cdef int shift = depth * self.Nc
        cdef int cshift = shift + self.Nc

        # initialize the weights for current branch
        self.initialize_weights(shift)

        # get free energy and update degeneracies
        deltaG = self.update_free_energy(depth, branch, parent_branch, deltaG)
        self.update_degeneracy(branch, shift, cshift)

        # update all branches (run recursion)
        self.update_branches(depth, branch, deltaG, shift, cshift)

        # update partition for current node
        self.update_partition(depth, branch, deltaG, shift, cshift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_branches(self,
                    int depth,
                    int branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil:
        """
        Update subsequent branches.

        Args:
        depth (long) - depth of current node
        branch (long) - state of current node
        deltaG (double) - free energy of current node
        shift (long) - index for current node
        cshift (long) - index for child node
        """

        cdef int child_branch

        # if tree has not reached target depth, update all branches
        if depth < (self.max_depth - 1):
            for child_branch in xrange(self.element.b):
                self.update_node(depth+1, child_branch, branch, deltaG)
                self.inherit_branch_weights(shift, cshift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double update_free_energy(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG) nogil:
        """
        Get microstate free energy.

        Args:
        depth (long) - depth of current node
        branch (long) - state of current node
        parent_branch (long) - state of parent node
        deltaG (double) - free energy of parent node

        Returns:
        deltaG (double) - free energy of current node
        """
        if branch != 0:
            deltaG += self.element.get_binding_energy(depth, branch)
            if branch == parent_branch:
                deltaG += self.element.gamma.data.as_doubles[branch-1]
        return deltaG

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_degeneracy(self,
                    int branch,
                    int shift,
                    int cshift) nogil:
        """
        Update degeneracies for current site and pass to children.

        Args:
        branch (long) - state of current node
        shift (long) - index for current node
        cshift (long) - index for child node
        """

        cdef int i
        cdef double C, d
        cdef int bshift

        # if site is occupied, increment degeneracies and pass to children
        if branch != 0:
            bshift = (branch-1)*self.Nc
            for i in xrange(self.Nc):
                C = self.C[bshift+i]
                d = self.degeneracy[shift+i] * C
                self.degeneracy[cshift+i] = d

        # if site is unoccupied, pass existing degeneracies to children
        else:
            for i in xrange(self.Nc):
                self.degeneracy[cshift+i] = self.degeneracy[shift+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void inherit_branch_weights(self,
                                int shift,
                                int cshift) nogil:
        """
        Add branch weights to current node.

        Args:
        shift (long) - index for current node
        cshift (long) - index for child node
        """
        cdef int i
        cdef double weight
        for i in xrange(self.Nc):
            weight = self.weights[shift+i] + self.weights[cshift+i]
            self.weights[shift+i] = weight

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_partition(self,
                    int depth,
                    int branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil:
        """
        Update subsequent branches.

        Args:
        depth (long) - current node depth
        branch (long) - state of current node
        deltaG (double) - free energy of current node
        shift (long) - index for current node
        cshift (long) - index for child node
        """

        cdef int i, index
        cdef int oshift
        cdef double exponential, boltzmann
        cdef double degeneracy, weight, occupancy, z

        # if site is unoccupied, don't make any assignments
        if branch != 0:

            # evaluate partition weight
            exponential = exp(deltaG)

            # get shift index for occupancy array
            oshift = (depth*self.element.n*self.Nc) + ((branch-1)*self.Nc)

            # update weights, occupancies, and partitions for each unique conc.
            for i in xrange(self.Nc):

                # compute boltzmann factor
                degeneracy = self.degeneracy[cshift+i]
                boltzmann = exponential * degeneracy

                # update weights
                index = shift+i
                weight = self.weights[index] + boltzmann
                self.weights[index] = weight

                # assign weights to current depth/branch pair
                index = oshift + i
                occupancy = self.occupancies[index]
                self.occupancies[index] = occupancy + weight

                # update partition function
                z = self.Z[i] + boltzmann
                self.Z[i] = z




