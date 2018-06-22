# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel import prange
from cpython.array cimport array
from libc.math cimport exp
from elements cimport cElement
from time import time


cdef class cRoot:
    """
    Memory allocation for the root of a tree without any recursion methods.

    Attributes:
        element (cElement instance) - binding site element of length Ns
        Nc (int) - number of unique protein concentration pairs
        C (double*) - protein concentrations, flattened N x Nc
        weights (double*) - boltzmann weights, flattened array
        degeneracy (double*) - degeneracy terms, flattened (Ns+1) x Nc
        occupancies (double*) - binding site occupancies, flattened Ns x N x Nc
        Z (double*) - partition function values, 1 x Nc array

    Notes:
        - all memory is allocated upon instantiation
        - degeneracy pointer is used for passing degeneracy terms down the tree
        - weights pointer is used for passing boltzmann weights up the tree
        - weights/occupancies are not normalized during recursion

    """

    def __cinit__(self,
                 cElement element,
                 int Nc,
                 array C,
                 int root = 0,
                 **kwargs):
        """
        Args:
        element (cElement) - binding element instance
        Nc (long) - number of unique protein concentration pairs
        C (array) - protein concentrations, flattened 2 x Nc array
        root (int) - root node for current branch
        kwargs - empty container for subclassing (cython requirement)
        """

        # set tree properties
        self.element = element.truncate(root)
        self.max_depth = self.element.Ns

        # set concentrations
        self.Nc = Nc
        self.C = C

        # set initial conditions
        self.root = root

        # initialize occupancies as zero
        self.allocate_root_memory()

    def __dealloc__(self):
        """ Deallocate memory blocks. """
        PyMem_Free(self.degeneracy)
        PyMem_Free(self.weights)
        PyMem_Free(self.root_weights)
        PyMem_Free(self.occupancies)
        PyMem_Free(self.Z)

    cdef void allocate_root_memory(self):
        """ Initialize all arrays (requires GIL for memory allocation) """

        # get dimensions
        cdef int d_shape = (self.max_depth+1)*self.Nc
        cdef int w_shape = self.max_depth*self.Nc
        cdef int o_shape = self.max_depth*self.element.n*self.Nc
        cdef int i

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

        # allocate memory for root weights
        self.root_weights = <double*> PyMem_Malloc(self.Nc * sizeof(double))
        for i in xrange(self.Nc):
            self.root_weights[i] = 0
        if not self.root_weights:
            raise MemoryError('Root weights memory block not allocated.')

        # allocate memory for occupancies
        self.occupancies = <double*> PyMem_Malloc(o_shape * sizeof(double))
        for i in xrange(o_shape):
            self.occupancies[i] = 0
        if not self.occupancies:
            raise MemoryError('Occupancies memory block not allocated.')

        # allocate memory for partition function
        self.Z = <double*> PyMem_Malloc(self.Nc * sizeof(double))
        for i in xrange(self.Nc):
            self.Z[i] = 0
        if not self.Z:
            raise MemoryError('Z memory block not allocated.')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_node_weights(self,
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
                C = self.C.data.as_doubles[bshift+i]
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
        for i in xrange(self.Nc):
            self.weights[shift+i] += self.weights[cshift+i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_partition(self,
                    int depth,
                    int branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil:
        """
        Evaluate Boltzmann factor for current node.

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
        cdef double degeneracy

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
                self.weights[shift+i] += boltzmann

                # assign weights to current depth/branch pair
                self.occupancies[oshift+i] += self.weights[shift+i]

                # update partition function
                self.Z[i] += boltzmann


cdef class cLeaf(cRoot):
    """
    Extension of cRoot allowing sequential traversal of all nodes.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void traverse(self,
                    int parent_branch=0) nogil:
        """ Traverse all nodes. """
        cdef int i, branch

        # iterate across branches
        for branch in xrange(self.element.b):
            self.update_node(0, branch, parent_branch, self.root_deltaG)

            # increment root level weights
            for i in xrange(self.Nc):
                self.root_weights[i] += self.weights[i]

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
        self.initialize_node_weights(shift)

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


cdef class cTree(cRoot):
    """
    Extension of cRoot allowing parallel traversal of leaves.

    Attributes:
        cut_depth (int) - depth at which parallel evaluation of leaves occurs
        num_leaves (int) - number of leaves to be run in parallel
        leaf_index (int) - counter for tracking leaves
    """

    def __cinit__(self, *args, int cut_depth=0):
        """
        Args:
        element (cElement) - binding element instance
        Nc (long) - number of unique protein concentration pairs
        C (array) - protein concentrations, flattened 2 x Nc array
        root (int) - root node for tree (should be zero)
        cut_depth (int) - depth at which parallel evaluation of leaves occurs
        """

        # set parallel branching properties (cRoot init called automatically)
        self.cut_depth = cut_depth
        if cut_depth >= self.max_depth:
            self.num_leaves = 0
        else:
            self.num_leaves = self.element.b ** cut_depth
        self.allocate_leaf_memory()
        self.leaf_index = 0

    def __dealloc__(self):
        """ Deallocate memory blocks. """
        PyMem_Free(self.leaf_deltaG)
        PyMem_Free(self.leaf_parent_branch)
        PyMem_Free(self.leaf_degeneracy)
        PyMem_Free(self.leaf_weights)

    cdef void allocate_leaf_memory(self):
        """ Initialize interface for branches (requires GIL) """

        # get dimensions
        cdef int dG_shape = self.num_leaves
        cdef int p_shape = self.num_leaves
        cdef int d_shape = self.num_leaves*self.Nc
        cdef int w_shape = self.num_leaves*self.Nc
        cdef int i

        # allocate memory for binding energies passed to each parallel branch
        self.leaf_deltaG=<double*> PyMem_Malloc(dG_shape * sizeof(double))
        for i in xrange(dG_shape):
            self.leaf_deltaG[i] = 0
        if not self.leaf_deltaG:
            raise MemoryError('Branch deltaG memory block not allocated.')

        # allocate memory for parent branch passed to each parallel branch
        self.leaf_parent_branch=<int*> PyMem_Malloc(p_shape * sizeof(int))
        for i in xrange(p_shape):
            self.leaf_parent_branch[i] = 0
        if not self.leaf_parent_branch:
            raise MemoryError('Branch parent memory block not allocated.')

        # allocate memory for degeneracy of parallel branches
        self.leaf_degeneracy=<double*> PyMem_Malloc(d_shape * sizeof(double))
        for i in xrange(d_shape):
            self.leaf_degeneracy[i] = 1
        if not self.leaf_degeneracy:
            raise MemoryError('Branch degeneracy memory block not allocated.')

        # allocate memory for cumulative weights of parallel branches
        self.leaf_weights=<double*> PyMem_Malloc(w_shape * sizeof(double))
        for i in xrange(w_shape):
            self.leaf_weights[i] = 0
        if not self.leaf_weights:
            raise MemoryError('Branch weights memory block not allocated.')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_leaf(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG) nogil:
        """
        Traverse tree until cut_depth

        Args:
        depth (long) - depth of current node
        branch (long) - state of current node
        parent_branch (long) - state of parent node
        deltaG (double) - free energy of parent node
        """

        # get node index shifts (* note degeneracy is shifted 1 place)
        cdef int shift = depth * self.Nc
        cdef int cshift = shift + self.Nc
        cdef int i, lshift, child_branch

        # get free energy and update degeneracies
        deltaG = self.update_free_energy(depth, branch, parent_branch, deltaG)
        self.update_degeneracy(branch, shift, cshift)

        # if tree has reached cut depth, update interface
        if depth < (self.max_depth-1):

            if depth == (self.cut_depth-1):

                #copy free energy and degeneracies to leaf interface
                self.leaf_deltaG[self.leaf_index] = deltaG
                self.leaf_parent_branch[self.leaf_index] = branch
                lshift = self.leaf_index*self.Nc
                for i in xrange(self.Nc):
                   self.leaf_degeneracy[lshift+i] = self.degeneracy[cshift+i]

                # update leaf index
                self.leaf_index += 1

            # otherwise, traverse subsequent branches
            else:
                for child_branch in xrange(self.element.b):
                    self.initialize_leaf(depth+1, child_branch, branch, deltaG)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void store_leaf(self, cLeaf leaf, int lshift, int oshift) nogil:
        """ Copy results from branch. """
        cdef int i

        # store weights and increment partition function
        for i in xrange(self.Nc):
            self.leaf_weights[lshift+i] = leaf.root_weights[i]
            self.Z[i] += leaf.Z[i]

        # increment occupancies
        for i in xrange(leaf.max_depth*leaf.element.n*self.Nc):
            self.occupancies[oshift+i] += leaf.occupancies[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void evaluate_leaves(self):
        """ Evaluate all independent leaves in parallel. """
        cdef int i, leaf_id, lshift
        cdef int parent_branch
        cdef int oshift = self.cut_depth*self.element.n*self.Nc
        cdef cLeaf leaf

        cdef double start

        # iterate across independent trees (parallelization occurs here)
        #for branch_id in prange(N, nogil=True, num_threads=N):
        for leaf_id in xrange(self.num_leaves):

            # instantiate branch (requires GIL)
            #with gil:
            start = time()
            leaf = cLeaf(self.element, self.Nc, self.C, self.cut_depth)

            # set root free energy and degeneracy for leaf
            leaf.root_deltaG = self.leaf_deltaG[leaf_id]
            lshift = leaf_id*self.Nc
            for i in xrange(self.Nc):
                leaf.degeneracy[i] = self.leaf_degeneracy[lshift+i]

            # traverse tree and copy results
            parent_branch = self.leaf_parent_branch[leaf_id]
            leaf.traverse(parent_branch=parent_branch)
            self.store_leaf(leaf, lshift, oshift)

            #nogil:
            #print('Completed branch {:d} in {:0.1f} s\n'.format(leaf_id, time()-start))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void traverse(self):
        """ Traverse all nodes. """

        cdef int branch

        # traverse leaves below cut point in parallel
        self.leaf_index = 0
        if self.cut_depth > 0:

            # initialize leaves
            for branch in xrange(self.element.b):
                self.initialize_leaf(0, branch, 0, 0)

            # traverse leaves (parallelized)
            self.evaluate_leaves()

        # sequentially traverse tree above cut point
        self.leaf_index = 0
        for branch in xrange(self.element.b):
            self.update_node(0, branch, 0, 0)

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
        self.initialize_node_weights(shift)

        # get free energy and update degeneracies
        deltaG = self.update_free_energy(depth, branch, parent_branch, deltaG)
        self.update_degeneracy(branch, shift, cshift)

        # update all branches
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
        Update branches below current depth.

        Args:
        depth (long) - depth of current node
        branch (long) - state of current node
        deltaG (double) - free energy of current node
        shift (long) - index for current node
        cshift (long) - index for child node
        """

        cdef int child_branch
        cdef int i, lshift

        # if tree has not reached target depth, update all branches
        if depth < (self.max_depth - 1):

            # if at the cut depth, copy weights from leaf interface
            if depth == (self.cut_depth - 1):
                lshift = self.leaf_index*self.Nc
                for i in xrange(self.Nc):
                    self.weights[shift+i] += self.leaf_weights[lshift+i]
                self.leaf_index += 1

            # otherwise, traverse subsequent branches
            else:
                for child_branch in xrange(self.element.b):
                    self.update_node(depth+1, child_branch, branch, deltaG)
                    self.inherit_branch_weights(shift, cshift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void print_array(self):
        """ Print internal attribute array by copying C pointer. """
        cdef int i
        cdef int shape = self.Nc
        cdef array x = array('d', shape*[0])
        for i in xrange(shape):
            x[i] = self.Z[i]
        print(x)

