# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False


import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.array cimport array, clone
from libc.math cimport exp
from libc.stdio cimport printf
from elements cimport cElement
from parallel cimport cSubprocess


cdef class cTree:
    """
    Defines a ternary tree that may be traversed sequentially.

    Attributes:
        element (cElement instance) - binding site element of length Ns
        max_depth (int) - maximum depth of tree (Ns)
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
                 int tree_id=-1,
                 **kwargs):
        """
        Args:
        element (cElement) - binding element instance
        Nc (long) - number of unique protein concentration pairs
        C (array) - protein concentrations, flattened 2 x Nc array
        kwargs - empty container for subclassing (cython requirement)
        """

        # set tree properties
        self.element = element
        self.max_depth = self.element.Ns

        # set concentrations
        self.Nc = Nc
        self.C = C

        # set parent node properties
        self.tree_id = tree_id
        self.set_parent_node()

        # initialize occupancies as zero
        self.allocate_root_memory()

    def __dealloc__(self):
        """ Deallocate memory blocks. """

        #printf("\nDeallocating leaf %d", self.tree_id)

        PyMem_Free(self.degeneracy)
        PyMem_Free(self.weights)
        PyMem_Free(self.root_weights)
        PyMem_Free(self.occupancies)
        PyMem_Free(self.Z)
        # print('Called DEALLOC for branch {:d}\n'.format(self.tree_id))

    def __reduce__(self):
        """ Instance reduction for pickling. """
        init_attr = (self.element, self.Nc, self.C, self.tree_id)
        buffer_attr = self.build_buffers()
        return (rebuild_tree, (init_attr, buffer_attr))

    cpdef tuple build_buffers(self):
        """ Return all attributes as serializable arrays. """

        cdef int i

        # get dimensions
        cdef int d_shape = (self.max_depth+1)*self.Nc
        cdef int w_shape = self.max_depth*self.Nc
        cdef int o_shape = self.max_depth*self.element.n*self.Nc

        # initialize buffers
        cdef array degeneracy = clone(array('d'), d_shape, False)
        cdef array weights = clone(array('d'), w_shape, False)
        cdef array root_weights = clone(array('d'), self.Nc, False)
        cdef array occupancies = clone(array('d'), o_shape, False)
        cdef array Z = clone(array('d'), self.Nc, False)

        # populate buffers
        for i in xrange(self.Nc):
            root_weights[i] = self.root_weights[i]
            Z[i] = self.Z[i]

        for i in xrange(d_shape):
            degeneracy[i] = self.degeneracy[i]

        for i in xrange(w_shape):
            weights[i] = self.weights[i]

        for i in xrange(o_shape):
            occupancies[i] = self.occupancies[i]

        return (degeneracy, weights, root_weights, occupancies, Z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void print_array(self):
        """ Print internal attribute array by copying C pointer. """
        cdef int i
        cdef int shape = self.Nc
        cdef array x = array('d', shape*[0])
        for i in xrange(shape):
            x[i] = self.Z[i]
        print(x)

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
    cdef void set_parent_node(self,
                    int parent_branch=0,
                    double parent_deltaG=0) nogil:
        """
        Set parent node properties.

        Args:
        parent_branch (long) - state of parent node
        parent_deltaG (double) - free energy of parent node
        """
        self.parent_branch = parent_branch
        self.parent_deltaG = parent_deltaG

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void traverse_tree(self) nogil:
        """ Traverse all nodes. """
        cdef int i, branch

        # iterate across branches
        for branch in xrange(self.element.b):
            self.update_node(0, branch, self.parent_branch, self.parent_deltaG)

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
        If tree has not reached target depth, evaluate all branches.

        Args:
        depth (long) - depth of current node
        branch (long) - state of current node
        deltaG (double) - free energy of current node
        shift (long) - index for current node
        cshift (long) - index for child node
        """
        if depth < (self.max_depth - 1):
            self.evaluate_branches(depth, branch, deltaG, shift, cshift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void evaluate_branches(self,
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
        for child_branch in xrange(self.element.b):
            self.update_node(depth+1, child_branch, branch, deltaG)
            self.inherit_branch_weights(shift, cshift)


cdef class cRoot(cTree):
    """
    Extension of cTree allowing parallel traversal of subtrees (leaves).

    Attributes:
        cut_depth (int) - depth at which parallel evaluation of subtrees occurs
        num_leaves (int) - number of leaves to be run in parallel
        leaf_index (int) - counter for tracking leaves
        oshift (int) - index by which leaf occupancies are shifted
    """

    def __cinit__(self, *args, int cut_depth=0):
        """
        Args:
        element (cElement) - binding element instance
        Nc (long) - number of unique protein concentration pairs
        C (array) - protein concentrations, flattened 2 x Nc array
        cut_depth (int) - depth at which parallel evaluation of leaves occurs
        """

        # set parallel branching properties (cTree init called automatically)
        self.cut_depth = cut_depth
        if cut_depth >= self.max_depth:
            self.num_leaves = 0
        else:
            self.num_leaves = self.element.b ** cut_depth
        self.allocate_leaf_memory()
        self.leaf_index = 0

        self.oshift = self.cut_depth*self.element.n*self.Nc

        # initialize list of leaves
        self.leaves = []

    def __dealloc__(self):
        """ Deallocate memory blocks. """
        PyMem_Free(self.leaf_weights)

    cdef void allocate_leaf_memory(self):
        """ Initialize interface for branches (requires GIL) """

        # get dimensions
        cdef int w_shape = self.num_leaves*self.Nc
        cdef int i

        # allocate memory for cumulative weights of parallel branches
        self.leaf_weights=<double*> PyMem_Malloc(w_shape * sizeof(double))
        for i in xrange(w_shape):
            self.leaf_weights[i] = 0
        if not self.leaf_weights:
            raise MemoryError('Branch weights memory block not allocated.')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_branches(self,
                    int depth,
                    int branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil:
        """
        Inherit leaves from below cut depth then update branches above.

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
                self.get_leaf_weights(shift, self.leaf_index)
                self.leaf_index += 1

            # otherwise, traverse subsequent branches
            else:
                self.evaluate_branches(depth, branch, deltaG, shift, cshift)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_leaf(self,
                    int branch,
                    double deltaG,
                    int cshift):
        """
        Create and store independent leaf.

        Args:
        branch (long) - state of current node
        deltaG (double) - free energy of parent node
        cshift (long) - index for child node
        """
        cdef int i
        cdef cElement leaf_element
        cdef cTree leaf

        # instantiate leaf
        leaf_element = self.element.truncate(self.cut_depth)
        leaf = cTree(leaf_element, self.Nc, self.C, self.leaf_index)
        for i in xrange(self.Nc):
            leaf.degeneracy[i] = self.degeneracy[cshift+i]
        leaf.set_parent_node(branch, deltaG)

        # store leaf in list
        self.leaves.append(leaf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void create_leaves(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG):
        """
        Traverse tree until cut_depth, storing deltaG/degeneracy for each leaf.

        Args:
        depth (long) - depth of current node
        branch (long) - state of current node
        parent_branch (long) - state of parent node
        deltaG (double) - free energy of parent node
        """

        # get node index shifts (* note degeneracy is shifted 1 place)
        cdef int shift = depth * self.Nc
        cdef int cshift = shift + self.Nc
        cdef int child_branch

        # get free energy and update degeneracies
        deltaG = self.update_free_energy(depth, branch, parent_branch, deltaG)
        self.update_degeneracy(branch, shift, cshift)

        # if tree has reached max depth, stop
        if depth < (self.max_depth-1):

            # if at the cut depth, instantiate and store leaf
            if depth == (self.cut_depth-1):
                self.create_leaf(branch, deltaG, cshift)
                self.leaf_index += 1

            # otherwise, traverse subsequent branches
            else:
                for child_branch in xrange(self.element.b):
                    self.create_leaves(depth+1, child_branch, branch, deltaG)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void store_leaf(self, cTree leaf) nogil:
        """ Copy results from branch. """
        cdef int i
        cdef int lshift = leaf.tree_id*self.Nc

        # store weights and increment partition function
        for i in xrange(self.Nc):
            self.leaf_weights[lshift+i] = leaf.root_weights[i]
            self.Z[i] += leaf.Z[i]

        # increment occupancies
        for i in xrange(leaf.max_depth*leaf.element.n*self.Nc):
            self.occupancies[self.oshift+i] += leaf.occupancies[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void get_leaf_weights(self,
                    int shift,
                    int leaf_index) nogil:
        """
        Copy weights from current leaf.

        Args:
        shift (long) - index for current node
        leaf_index (long) - index of current leaf
        """
        cdef int i
        cdef int lshift = leaf_index*self.Nc
        for i in xrange(self.Nc):
            self.weights[shift+i] += self.leaf_weights[lshift+i]

    @staticmethod
    cdef void evaluate_leaf(object queue, cTree leaf):
        """
        Evaluate individual leaf and append to queue.

        Args:
        queue (Queue) - multiprocessing queue in which trees are stored
        leaf (cTree) - tree to be evaluated
        """
        leaf.traverse_tree()
        queue.put(leaf)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void evaluate_leaves(self):
        """
        Evaluate all leaves as parallel subprocesses.
        """
        cdef cTree leaf
        cdef cSubprocess sp

        # traverse each leaf (parallel subprocesses)
        sp = cSubprocess()
        _ = [sp.run(cRoot.evaluate_leaf, leaf) for leaf in self.leaves]
        _ = [self.store_leaf(leaf) for leaf in sp.gather()]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void evaluate(self):
        """ Traverse all nodes. """

        cdef int branch

        # traverse leaves below cut point in parallel
        self.leaf_index = 0
        if self.cut_depth > 0:

            # initialize leaves
            for branch in xrange(self.element.b):
                self.create_leaves(0, branch, 0, 0)

            # evaluate leaves in parallel
            self.evaluate_leaves()

        # sequentially traverse tree above cut point
        self.leaf_index = 0
        self.traverse_tree()


# standalone functions
cpdef cTree rebuild_tree(tuple init_attr, tuple buffer_attr):
    """
    Rebuild cTree instance from pickled attributes.weights

    Args:
        init_attr (tuple) - tree instantiation attributes
        buffer_attr (tuple) - tree array attributes

    Returns:
        tree (cTree instance)
    """
    cdef int i, d_shape, w_shape, o_shape
    cdef cTree tree

    # initialize tree
    tree = cTree(*init_attr)

    # get buffer dimensions
    d_shape = (tree.max_depth+1)*tree.Nc
    w_shape = tree.max_depth*tree.Nc
    o_shape = tree.max_depth*tree.element.n*tree.Nc

    # unpack buffer attributes
    (degeneracy, weights, root_weights, occupancies, Z) = buffer_attr

    # populate tree attributes from buffers
    for i in xrange(tree.Nc):
        tree.root_weights[i] = root_weights[i]
        tree.Z[i] = Z[i]

    for i in xrange(d_shape):
        tree.degeneracy[i] = degeneracy[i]

    for i in xrange(w_shape):
        tree.weights[i] = weights[i]

    for i in xrange(o_shape):
        tree.occupancies[i] = occupancies[i]

    return tree
