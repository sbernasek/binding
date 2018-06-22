# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

# import binding element
from elements cimport cElement
from cpython.array cimport array


cdef class cRoot:

    # attributes
    cdef cElement element
    cdef int max_depth
    cdef int root
    cdef double root_deltaG
    cdef int Nc

    cdef array C
    cdef double *degeneracy
    cdef double *weights
    cdef double *root_weights
    cdef double *occupancies
    cdef double *Z

    # methods
    cdef void allocate_root_memory(self)

    cdef void initialize_node_weights(self,
                    int shift) nogil

    cdef double update_free_energy(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG) nogil

    cdef void update_degeneracy(self,
                    int branch,
                    int shift,
                    int cshift) nogil

    cdef void inherit_branch_weights(self,
                    int shift,
                    int cshift) nogil

    cdef void update_partition(self,
                    int depth,
                    int branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil


cdef class cLeaf(cRoot):

    # methods

    cdef void traverse(self,
                    int parent_branch=*) nogil

    cdef void update_node(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG) nogil

    cdef void update_branches(self,
                    int depth,
                    int parent_branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil


cdef class cTree(cRoot):

    # attributes
    cdef int cut_depth
    cdef int num_leaves
    cdef int leaf_index
    cdef double *leaf_deltaG
    cdef int *leaf_parent_branch
    cdef double *leaf_degeneracy
    cdef double *leaf_weights

    # methods
    cdef void allocate_leaf_memory(self)

    cdef void traverse(self)

    cdef void initialize_leaf(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG) nogil

    cdef void evaluate_leaves(self)

    cdef void store_leaf(self,
                    cLeaf leaf,
                    int lshift,
                    int oshift) nogil

    cdef void update_node(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG) nogil

    cdef void update_branches(self,
                    int depth,
                    int branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil

    cdef void print_array(self)

