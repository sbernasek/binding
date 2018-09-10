# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

# import binding element
from elements cimport cElement
from cpython.array cimport array


cdef class cTree:

    # attributes
    cdef cElement element
    cdef int max_depth
    cdef int Nc
    cdef array C
    cdef double *degeneracy
    cdef double *weights
    cdef double *root_weights
    cdef double *occupancies
    cdef double *Z

    cdef int tree_id
    cdef int parent_branch
    cdef double parent_deltaG

    # methods

    cpdef tuple build_buffers(self)

    cpdef void print_array(self)

    cdef void allocate_root_memory(self)

    cdef void set_parent_node(self,
                    int parent_branch=*,
                    double parent_deltaG=*) nogil

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

    cdef void traverse_tree(self) nogil

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

    cdef void evaluate_branches(self,
                    int depth,
                    int parent_branch,
                    double deltaG,
                    int shift,
                    int cshift) nogil


cdef class cRoot(cTree):

    # attributes
    cdef int cut_depth
    cdef int num_leaves
    cdef int leaf_index
    cdef int oshift
    cdef double *leaf_weights

    cdef list leaves

    # methods
    cdef void allocate_leaf_memory(self)

    cdef void create_leaf(self,
                    int branch,
                    double deltaG,
                    int cshift)

    cdef void create_leaves(self,
                    int depth,
                    int branch,
                    int parent_branch,
                    double deltaG)

    cdef void store_leaf(self,
                    cTree leaf) nogil

    cdef void get_leaf_weights(self,
                    int shift,
                    int leaf_index) nogil

    @staticmethod
    cdef void evaluate_leaf(object queue, cTree leaf)

    cdef void evaluate_leaves(self)

    cdef void evaluate(self)

