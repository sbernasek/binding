# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

# import binding element
from elements cimport cElement


cdef class cTree:

    # attributes
    cdef cElement element
    cdef int max_depth
    cdef int root
    cdef int branch
    cdef double root_deltaG
    cdef int Nc

    cdef double *C
    cdef double *weights
    cdef double *degeneracy
    cdef double *occupancies
    cdef double *Z

    # methods
    cdef void initialize(self)

    cdef void set_root(self,
                       double deltaG,
                       double[:] degeneracy) nogil

    cdef void initialize_weights(self,
                    int shift) nogil

    cdef void traverse(self) nogil

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

