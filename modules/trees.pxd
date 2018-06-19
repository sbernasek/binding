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

    # attributes
    cdef cElement element
    cdef int cut_point
    cdef int Nc
    cdef array C
    cdef array weights
    cdef array degeneracy
    cdef array occupancies
    cdef array Z

    # methods
    cdef void initialize(self)

    cdef void traverse(self)

    cdef void update_branch(self,
                   int site,
                   int state,
                   int neighbor_state,
                   double deltaG)

    cdef void create_subtree(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG)

    cdef void update_node(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG)

    cdef void initialize_weights(self,
                    int shift) nogil

    cdef double get_free_energy(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) nogil

    cdef void update_degeneracy(self,
                    int state,
                    int shift,
                    int cshift) nogil

    cdef void update_branches(self,
                    int site,
                    int state,
                    double deltaG,
                    int shift,
                    int cshift)

    cdef void update_partition(self,
                    int site,
                    int state,
                    double deltaG,
                    int shift,
                    int cshift) nogil


cdef class cBranch(cTree):

    # attributes
    cdef int root_id
    cdef int branch_id

    # methods
    cdef void initialize_branch(self,
                    cTree tree) nogil

    cdef void update_node_nogil(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) nogil

    cdef void update_branches_nogil(self,
                    int site,
                    int state,
                    double deltaG,
                    int shift,
                    int cshift) nogil

