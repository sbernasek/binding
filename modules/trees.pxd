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
    cdef int Ns
    cdef int cut_point
    cdef int b
    cdef int n
    cdef int Nc
    cdef array C
    cdef array weights
    cdef array degeneracy
    cdef array occupancies
    cdef array Z

    cdef int flag

    # methods
    cdef void initialize(self)

    cdef void traverse(self)

    cdef void branch(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) with gil

    cdef void walk(self,
                   int site,
                   int state,
                   int neighbor_state,
                   double deltaG) with gil

    cdef void step(self,
                    int site,
                    int state,
                    int neighbor_state,
                    double deltaG) with gil


cdef class cSubTree(cTree):
    pass

