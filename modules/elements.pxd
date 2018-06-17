# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

cimport numpy as np
from cpython.array cimport array


cdef class cElement:

    # attributes
    cdef double R
    cdef double T
    cdef int b
    cdef int n
    cdef int Ns
    cdef long long Nm
    cdef array alpha
    cdef array beta
    cdef array gamma
    cdef array ets

    # methods
    cdef void set_params(self, dict params, int scale)
    cdef double get_binding_energy(self, int site_index, int site_state) with gil
    cdef cElement truncate(self, int cut_point)


cdef class cRecursiveElement(cElement):

    # attributes
    cdef array E
    cdef long long index

    # methods
    cpdef np.ndarray get_E(self)
    cdef void reset(self)
    cdef void set_energies(self) with gil
    cdef void set_energy(self,
                          int site_index,
                          int site_state,
                          int neighbor_state,
                          double E) with gil

cdef class cIterativeElement(cElement):

    # attributes
    cdef array E
    cdef array a

    # methods
    cpdef np.ndarray get_E(self)
    cpdef np.ndarray get_a(self)
    cdef void reset(self)
    cdef void set_energies(self) with gil
    cdef void set_energy(self,
                          int site_index,
                          int site_state,
                          long long neighbor_microstate,
                          int neighbor_state,
                          int a1, int a2, double E) with gil

