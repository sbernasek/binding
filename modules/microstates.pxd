# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

cimport numpy as np
from cpython.array cimport array, clone


cdef class cMicrostates:

    # attributes
    cdef double R
    cdef double T
    cdef unsigned int b
    cdef unsigned int n
    cdef unsigned int Ns
    cdef unsigned int Nm
    cdef array alpha
    cdef array beta
    cdef array gamma
    cdef array ets
    cdef array E

    # methods
    cpdef np.ndarray get_E(self)
    cdef void set_params(self, dict params)
    cdef void reset(self)
    cdef void set_energies(self) nogil
    cdef double get_binding_energy(self, unsigned int site_index, unsigned int site_state) nogil


cdef class cRecursiveMicrostates(cMicrostates):

    # attributes
    cdef unsigned int index

    # methods
    cdef void set_energy(self,
                          unsigned int site_index,
                          unsigned int site_state,
                          unsigned int neighbor_state,
                          double E) nogil

cdef class cIterativeMicrostates(cMicrostates):

    # attributes
    cdef array a

    # methods
    cpdef np.ndarray get_a(self)
    cdef void set_energy(self,
                          unsigned int site_index,
                          unsigned int site_state,
                          unsigned int neighbor_microstate,
                          unsigned int neighbor_state,
                          unsigned int a1, unsigned int a2, double E) nogil








