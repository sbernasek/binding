# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=True

cimport numpy as np
from cpython.array cimport array, clone


cdef class cMicrostates:

    # attributes
    cdef unsigned int Ns
    cdef unsigned int b
    cdef unsigned int n
    cdef unsigned int Nm
    cdef array alpha
    cdef array beta
    cdef array gamma
    cdef array ets
    cdef array indices
    cdef array a
    cdef array G
    cdef array E

    # methods
    cpdef np.ndarray get_a(self)
    cpdef np.ndarray get_G(self)
    cpdef np.ndarray get_E(self)
    cpdef np.ndarray get_masks(self)
    cdef void set_params(self, dict params)
    cdef double get_binding_energy(self, unsigned int site_index, unsigned int site_state) with gil
    cdef void set_energies(self) with gil
    cdef void set_energy(self,
                          unsigned int site_index,
                          unsigned int site_state,
                          unsigned int neighbor_microstate,
                          unsigned int neighbor_state,
                          unsigned int a1, unsigned int a2, double G) with gil
    cpdef tuple get_energy_contributions(self)


cdef class cRecursiveMicrostates(cMicrostates):

    # attributes
    cdef unsigned int index

    # methods
    cdef void rset_energy(self,
                          unsigned int site_index,
                          unsigned int site_state,
                          unsigned int neighbor_state,
                          unsigned int a1, unsigned int a2, double G) with gil

