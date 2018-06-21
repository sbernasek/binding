#from cython import parallel
from time import time
cimport numpy as np
from cpython.array cimport array, clone
from libc.stdlib cimport free


# def test_func():
#     s = time()
#     cdef int thread_id = -1
#     with nogil, parallel.parallel(num_threads=10):
#         thread_id = parallel.threadid()
#         with gil:
#             t = time()
#             print("Thread ID: {:d}, Time: {:0.2f}\n".format(thread_id, t-s))




cdef class cClass:
    cdef int Nc
    cdef double[:] C

    def __cinit__(self, double[:] arr):
        self.Nc = len(arr)
        self.C = arr

    cpdef test(self):
        cdef double[:] x = self.C

        cdef int i
        for i in xrange(self.Nc):
            x[i] = self.C[i] * 2

        print('C')
        self.print_sum(self.C, self.Nc)

        print('X')
        self.print_sum(x, self.Nc)



    cdef double print_sum(self, double[:] x, int N):
        cdef int i
        cdef double total = 0
        for i in xrange(N):
            total += x[i]
        print(total)










# cdef class Class:

#     cdef double* attribute
#     cdef int length

#     cdef Concentrations C

#     def __cinit__(self, double[:] arr):
#         self.attribute = arr.data.as_doubles
#         self.length = len(arr)

#         # store concentrations
#         self.C.length = len(arr)
#         self.C.values = arr.data.as_doubles

#     def __dealloc__(self):
#         free(<double*>self.attribute)

#     cpdef double get_value(self, int ind):
#         cdef double a
#         a = self.attribute[ind]
#         return a

#     cpdef double get_sum(self):
#         return self.c_get_sum()

#     cdef double c_get_sum(self) nogil:
#         cdef double* temp = self.attribute
#         return c_sum_array(temp, self.length)




cdef double c_sum_array(double* values, int length) nogil:
    cdef int i
    cdef double total = 0
    for i in xrange(length):
        total += values[i]
    return total




