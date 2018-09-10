# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

from multiprocessing import Queue, Process


cdef class cSubprocess:
    """ Interface to multiprocessing module for generating subprocesses. """

    # attributes
    cdef list processes
    cdef object queue

    # methods
    cdef void run(self,
                  func,
                  args)

    cdef list gather(self)
