__author__ = 'Sebastian Bernasek'

# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

from multiprocessing import Queue, Process
from parallel cimport cSubprocess


cdef class cSubprocess:
    """ Interface to multiprocessing module for generating subprocesses. """

    def __init__(self):
        """
        Instantiate queue of subprocesses.
        """
        self.processes = []
        self.queue = Queue()

    cdef void run(self, func, args):
        """ Run individual subprocess. """
        p = Process(target=func, args=[self.queue, args])
        self.processes.append(p)
        p.start()

    cdef list gather(self):
        """ Return unordered outputs after all subprocesses are complete. """

        # compile results
        cdef list results = [self.queue.get() for p in self.processes]

        # wait until all subprocesses run to completion
        _ = [p.join() for p in self.processes]

        return results

