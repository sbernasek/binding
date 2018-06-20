from cython import parallel
from time import time

def test_func():
    s = time()
    cdef int thread_id = -1
    with nogil, parallel.parallel(num_threads=10):
        thread_id = parallel.threadid()
        with gil:
            t = time()
            print("Thread ID: {:d}, Time: {:0.2f}\n".format(thread_id, t-s))

