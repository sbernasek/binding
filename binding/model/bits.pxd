cimport numpy as np
from cpython.array cimport array, clone


cdef inline unsigned int get_ternary_dim(unsigned int x):
    """ Get highest dimension of ternary representation (python interface). """
    return c_get_ternary_dim(x)

cdef inline unsigned int c_get_ternary_dim(unsigned int x) nogil:
    """ Get highest dimension of ternary representation (cython only). """
    cdef unsigned int n = 0
    while x // (3**(n+1)) > 0:
        n += 1
    return n

cdef inline tuple get_ternary_repr(unsigned int x):
    """ Gets ternary representation of string (python interface). """
    cdef int n = <int>c_get_ternary_dim(x)
    cdef array bits = clone(array('i'), n+1, False)
    c_set_ternary_bits(x, n, bits)
    return (n, bits)

cdef inline void c_set_ternary_bits(unsigned int x,
                                    int n,
                                    array bits) nogil:
    """ Sets ternary bit values. """
    cdef unsigned int base, num

    # add digits
    while n >= 0:
        base = 3**n
        num = x // base
        bits.data.as_ints[n] = num
        x -= num*base
        n -= 1

cdef inline unsigned int c_bits_to_int(array bits,
                                       unsigned int n,
                                       unsigned int base) nogil:
    """ Converts bits to integer value (cython only). """
    cdef unsigned int index
    cdef unsigned int k = 0
    for index in xrange(n):
        k += bits.data.as_uints[index] * (base**index)
    return k

cdef inline unsigned int bits_to_int(array bits,
                                     unsigned int n,
                                     unsigned int base):
    """ Converts bits to integer value (python interface). """
    return c_bits_to_int(bits, n, base)
