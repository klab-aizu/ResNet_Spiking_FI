from libc.stdint cimport *

cdef extern:
    uint32_t add16u_1A5(uint16_t A, uint16_t B)


cpdef int add(int a,int b):
    return add16u_1A5(a,b)

