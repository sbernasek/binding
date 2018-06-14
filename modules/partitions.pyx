# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
import numpy as np
import math
cimport numpy as np
from array import array
from cpython.array cimport array, clone
from libc.math cimport exp

# import microstates
from microstates cimport cMS
from microstates cimport cIMS, cRMS


cdef class cPF:
    cdef int Ns
    cdef int Nc
    cdef long long Nm
    cdef int b
    cdef int n
    cdef int density
    cdef array C
    cdef array occupancies

    def __init__(self, microstates, concentrations):
        self.Ns = microstates.Ns
        self.Nc = concentrations.size ** 2
        self.Nm = microstates.Nm
        self.b = microstates.b
        self.n = microstates.b-1
        self.density = concentrations.size
        self.C = array('d', concentrations)
        self.set_microstates(microstates)

    cpdef array get_occupancies(self):
        """ Get flattened Nc x b x Ns occupancy array. """
        self.reset()
        self.set_occupancies()
        return self.occupancies

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_occupancies(self) with gil:
        pass

    cdef void set_microstates(self, microstates):
        """ Set microstate energies. """
        pass

    cdef void reset(self):
        """ Initialize probabilities and occupancies with zeros. """
        self.occupancies = clone(array('d'),self.Nc*self.n*self.Ns, True)


cdef class cRecursiveLight(cPF):
    """ Partition function in which arrays are ordered via tree traversal. """

    # attributes
    cdef int row
    cdef double Z
    cdef array concentrations
    cdef cMS ms

    def __init__(self, microstates, concentrations):
        cPF.__init__(self, microstates, concentrations)
        self.concentrations = clone(array('d'), self.n, True)

    cdef void set_microstates(self, microstates):
        """ Set microstate energies. """
        self.ms = microstates.get_c_microstates(ms_type='base')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_concentrations(self, int c_index) with gil:
        """ Set concentration vector. """
        cdef double c0, c1
        c0 = self.C.data.as_doubles[c_index // self.density]
        c1 = self.C.data.as_doubles[c_index % self.density]
        self.concentrations.data.as_doubles[0] = c0
        self.concentrations.data.as_doubles[1] = c1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_occupancies(self) with gil:
        cdef int c_index, state
        cdef double p

        # iterate across concentrations
        for c_index in xrange(self.Nc):
            self.row = c_index*self.n*self.Ns
            self.set_concentrations(c_index)

            # initialize partition function (define ground state)
            self.Z = 1.

            # run recursion
            for state in xrange(self.b):
                p = self.set_occupancy(0, state, 0, 0., 1.)

            # apply normalization
            self.normalize()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double set_occupancy(self,
                            int site,
                            int state,
                            int neighbor_state,
                            double deltaG,
                            double degeneracy,) with gil:

        cdef int index
        cdef int new_state
        cdef double boltzmann_factor = 0
        cdef double p = 0
        cdef double occupancy

        # evaluate binding strength of current microstate
        if state != 0:

            # update microstate free energy
            deltaG += self.ms.get_binding_energy(site, state)
            if state == neighbor_state:
                deltaG += self.ms.gamma.data.as_doubles[state-1]

            # update microstate degeneracy
            degeneracy *= self.concentrations.data.as_doubles[state-1]
            if degeneracy == 0:
                return 0

        # recurse (evaluate derivative microstates)
        if site < (self.Ns - 1):
            for new_state in xrange(self.b):
                p += self.set_occupancy(site+1, new_state, state, deltaG, degeneracy)

        # if site is unoccupied, don't make an assignment but continue
        if state != 0:

            # evaluate (non-normalized) partition
            boltzmann_factor = exp(-deltaG/(self.ms.R*self.ms.T))*degeneracy
            p += boltzmann_factor

            # assign probabilities to current site/occupant pair
            index = self.row + (state-1)*self.Ns + site
            occupancy = self.occupancies.data.as_doubles[index]
            self.occupancies.data.as_doubles[index] = occupancy + p

            # update partition function and microstate index
            self.Z += boltzmann_factor

        return p

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void normalize(self) with gil:
        """ Normalize occupancies by partition function value. """
        cdef int i
        cdef double occupancy
        for i in xrange(self.n*self.Ns):
            occupancy = self.occupancies.data.as_doubles[self.row+i]
            self.occupancies.data.as_doubles[self.row+i] = occupancy/self.Z


cdef class cRecursivePF(cPF):
    """ Partition function in which arrays are ordered via tree traversal. """

    # attributes
    cdef array probabilities
    cdef long long index
    cdef array concentrations
    cdef cRMS ms

    def __init__(self, microstates, concentrations):
        cPF.__init__(self, microstates, concentrations)
        self.concentrations = clone(array('d'), self.n, True)
        self.index = 0

    cdef void set_microstates(self, microstates):
        """ Set microstate energies. """
        self.ms = microstates.get_c_microstates(ms_type='recursive')

    cdef void reset(self):
        """ Initialize probabilities and occupancies with zeros. """
        self.probabilities = clone(array('d'), self.Nm, True)
        self.occupancies = clone(array('d'),self.Nc*self.n*self.Ns, True)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_concentrations(self, int c_index) with gil:
        """ Set concentration vector. """
        cdef double c0, c1
        c0 = self.C.data.as_doubles[c_index // self.density]
        c1 = self.C.data.as_doubles[c_index % self.density]
        self.concentrations.data.as_doubles[0] = c0
        self.concentrations.data.as_doubles[1] = c1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_probabilities(self) with gil:
        cdef int i
        cdef int state
        cdef double Z
        cdef double boltzmann_weight

        # set ground state
        Z = self.ms.E.data.as_doubles[0]
        self.probabilities.data.as_doubles[0] = Z

        # run recursion
        self.index = 1 # skips empty state
        for state in xrange(self.b):
            Z += self.set_probability(0, state, 1)

        # apply normalization
        for i in xrange(self.Nm):
            boltzmann_weight = self.probabilities.data.as_doubles[i]
            self.probabilities.data.as_doubles[i] = (boltzmann_weight / Z)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double set_probability(self,
                          int site,
                          int state,
                          double degeneracy) with gil:

        cdef int new_state
        cdef double boltzmann_weight
        cdef double Z = 0

        # increment degeneracy
        if state != 0:
            degeneracy *= self.concentrations.data.as_doubles[state-1]

        # recurse
        if site < (self.Ns - 1):
            for new_state in xrange(self.b):
                Z += self.set_probability(site+1, new_state, degeneracy)

        # if site is unoccupied, don't make an assignment but continue
        if state != 0:

            # set (non-normalized) partition
            boltzmann_weight = self.ms.E.data.as_doubles[self.index] * degeneracy
            self.probabilities.data.as_doubles[self.index] = boltzmann_weight
            Z += boltzmann_weight

            # increment index
            self.index += 1

        return Z

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_occupancies(self) with gil:
        cdef int c_index, state, row
        cdef double p

        # iterate across concentrations
        for c_index in xrange(self.Nc):
            row = c_index*self.n*self.Ns
            self.set_concentrations(c_index)
            self.set_probabilities()

            # run recursion
            self.index = 1
            for state in xrange(self.b):
                p = self.set_occupancy(0, state, row)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double set_occupancy(self,
                          int site,
                          int state,
                          int row) with gil:
        cdef int index
        cdef int new_state
        cdef double p = 0
        cdef double occupancy

        # recurse
        if site < (self.Ns - 1):
            for new_state in xrange(self.b):
                p += self.set_occupancy(site+1, new_state, row)

        # if site is unoccupied, don't make an assignment but continue
        if state != 0:

            # add probability for current state
            p += self.probabilities.data.as_doubles[self.index]

            # assign probabilities to current site/occupant pair
            index = row + (state-1)*self.Ns + site
            occupancy = self.occupancies.data.as_doubles[index]
            self.occupancies.data.as_doubles[index] = occupancy + p

            # increment index
            self.index += 1

        return p


cdef class cIterativePF(cPF):
    """ Partition function in which arrays are ordered by bit rep. """
    cdef long long Nmasks
    cdef array probabilities
    cdef array m_ind
    cdef array b_ind
    cdef array s_ind
    cdef array a
    cdef array activities
    cdef cIMS ms

    def __init__(self, microstates, concentrations):
        cPF.__init__(self, microstates, concentrations)
        self.set_masks(microstates)

    cpdef array get_probabilities(self):
        """ Get preallocated Nc x Nm probability array. (not used) """
        cdef array[double] activities
        activities = self.preallocate_activities()
        return self.c_preallocate_probabilities(activities)

    cpdef void preallocate_activities(self):
        """ Get preallocated activity array. (python interface) """
        self.c_preallocate_activities(self.activities)

    cdef void set_microstates(self, microstates):
        """ Set microstate energies. """
        self.ms = microstates.get_c_microstates(ms_type='iterative')

    cdef void reset(self):
        """ Initialize probabilities and occupancies with zeros. """
        self.activities = clone(array('d'), self.density*self.n*self.Nm, True)
        self.probabilities = clone(array('d'), self.Nm, True)
        self.occupancies = clone(array('d'), self.Nc*self.n*self.Ns, True)

    cdef void set_masks(self, microstates):
        """ Set occupancy masks. """

        # set mask dimension
        self.Nmasks = self.n*self.Ns*(self.b**(self.Ns-1))

        # set masks
        masks = microstates.get_masks()[:, 1:, :]
        m_ind, b_ind, s_ind = masks.nonzero()
        self.m_ind = array('q', m_ind)
        self.b_ind = array('q', b_ind)
        self.s_ind = array('q', s_ind)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_probabilities(self, int c_index) with gil:
        """ Preallocate microstate probabilities for given concentration. """

        cdef int c0 = c_index // self.density
        cdef int c1 = c_index % self.density
        cdef long long i
        cdef long long row0 = c0*self.n*self.Nm
        cdef long long row1 = (c1*self.n + 1)*self.Nm
        cdef double a0, a1, energy
        cdef double total_energy = 0

        for i in xrange(self.Nm):
            a0 = self.activities.data.as_doubles[row0 + i]
            a1 = self.activities.data.as_doubles[row1 + i]
            energy = self.ms.E.data.as_doubles[i] * a0 * a1
            self.probabilities.data.as_doubles[i] = energy
            total_energy += energy

        for i in xrange(self.Nm):
            energy = self.probabilities.data.as_doubles[i]
            self.probabilities.data.as_doubles[i] = energy / total_energy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_occupancies(self) with gil:
        cdef int i, row
        cdef long long index, j, k, l
        cdef double p, occupancy

        # preallocate activities
        self.c_preallocate_activities(self.activities)

        # iterate across concentrations
        for i in xrange(self.Nc):
            row = i*self.n*self.Ns
            self.set_probabilities(i)

            # iterate over unmasked sites
            for index in xrange(self.Nmasks):
                j = self.m_ind.data.as_longlongs[index]
                k = self.b_ind.data.as_longlongs[index]
                l = self.s_ind.data.as_longlongs[index]
                p = self.probabilities.data.as_doubles[j]
                occupancy = self.occupancies.data.as_doubles[row+k*self.Ns+l]
                self.occupancies.data.as_doubles[row+k*self.Ns+l] = occupancy+p

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void c_preallocate_activities(self, array activity) with gil:
        """ Get preallocated activity array. (used) """

        cdef int i, j
        cdef long long k
        cdef int a
        cdef long long a_row, a_col
        cdef double C

        # get weights
        for i in xrange(self.density):
            a_row = i*self.n*self.Nm
            C = self.C.data.as_doubles[i]

            for j in xrange(self.n):
                a_col = j*self.Nm

                for k in xrange(self.Nm):
                    a = self.ms.a.data.as_longs[a_col + k]
                    activity.data.as_doubles[a_row + a_col + k] = C**a

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array c_preallocate_probabilities(self, array activities):
        """ Get preallocated Nc x Nm probability array. (not used) """

        cdef int i, j
        cdef long long k
        cdef array p
        cdef array es = array('d', np.empty(self.Nm, dtype=np.float64))
        cdef double a0, a1
        cdef double e, total_e
        cdef long long a_row0, a_row1, p_row

        # compute probabilities
        p = array('d', np.zeros(self.Nc*self.Nm, dtype=np.float64))

        for i in xrange(self.density):
            a_row0 = i*self.n*self.Nm

            for j in xrange(self.density):
                a_row1 = j*self.n*self.Nm + self.Nm
                p_row = (i*self.density + j) * self.Nm

                total_e = 0
                for k in xrange(self.Nm):
                    a0 = activities.data.as_doubles[a_row0 + k]
                    a1 = activities.data.as_doubles[a_row1 + k]
                    e = self.ms.E.data.as_doubles[k] * a0 * a1
                    es.data.as_doubles[k] = e
                    total_e += e

                for k in xrange(self.Nm):
                    p.data.as_doubles[p_row + k] = es.data.as_doubles[k]/total_e

        return p



cdef class cTest(cRecursiveLight):
    cdef dict cache




class PartitionFunction:
    def __init__(self, microstates, concentrations):

        # get system dimensions
        self.Nc = concentrations.size ** 2
        self.b = microstates.b
        self.Ns = microstates.Ns

        # extra for python version
        self.density = concentrations.size
        self.Nm = microstates.Nm
        self.microstates = microstates
        self.concentrations = concentrations

    def c_get_occupancies(self, method='base'):
        """ Get Nc x b x Ns occupancy array. """

        # instantiate partition function
        if method == 'iterative':
            c_pf = cIterativePF(self.microstates, self.concentrations)
        elif method == 'recursive':
            c_pf = cRecursivePF(self.microstates, self.concentrations)
        elif method == 'test':
            c_pf = cTest(self.microstates, self.concentrations)
        else:
            c_pf = cRecursiveLight(self.microstates, self.concentrations)

        # convert array to ndarray
        shape = (self.Nc, self.b-1, self.Ns)
        c_occupancies = c_pf.get_occupancies()
        occupancies = np.array(c_occupancies, dtype=np.float64).reshape(*shape)

        # append balance
        balance = (1 - occupancies.sum(axis=1)).reshape(self.Nc, 1, self.Ns)
        occupancies = np.append(balance, occupancies, axis=1)

        return occupancies

    def preallocate_probabilities(self):
        """ Preallocate microstate probabilities """

        a = self.microstates.get_c_microstates().get_a()
        E = self.microstates.get_c_microstates().get_E()

        dict1, dict2 = {}, {}
        for i, C in enumerate(self.concentrations):
            dict1[i] = C ** a[0].T
            dict2[i] = C ** a[1].T

        probabilities = np.zeros((self.Nc, self.Nm))

        j = 0
        for k1 in range(self.density):
            for k2 in range(self.density):
                probabilities[j, :] = (dict1[k1] * dict2[k2])
                j += 1
        probabilities *= E
        probabilities /= probabilities.sum(axis=1).reshape(-1, 1)

        return probabilities

    def get_occupancies(self):
        """ Get equilibrium occupancies. """

        # preallocate microstate probabilities and site occupancy masks
        probabilities = self.preallocate_probabilities()

        # run computation
        p = array('d', probabilities.flatten())
        m = array('l', self.microstates.get_masks().T.flatten())
        o = array('d', np.zeros(self.Ns*self.Nc*self.b, dtype=np.float64))

        c_set_occupancies(p, m, o, self.Ns, self.b, self.Nc, self.Nm)

        return np.asarray(o,dtype=np.float64).reshape(self.Ns, self.Nc, self.b)


cdef void c_set_occupancies(array probabilities,
                              array masks,
                              array occupancies,
                              int Ns, int b, int Nc, int Nm) with gil:

    """ Pure cython implementation of set_occupancies. """
    cdef int i, j, k, index
    cdef int mask
    cdef double p

    for i in xrange(Ns): # iterate across binding sites
        for j in xrange(Nc): # iterate across concentrations
            for k in xrange(b): # iterate across binding species
                for index in xrange(Nm):
                    mask = masks.data.as_longs[i*b*Nm + k*Nm + index]
                    if mask == 1:
                        p = probabilities.data.as_doubles[j*Nm + index]
                        occupancies.data.as_doubles[i*Nc*b + j*b + k] += p
