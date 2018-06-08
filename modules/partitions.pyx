# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False

import cython
import numpy as np
import math
cimport numpy as np
from array import array
from cpython.array cimport array, clone


cdef class cMicrostates:
    cdef unsigned int Ns
    cdef unsigned int b
    cdef unsigned int Nm

    cdef array alpha
    cdef array beta
    cdef array gamma

    cdef array ets
    cdef array indices
    cdef array a
    cdef array G
    cdef array E

    def __init__(self, unsigned int Ns,
                 unsigned int N_species,
                 dict params,
                 array ets,
                 double R = 1.987204118*1E-3, double T = 300):

        cdef unsigned int index
        cdef double energy

        # set system size
        self.Ns = Ns
        self.b = N_species + 1
        self.Nm = self.b**Ns

        # set parameters
        self.alpha = clone(array('d'), N_species, False)
        self.beta = clone(array('d'), N_species, False)
        self.gamma = clone(array('d'), N_species, False)
        self.set_params(params)

        # define ETS sites
        self.ets = ets

        # initialize occupancy counts and microstate energies
        self.a = clone(array('I'), self.Nm * N_species, True)
        self.G = clone(array('d'), self.Nm, True)

        # enumerate microstates and evaluate partition functions
        self.set_ground_states()
        self.set_microstate_energies()

        self.E = clone(array('d'), self.Nm, True)
        for index in xrange(self.Nm):
            energy = math.exp(-self.G.data.as_doubles[index] / (R*T))
            self.E.data.as_doubles[index] = energy

    cpdef np.ndarray get_a(self):
        """ Returns microstate TF occupancy counts as np array. """
        return np.array(self.a, dtype=np.int64).reshape((self.b-1, self.Nm))

    cpdef np.ndarray get_G(self):
        """ Returns microstate deltaG as np array. """
        return np.array(self.G, dtype=np.float64).reshape(self.Nm)

    cpdef np.ndarray get_E(self):
        """ Returns microstate energies as np array. """
        return np.array(self.E, dtype=np.float64).reshape(self.Nm)

    cdef void set_params(self, dict params):
        cdef unsigned int index
        for index in xrange(self.b-1):
            self.alpha.data.as_doubles[index] = params['alpha'][index]
            self.beta.data.as_doubles[index] = params['beta'][index]
            self.gamma.data.as_doubles[index] = params['gamma'][index]

    cdef void set_ground_states(self):
        """ Initialize ground states """
        cdef unsigned int index
        cdef double energy
        self.G.data.as_doubles[0] = 0
        for index in xrange(1, self.b):

            # check if ETS site
            energy = self.get_binding_energy(0, index)

            # set binding energy
            self.G.data.as_doubles[index] = energy
            self.a.data.as_uints[(index-1)*(self.Nm) +  index] = 1

    cdef double get_binding_energy(self, unsigned int site_index, unsigned int site_state):
        """ Get binding energy for single site. """
        cdef double energy
        if self.ets.data.as_uints[site_index] == 1:
            energy = self.alpha.data.as_doubles[site_state-1]
        else:
            energy = self.beta.data.as_doubles[site_state-1]
        return energy

    cdef void set_microstate_energies(self):
        cdef unsigned int j, k
        cdef array microstate
        cdef double binding_energy
        cdef unsigned int n, site_state, neighbor

        # update higher order elements
        for k in xrange(self.b, self.Nm):

            # TODO: could transpose A for speed
            # get microstate
            n, microstate = get_ternary_repr(k) # note there are n+1 digits
            site_state = microstate.data.as_uints[n]
            binding_energy = self.get_binding_energy(n, site_state)

            # assign activities
            for j in xrange(self.b-1):
                self.a.data.as_uints[j*self.Nm + k] = microstate.count(j+1)

            # compute energy by incrementing N-1 neighbor's energy
            neighbor = c_bits_to_int(microstate, n)
            self.G.data.as_doubles[k] = self.G.data.as_doubles[neighbor] + binding_energy

            # if N-1 neighbor shares same occupant, add gamma
            if site_state == microstate.data.as_uints[n-1]:
                if site_state != 0:
                    self.G.data.as_doubles[k] += self.gamma.data.as_doubles[site_state-1]

    cpdef tuple get_energy_contributions(self):
        """ Returns a/b/g energy contributions to each microstate. """

        cdef array alpha, beta, gamma
        cdef unsigned int index
        cdef double energy
        cdef unsigned int j, k
        cdef array microstate
        cdef unsigned int n, site_state, neighbor

        alpha = clone(array('d'), self.Nm*2, True)
        beta = clone(array('d'), self.Nm*2, True)
        gamma = clone(array('d'), self.Nm*2, True)

        # SET GROUND STATES
        for index in xrange(1, self.b):
            if self.ets.data.as_uints[0] == 1:
                energy = self.alpha.data.as_doubles[index-1]
                alpha.data.as_doubles[(index-1)*self.Nm + 0] = energy
            else:
                energy = self.beta.data.as_doubles[index-1]
                beta.data.as_doubles[(index-1)*self.Nm + 0] = energy

        # SET ALL OTHER STATES
        for k in xrange(self.b, self.Nm):

            # get microstate
            n, microstate = get_ternary_repr(k) # note there are n+1 digits
            site_state = microstate.data.as_uints[n]

            # add N-1 neighbor's energy
            neighbor = c_bits_to_int(microstate, n)
            for j in xrange(self.b-1):
                alpha.data.as_doubles[j*self.Nm + k] += alpha.data.as_doubles[j*self.Nm + neighbor]
                beta.data.as_doubles[j*self.Nm + k] += beta.data.as_doubles[j*self.Nm + neighbor]
                gamma.data.as_doubles[j*self.Nm + k] += gamma.data.as_doubles[j*self.Nm + neighbor]

            # add binding energy
            if self.ets.data.as_uints[n] == 1:
                energy = self.alpha.data.as_doubles[site_state-1]
                alpha.data.as_doubles[(site_state-1)*self.Nm + k] += energy
            else:
                energy = self.beta.data.as_doubles[site_state-1]
                beta.data.as_doubles[(site_state-1)*self.Nm + k] += energy

            # if N-1 neighbor shares same occupant, add gamma
            if site_state == microstate.data.as_uints[n-1] and site_state != 0:
                gamma.data.as_doubles[(site_state-1)*self.Nm + k] += self.gamma.data.as_doubles[site_state-1]

        a = np.array(alpha, dtype=np.float64).reshape((self.b-1, self.Nm))
        b = np.array(beta, dtype=np.float64).reshape((self.b-1, self.Nm))
        g = np.array(gamma, dtype=np.float64).reshape((self.b-1, self.Nm))

        return (a, b, g)


class Microstates:
    def __init__(self, Ns, N_species=2, params=None, ets=(0,)):

        # set system size
        self.Ns = Ns
        self.b = N_species + 1
        self.Nm = self.b**Ns
        self.params = params

        # set ets sites
        site_indices = np.zeros(int(Ns), dtype=np.uint8)
        for i in ets:
            site_indices[int(i)] = 1
        self.ets = array('I', site_indices)

    def get_c_microstates(self, **kwargs):
        return cMicrostates(self.Ns, self.b-1, self.params, self.ets, **kwargs)

    def get_mask(self, v, p, indices):
        return ((indices//(self.b**p)) % self.b) == v

    def get_masks(self):
        """ Returns boolean masks in Nm x b x Ns array. """
        indices = np.arange(self.Nm)
        masks = np.zeros((self.Nm, self.b, self.Ns), dtype=bool)
        for k in range(self.Ns):
            for j in range(self.b):
                masks[:, j, k] = self.get_mask(j, k, indices)
        return masks


cdef class cPartitionFunction:
    cdef int Ns, Nc, Nm, b, n
    cdef int density
    cdef array C
    cdef array a
    cdef array m
    cdef array E

    cdef array m_ind
    cdef array b_ind
    cdef array s_ind

    def __init__(self, microstates, concentrations):
        self.Ns = microstates.Ns
        self.Nc = concentrations.size ** 2
        self.Nm = microstates.Nm
        self.b = microstates.b
        self.n = microstates.b-1
        self.density = concentrations.size

        self.C = array('d', concentrations)
        self.set_energies(microstates)
        masks = microstates.get_masks()
        m_ind, b_ind, s_ind = masks.nonzero()
        self.m_ind = array('I', m_ind)
        self.b_ind = array('I', b_ind)
        self.s_ind = array('I', s_ind)

    cpdef void set_energies(self, microstates):
        """ Set microstate energies. """
        cdef cMicrostates ms = microstates.get_c_microstates()
        self.a = ms.a
        self.E = ms.E

    cpdef array get_occupancies(self):
        """ Get flattened Nc x b x Ns occupancy array. """
        cdef array occupancies = clone(array('d'),self.Nc*self.b*self.Ns, True)
        cdef array probabilities = clone(array('d'), self.Nm, True)
        cdef array activities = clone(array('d'), self.density*self.n*self.Nm, False)
        self.c_preallocate_activities(activities)
        self.set_occupancy(activities, probabilities, occupancies)
        return occupancies

    cpdef array get_probabilities(self):
        """ Get preallocated Nc x Nm probability array. (not used) """
        cdef array[double] activities
        activities = self.preallocate_activities()
        return self.preallocate_probabilities(activities)

    cpdef array preallocate_activities(self):
        """ Get preallocated activity array. (python interface) """
        cdef array activities = clone(array('d'), self.density*self.n*self.Nm, False)
        self.c_preallocate_activities(activities)
        return activities

    cdef void c_preallocate_activities(self, array activity) nogil:
        """ Get preallocated activity array. (used) """

        cdef int i, j, k
        cdef unsigned int a
        cdef int a_row, a_col
        cdef double C

        # get weights
        for i in xrange(self.density):
            a_row = i*self.n*self.Nm
            C = self.C.data.as_doubles[i]

            for j in xrange(self.n):
                a_col = j*self.Nm

                for k in xrange(self.Nm):
                    a = self.a.data.as_uints[a_col + k]
                    activity.data.as_doubles[a_row + a_col + k] = C**a

    cdef array c_preallocate_probabilities(self, array activities):
        """ Get preallocated Nc x Nm probability array. (not used) """

        cdef int i, j, k
        cdef array p
        cdef array es = array('d', np.empty(self.Nm, dtype=np.float64))
        cdef double a0, a1
        cdef double e, total_e
        cdef int a_row0, a_row1
        cdef int p_row

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
                    e = self.E.data.as_doubles[k] * a0 * a1
                    es.data.as_doubles[k] = e
                    total_e += e

                for k in xrange(self.Nm):
                    p.data.as_doubles[p_row + k] = es.data.as_doubles[k]/total_e

        return p

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_probabilities(self, array activities, array probabilities, int ind) nogil:
        """ Preallocate microstate probabilities """

        cdef int i
        cdef int c0 = ind // self.density
        cdef int c1 = ind % self.density
        cdef int row0 = c0*self.n*self.Nm
        cdef int row1 = (c1*self.n + 1)*self.Nm
        cdef double a0, a1, energy
        cdef double total_energy = 0

        for i in xrange(self.Nm):
            a0 = activities.data.as_doubles[row0 + i]
            a1 = activities.data.as_doubles[row1 + i]
            energy = self.E.data.as_doubles[i] * a0 * a1
            probabilities.data.as_doubles[i] = energy
            total_energy += energy

        for i in xrange(self.Nm):
            probabilities.data.as_doubles[i] /= total_energy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_occupancy(self, array activities,
                                array probabilities,
                                array occupancies) nogil:
        cdef unsigned int i, j, k, l, row
        cdef double p
        cdef unsigned int size = self.Nm * self.Ns

        # iterate across concentrations
        for i in xrange(self.Nc):
            row = i*self.b*self.Ns
            self.set_probabilities(activities, probabilities, i)

            # iterate over unmasked sites
            for index in xrange(size):
                j = self.m_ind.data.as_uints[index]
                k = self.b_ind.data.as_uints[index]
                l = self.s_ind.data.as_uints[index]
                p = probabilities.data.as_doubles[j]
                occupancies.data.as_doubles[row + k*self.Ns + l] += p


class PartitionFunction:
    def __init__(self, microstates, concentrations):
        self.c_pf = cPartitionFunction(microstates, concentrations)

        # get system dimensions
        self.Nc = concentrations.size ** 2
        self.b = microstates.b
        self.Ns = microstates.Ns

        # extra for python version
        self.density = concentrations.size
        self.Nm = microstates.Nm
        self.microstates = microstates
        self.concentrations = concentrations

    def c_get_occupancies(self):
        """ Get Nc x b x Ns occupancy array. """
        shape = (self.Nc, self.b, self.Ns)
        occupancies = self.c_pf.get_occupancies()
        return np.array(occupancies, dtype=np.float64).reshape(*shape)

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




cdef void c_set_occupancies(array[double] probabilities,
                              array[long] masks,
                              array[double] occupancies,
                              int Ns, int b, int Nc, int Nm):
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


cpdef unsigned int get_ternary_dim(unsigned int x):
    """ Get highest dimension of ternary representation (python interface). """
    return c_get_ternary_dim(x)

cdef unsigned int c_get_ternary_dim(unsigned int x) nogil:
    """ Get highest dimension of ternary representation (cython only). """
    cdef unsigned int n = 0
    while x // (3**(n+1)) > 0:
        n += 1
    return n

cpdef tuple get_ternary_repr(unsigned int x):
    """ Gets ternary representation of string (python interface). """
    cdef int n = <int>c_get_ternary_dim(x)
    cdef array bits = clone(array('i'), n+1, False)
    c_set_ternary_bits(x, n, bits)
    return (n, bits)

cdef void c_set_ternary_bits(unsigned int x, int n, array bits) nogil:
    """ Sets ternary bit values. """
    cdef unsigned int base, num

    # add digits
    while n >= 0:
        base = 3**n
        num = x // base
        bits.data.as_ints[n] = num
        x -= num*base
        n -= 1

cdef unsigned int c_bits_to_int(array bits, unsigned int n, unsigned int base=3) nogil:
    """ Converts bits to integer value (cython only). """
    cdef unsigned int index
    cdef unsigned int k = 0
    for index in xrange(n):
        k += bits.data.as_uints[index] * (base**index)
    return k

cpdef unsigned int bits_to_int(array bits, unsigned int n, unsigned int base=3):
    """ Converts bits to integer value (python interface). """
    return c_bits_to_int(bits, n, base)

