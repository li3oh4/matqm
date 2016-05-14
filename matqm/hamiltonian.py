"""classes to construct Hamiltonian"""

import numpy as np
import scipy.sparse as sps
import copy

from . import DoF


__all__ = ['HamSP']


class HamSP(object):
    """A single-particle lattice Hamiltonian.

    Attributes
    ----------
    _hop : dict
        Real-space rep of Ham as {displacement_tuple : hopping_matrix,}
    rdim : int
        Real-space dimension
    dof : DoF object
        Inner degrees of freedom

    Notes
    -----
    Hamiltonian is stored in _hop as r:t pairs, corresponding to the
    c_{r}^{\dag}.t.c_{0} term.

    Important: only half of the hopping terms are stored in _hop because
    of the hermicity. Especially, on-site term is stored as its HALF.

    Limitations, TODO

    Examples
    --------
    >>> from matqm.hamiltonian import HamSP
    >>> h1 = HamSP({0: 2, 1: -1})
    >>> print(h1)
    dof types: ()
    hop vectors: (0,), (1,)

    >>> h1m, h1mdof = h1.on_lattice({'L': 5, 'pbc': True})
    >>> h1m.todense()
    matrix([[ 2., -1.,  0.,  0., -1.],
            [-1.,  2., -1.,  0.,  0.],
            [ 0., -1.,  2., -1.,  0.],
            [ 0.,  0., -1.,  2., -1.],
            [-1.,  0.,  0., -1.,  2.]])

    >>> print(h1mdof)
    (): range(0, 1)
    r1: range(0, 5)

    >>> import numpy as np
    >>> import scipy.sparse as sps
    >>> hop2 = {(0, 0): [[2, 0], [0, -2]]}
    >>> hop2[(1, 0)] = sps.eye(2)
    >>> hop2[(0, 1)] = [[0, 1j], [1j, 0]]
    >>> dof2 = DoF({'s': ('u', 'd')})
    >>> h2 = HamSP(hop2, dof2)
    >>> print(h2)
    dof types: s
    hop vectors: (0, 0), (0, 1), (1, 0)

    >>> h2.print_hop()
    (0, 0):
    [[ 2.  0.]
     [ 0. -2.]]
    (0, 1):
    [[ 0.+0.j  0.+1.j]
     [ 0.+1.j  0.+0.j]]
    (1, 0):
    [[ 1.  0.]
     [ 0.  1.]]

    >>> conf2 = [{'k': 0.1}, {'L': 3}]
    >>> h2m, h2mdof = h2.on_lattice(conf2)
    >>> np.set_printoptions(precision=2)
    >>> h2m.todense()
    matrix([[ 3.9+0.j,  0.0+0.j,  0.0+0.j,  0.0-1.j,  0.0+0.j,  0.0+0.j],
            [ 0.0+0.j, -0.1+0.j,  0.0-1.j,  0.0+0.j,  0.0+0.j,  0.0+0.j],
            [ 0.0+0.j,  0.0+1.j,  3.9+0.j,  0.0+0.j,  0.0+0.j,  0.0-1.j],
            [ 0.0+1.j,  0.0+0.j,  0.0+0.j, -0.1+0.j,  0.0-1.j,  0.0+0.j],
            [ 0.0+0.j,  0.0+0.j,  0.0+0.j,  0.0+1.j,  3.9+0.j,  0.0+0.j],
            [ 0.0+0.j,  0.0+0.j,  0.0+1.j,  0.0+0.j,  0.0+0.j, -0.1+0.j]])

    >>> print(h2mdof)
    s: ('u', 'd')
    r2: range(0, 3)

    """

    def __init__(self, hop, dof=None):
        rdim, ncom, hop2 = 0, 0, {}
        for r, t in hop.items():
            if np.isscalar(r):
                this_rdim = 1
                this_r = (r,)
                neg_r = (-r,)
            else:
                this_rdim = np.size(r)
                this_r = tuple(r)
                neg_r = tuple(-np.array(r))
            if np.isscalar(t):
                this_ncom = 1
            else:
                this_ncom = int(np.sqrt(np.prod(np.shape(t))))
            if rdim == 0:
                rdim, ncom = this_rdim, this_ncom
            elif (rdim, ncom) != (this_rdim, this_ncom):
                raise Exception('hop sizes are not consistent')
            if neg_r not in hop2 and abs(np.array(t)).sum() != 0:
                hop2[this_r] = sps.lil_matrix(t)
                if abs(np.array(this_r)).sum() == 0:
                    hop2[this_r] = hop2[this_r] * 0.5
        if dof is None:
            dof = DoF({(): range(ncom)})
        elif not isinstance(dof, DoF):
            dof = DoF(dof)
        if ncom != np.prod(dof.num):
            raise Exception('dof size is not consistent with hop matrix')
        self._hop = hop2
        self.rdim = rdim
        self.dof = dof

    def __repr__(self):
        return ('<HamSP: {} spatial dimensions, {} components>'
                .format(self.rdim, np.prod(self.dof.num)))

    def __str__(self):
        return ('dof types: {}\nhop vectors: {}'
                .format(', '.join(map(str, self.dof.types())),
                        ', '.join(map(str, sorted(self._hop.keys())))))

    def print_hop(self):
        for r, t in sorted(self._hop.items()):
            print('\033[0;36;49m' + str(r) + ':\033[0m')
            if sum(abs(np.array(r))) == 0:
                print(2*t.todense())
            else:
                print(t.todense())

    def on_lattice(self, config):
        """Return a Hamiltonian matrix on a lattice."""
        config = [config, ] if isinstance(config, dict) else config
        hop = self._hop.copy()
        dof = copy.deepcopy(self.dof)
        for c in config:
            hop_new = {}  # reduce one dimension and construct new hop
            for r, t in hop.items():
                r0 = r[0]
                r_new = r[1:]
                if 'k' in c:
                    v = np.exp(-1j*np.pi*r0*c['k'])
                elif 'L' in c:
                    L = c['L']
                    v = sps.eye(L, L, -r0)
                    if r0 != 0 and 'pbc' in c and c['pbc']:
                        v = v + sps.eye(L, L, np.sign(r0)*(L-abs(r0)))
                    dof.update({'r'+str(self.rdim-len(r_new)): range(L)})
                else:
                    raise Exception('illegal lattice configuration')
                if r_new in hop_new:
                    hop_new[r_new] = hop_new[r_new] + sps.kron(v, t)
                else:
                    hop_new[r_new] = sps.kron(v, t)
            hop = hop_new.copy()
        return hop[()] + hop[()].getH(), dof

    def at_surface(self, config):
        pass

    def reconstruct(self, config):
        pass
