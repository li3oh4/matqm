"""matrix quantum mechanics"""

import numpy as np

from collections import OrderedDict


__all__ = ['DoF', 'QArr', 'QMat']

pauli_s0 = np.matrix([[1., 0.], [0., 1.]])
pauli_s1 = np.matrix([[0., 1.], [1., 0.]])
pauli_s2 = np.matrix([[0., -1j], [1j, 0.]])
pauli_s3 = np.matrix([[1., 0.], [0., -1.]])


class DoF(object):
    """Degrees of Freedom.

    Attributes
    ----------
    _odict : OrderedDict
        Degrees of freedom as type: labels
    ncom : int
        Total number of components

    Notes
    -----
    When initialize, argument must be convertible to an OrderedDict object.

    Examples
    --------
    >>> from matqm import DoF
    >>> dof = DoF([('s', ('u', 'd')), ('c', ('e', 'h')), ('x', range(5))])
    >>> dof.ncom
    20

    >>> dof.types()
    ['s', 'c', 'x']

    >>> dof.labels(['x', 's'])
    [range(0, 5), ('u', 'd')]

    >>> dof.delete_types('x')
    >>> print(dof)
    s: ('u', 'd')
    c: ('e', 'h')

    >>> dof.reorder_types(['c', 's'])
    >>> print(dof)
    c: ('e', 'h')
    s: ('u', 'd')

    >>> dof.update([('x', range(5)), ('c', ('h', 'e'))])
    >>> print(dof)
    c: ('h', 'e')
    s: ('u', 'd')
    x: range(0, 5)

    """
    def __init__(self, dof_raw):
        self._odict = self._convert2odict(dof_raw)
        self._update_ncom()

    def __repr__(self):
        return ('<DoF: {} types, {} components>'
                .format(len(self._odict), self.ncom))

    def __str__(self):
        return ('\n'.join(['{}: {}'.format(k, v)
                           for k, v in self._odict.items()]))

    def _convert2odict(self, dof_raw):
        try:
            odict = OrderedDict(dof_raw)
        except:
            raise Exception('DoF cannot convert input to OrderedDict')
        else:
            for k, v in odict.items():
                if np.isscalar(v):
                    odict[k] = (v,)
        return odict

    def _update_ncom(self):
        self.ncom = np.prod(list(map(len, self._odict.values())))

    def types(self):
        return list(self._odict.keys())

    def labels(self, types):
        return [self._odict[t] for t in types]

    def delete_types(self, types):
        for t in types:
            del self._odict[t]
        self._update_ncom()

    def reorder_types(self, types):
        new_order = [(t, self._odict[t]) for t in types]
        self._odict = OrderedDict(new_order)
        self._update_ncom()  # number of types can change

    def update(self, dof_raw):
        odict = self._convert2odict(dof_raw)
        self._odict.update(odict)  # replace same key
        self._update_ncom()

    def index(self, req):
        """Return indices of components for a specific combination of dof.

        Parameters
        ----------
        req : dict or list or tuple
            Request can take any of the following form.
            dict: order and number of types can be flexible.
            tuple or list of tuples: order of types must be the same as
            in _odict; use () for unspecified types before specified ones,
            unspecified types after specified ones can be omitted.

        Returns
        ----------
        ind : ndarray
            Number and order of dimensions are the same as dof types.
        dof : DoF
            Degrees of freedom corresponding to ind.

        Notes
        -----
        Order inside each type may change if requested so!

        Examples
        --------
        >>> from matqm import DoF
        >>> dof = DoF([('s', ('u', 'd')), ('c', ('e', 'h')), ('x', range(5))])
        >>> ind0, ind0dof = dof.index(()) #  same as index({}) or index([])
        >>> ind0
        array([[[ 0,  4,  8, 12, 16],
                [ 2,  6, 10, 14, 18]],

               [[ 1,  5,  9, 13, 17],
                [ 3,  7, 11, 15, 19]]])

        >>> print(ind0dof)
        s: ('u', 'd')
        c: ('e', 'h')
        x: range(0, 5)

        >>> ind1, ind1dof = dof.index({'c': 'e', 's': 'd'})
        >>> ind0
        array([[[ 1,  5,  9, 13, 17]]])

        >>> print(ind1dof)
        s: ('d',)
        c: ('e',)
        x: range(0, 5)

        >>> ind2, ind2dof = dof.index([('u'), (), (3,1)])
        >>> ind2
        array([[[12,  4],
                [14,  6]]])

        >>> print(ind2dof)
        s: ('u',)
        c: ('e', 'h')
        x: (3, 1)

        """
        if isinstance(req, (tuple, list)):
            req = {list(self._odict.keys())[i]: v
                   for i, v in enumerate(req) if v}
        elif not isinstance(req, dict):
            raise Exception('invalid index request')
        ndof = list(map(len, self._odict.values()))
        ind = np.reshape(np.arange(self.ncom), ndof, order='F')
        dof = DoF(self._odict)
        for ik, k in enumerate(self._odict.keys()):
            if k in req.keys():
                vs = (req[k],) if np.isscalar(req[k]) else req[k]
                ndof[ik] = len(vs)
                ivs = [self._odict[k].index(v) for v in vs]
                ind = np.reshape(np.take(ind, ivs, ik), ndof, order='F')
                dof.update({k: vs})
        return ind, dof


class QArr(np.ndarray):
    """ndarray with basis.

    Attributes
    ----------
    dof : list
        Degrees of freedom, each element is a DoF object
    bas : list
        Basis, each element is a list of vectors

    Notes
    -----
    TODO: direct indexing - from collections import MutableSequence
          indirect indexing using dof labels

    Examples
    --------
    >>> from hamiltonian import QVec
    >>>

    """
    def __init__(self, vec):
        pass


class QMat(np.matrix):
    """Matrix with basis.

    Attributes
    ----------
    _mat : matrix
    dofr : DoF
        Degrees of freedom for the right basis
    dofl : DoF
        Degrees of freedom for the left basis
    basr : list of vectors
        Basis vectors on the right
    basl : list of vectors
        Basis vectors on the left

    Notes
    -----
    left basis can be different from right basis
    TODO: direct and indirect indexing

    Examples
    --------
    >>> from hamiltonian import QMat
    >>>

    """
    def __init__(self):
        pass
