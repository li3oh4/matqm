"""matrix quantum mechanics"""

import numpy as np
import scipy.sparse as sps

from collections import OrderedDict


__all__ = ['DoF', 'QVec', 'QMat']

pauli_s0 = np.matrix([[1., 0.], [0., 1.]])
pauli_s1 = np.matrix([[0., 1.], [1., 0.]])
pauli_s2 = np.matrix([[0., -1j], [1j, 0.]])
pauli_s3 = np.matrix([[1., 0.], [0., -1.]])


def convert2odict(rawin):
    try:
        odict = OrderedDict(rawin)
    except:
        raise Exception('input cannot be converted to OrderedDict')
    else:
        for k, v in odict.items():
            if np.isscalar(v):
                odict[k] = (v,)
    return odict


def resh(a, shape):
    if np.isscalar(shape):
        shape = (shape, 1)
    elif len(shape) == 1:
        shape = tuple(shape) + (1,)
    if sps.issparse(a):
        return np.reshape(a.todense(), shape, order='F')
    else:
        return np.reshape(a, shape, order='F')


class DoF(object):
    """Degrees of Freedom.

    Attributes
    ----------
    _odict : OrderedDict
        Degrees of freedom as type: labels
    num : list of int
        number in each type

    Notes
    -----
    When initialize, argument must be convertible to an OrderedDict object.

    Examples
    --------
    >>> from matqm import DoF
    >>> dof = DoF([('s', ('u', 'd')), ('c', ('e', 'h')), ('x', range(5))])
    >>> dof.num
    [2, 2, 5]

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

    >>> dof.items()
    [('s', ('u', 'd')), ('c', ('e', 'h')), ('x', range(0, 5))]

    >>> dof.items(1)
    [('c', ('e', 'h'))]

    >>> dof.items([2, 1])
    [('x', range(0, 5)), ('c', ('e', 'h'))]

    >>> dof.items('x')
    [('x', range(0, 5))]

    >>> dof.items(('x', 's'))
    [('x', range(0, 5)), ('s', ('u', 'd'))]

    """
    def __init__(self, dof_raw):
        self._odict = convert2odict(dof_raw)
        self._update_num()

    def __repr__(self):
        return ('<DoF: {} types, {} components>'
                .format(len(self._odict), np.prod(self.num)))

    def __str__(self):
        return ('\n'.join(['{}: {}'.format(k, v)
                           for k, v in self._odict.items()]))

    def _update_num(self):
        self.num = list(map(len, self._odict.values()))

    def types(self):
        return list(self._odict.keys())

    def labels(self, types):
        if np.isscalar(types) or types == ():
            types = (types,)
        return [self._odict[t] for t in types]

    def delete_types(self, types):
        if np.isscalar(types) or types == ():
            types = (types,)
        for t in types:
            del self._odict[t]
        self._update_num()

    def reorder_types(self, types):
        if np.isscalar(types) or types == ():
            types = (types,)
        new_order = [(t, self._odict[t]) for t in types]
        self._odict = OrderedDict(new_order)
        self._update_num()  # number of types can change

    def update(self, dof_raw):
        self._odict.update(convert2odict(dof_raw))  # replace same key
        self._update_num()

    def items(self, types=None):
        if types is None:
            return list(self._odict.items())
        if np.isscalar(types) or types == ():
            types = (types,)
        if all(isinstance(t, int) for t in types):  # numerical index
            types = [list(self._odict.keys())[i] for i in types]
        return [(t, self._odict[t]) for t in types]

    def index(self, req=None):
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
        >>> ind0, ind0dof = dof.index() #  same as index({}) or ([]) or (())
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
        >>> ind1
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
        ndof = list(self.num)  # make a copy
        ind = resh(np.arange(np.prod(ndof)), ndof)
        dof = DoF(self._odict)
        if req is None:
            return ind, dof
        elif isinstance(req, (tuple, list)):
            req = {list(self._odict.keys())[i]: v
                   for i, v in enumerate(req) if v != ()}
        elif not isinstance(req, dict):
            raise Exception('invalid index request')
        for ik, k in enumerate(self._odict.keys()):
            if k in req.keys():
                vs = (req[k],) if np.isscalar(req[k]) else req[k]
                ndof[ik] = len(vs)
                ivs = [self._odict[k].index(v) for v in vs]
                ind = resh(np.take(ind, ivs, ik), ndof)
                dof.update({k: vs})
        return ind, dof


class QVec(object):
    """Vector with basis.

    Attributes
    ----------
    vec : np.array
        Values in the vector
    dof : DoF
        Degrees of freedom
    bas : np.matrix
        Basis

    Notes
    -----
    TODO: __repr__ and __str__, dot and kron

    Limitations: storage of sparse matrix is not supported because of 'take'
    function is not yet available -- possibly implement 'take' for sparse
    matrix to solve this issue.

    Examples
    --------
    >>> from matqm import QVec
    >>> import numpy.random as npr
    >>> npr.seed(0)
    >>> qv = QVec(npr.rand(4))
    >>> qv.vec
    array([[ 0.5488135 ],
           [ 0.71518937],
           [ 0.60276338],
           [ 0.54488318]])

    >>> qv.op_by_index(np.sqrt).vec
    array([[ 0.74081948],
           [ 0.84568869],
           [ 0.77637837],
           [ 0.73816203]])

    >>> qv.op_by_index(np.sum, [1, 2]).vec
    array([[ 1.31795274]])

    >>> qv.part([0, 3, 1]).vec
    array([[ 0.5488135 ],
           [ 0.54488318],
           [ 0.71518937]])

    >>> qv.apply(min, [0, 3, 1]).vec
    array([[ 0.54488318]])

    >>> dof = [('s', ('u', 'd')), ((), range(0, 2))]
    >>> import scipy.sparse as sps
    >>> bas = sps.csr_matrix([[ 0.6,  0. , -0.8,  0. ],
                              [ 0. ,  0. ,  0. ,  1. ],
                              [ 0.8,  0. ,  0.6,  0. ],
                              [ 0. , -1. ,  0. ,  0. ]])
    >>> qv = QVec(qv.vec, dof)
    >>> qvb = QVec(qv.vec, dof, bas)
    >>> qv.vec.T
    array([[ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318]])

    >>> qvb.vec.T
    array([[ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318]])

    >>> print(qvb.dof)
    s: ('u', 'd')
    (): range(0, 2)

    >>> qvb.bas.data
    array([ 0.6, -0.8,  1. ,  0.8,  0.6, -1. ])

    >>> qvb.bas.indices
    array([0, 2, 3, 0, 2, 1], dtype=int32)

    >>> qv.op_by_index(np.sqrt, [1, 2]).vec.T
    array([[ 0.84568869,  0.77637837]])

    >>> qvb.op_by_index(np.sqrt, [1, 2]).vec.T
    array([[ 0.84568869,  0.77637837]])

    >>> qvp = qv.part()
    >>> qvp.vec.T
    array([[ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318]])

    >>> print(qvp.dof)
    s: ('u', 'd')
    (): range(0, 2)

    >>> qvbp = qvb.part()
    >>> qvbp.vec.T
    array([[-0.1529226 ,  0.54488318,  0.80070883, -0.71518937]])

    >>> print(qvbp.dof)
    s: ('u', 'd')
    (): range(0, 2)

    >>> qvbp.bas is None
    True

    >>> qvp = qv.part([('u'), (1, 0)])
    >>> qvp.vec.T
    array([[ 0.60276338,  0.5488135 ]])

    >>> print(qvp.dof)
    s: ('u',)
    (): (1, 0)

    >>> qvbp = qvb.part([('u'), (1, 0)])
    >>> qvbp.vec.T
    array([[ 0.80070883, -0.1529226 ]])

    >>> sum2 = lambda a: np.sum(np.abs(a)**2)
    >>> qv.op_by_index(sum2).vec, qvb.op_by_index(sum2).vec
    (array([[ 1.47291346]]), array([[ 1.47291346]]))

    >>> qv.apply(sum2).vec, qvb.apply(sum2).vec
    (array([[ 1.47291346]]), array([[ 1.47291346]]))

    >>> qv.apply(sum2, 's').vec.T, qvb.apply(sum2, 's').vec.T
    (array([[ 0.81269209,  0.66022137]]), array([[ 0.320283  ,  1.15263046]]))

    >>> qvb.apply(np.abs, ((), 's')).vec.T
    array([[ 0.1529226 ,  0.80070883,  0.54488318,  0.71518937]])

    >>> qvb.apply(np.abs, ('s', ())).vec.T
    array([[ 0.1529226 ,  0.54488318,  0.80070883,  0.71518937]])

    >>> qvb.apply(np.abs).vec.T
    array([[ 0.1529226 ,  0.54488318,  0.80070883,  0.71518937]])

    >>> print(qvb.apply(np.abs).dof)
    absolute['s', ()]: range(0, 4)

    >>> print(qvb.apply(np.abs, ()).dof)
    absolute((),): range(0, 2)
    s: ('u', 'd')

    >>> print(qvb.apply(sum2).dof)
    <lambda>['s', ()]: range(0, 1)

    >>> print(qvb.apply(sum2, 's').dof)
    (): range(0, 2)

    """
    def __init__(self, vec, dof=None, bas=None):
        self.vec = resh(vec, -1)  # sparse matrix will be converted to ndarray
        if dof is None:
            self.dof = None
        elif isinstance(dof, DoF):
            self.dof = dof
        else:
            self.dof = DoF(dof)
        if dof is None or bas is None:
            self.bas = None
        elif bas.shape == (np.prod(self.dof.num), vec.size):
            self.bas = bas
        elif bas.shape == (vec.size, np.prod(self.dof.num)):
            self.bas = bas.T
        else:
            raise Exception('invalid input basis')
        #  TODO: if self.bas.H*self.bas != eye()

    def op_by_index(self, func=None, ind=None):
        """Operate directly on vec, disregarding basis."""
        if ind is None:
            vec = self.vec
        else:
            vec = self.vec.take(list(ind))
        if func is None:
            return QVec(vec)
        else:
            return QVec(func(vec))

    def part(self, req=None):
        """Take part of vec."""
        if self.dof is None:
            return self.op_by_index(None, req)
        if self.bas is None:
            vec = self.vec
        else:
            vec = self.bas.dot(self.vec)
        ind, idof = self.dof.index(req)
        return QVec(vec.take(ind), idof)

    def apply(self, func, types=None):
        """Apply func to a certain combination of dof types."""
        if self.dof is None:
            return self.op_by_index(func)
        else:
            ts = self.dof.types()
        if self.bas is None:
            vec = self.vec
        else:
            vec = self.bas.dot(self.vec)
        if types is None:  # all types are included by default
            a = func(vec)
            return QVec(a, DoF({func.__name__ + str(ts): range(a.size)}))
        if np.isscalar(types) or types == ():
            types = (types,)
        its = [ts.index(t) for t in types]
        its2 = sorted(list(set(range(len(ts))) - set(its)))
        ndof = list(np.take(self.dof.num, its2))
        a = np.apply_along_axis(func, 0, resh(
                resh(vec, self.dof.num).transpose(its + its2), [-1] + ndof))
        dof = self.dof.items(its2)
        if len(a.shape) > len(ndof):
            dof.insert(0, (func.__name__ + str(types), range(a.shape[0])))
        return QVec(a, dof)


class QMat(object):
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
