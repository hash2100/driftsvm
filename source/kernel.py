#
# Copyright 2020 Honorius Galmeanu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
# of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import torch


class Kernel:
    def __init__(self):
        self._x = None

    def set_x(self, x):
        self._x = x

    def _compute(self, t=None):
        """
        Computes the kernel for x and t
        :param t: if None, t would be equal to self._x
        :return:
        """
        raise NotImplementedError("base class method is not implemented")

    def __call__(self, *args, **kwargs):
        if self._x is None:
            raise AttributeError('Kernel not initialized yet, cannot compute')

        return self._compute(*args, **kwargs)


class PolyKernel(Kernel):
    def __init__(self, **kwargs):
        super(PolyKernel, self).__init__()

    def _compute(self, t=None):
        """
        Computes < t, x.T >
        :param t:
        :return:
        """
        t = self._x if t is None else t
        return t.matmul(self._x.transpose(0, 1))


class RbfKernel(Kernel):
    def __init__(self, gamma):
        super(RbfKernel, self).__init__()
        self._gamma = gamma

    @staticmethod
    def __norm_diff_pairs(u, t):
        """
        Generates all pairs between matrices of vectors U and V and subtracts them
        Dimensions: U (m x n), T (p x n)

        Will generate [[u1-t1, u2-t1 .. um-t1], .. [u1-tp, u2-tp .. um-tp]] (p x m)
        """
        assert u.shape[1] == t.shape[1], 'matrices have to have same number of columns (features)'
        m, n = u.shape
        p, n = t.shape

        a = u.repeat(p, 1).reshape(p, m, n)
        b = t.repeat_interleave(m, axis=0).reshape(p, m, n)
        return torch.norm((a - b).double(), dim=2)

    def _compute(self, t=None):
        t = self._x if t is None else t
        t = t.unsqueeze(0) if len(t.shape) == 1 else t

        norm = RbfKernel.__norm_diff_pairs(self._x, t)
        return torch.exp(- self._gamma * (norm ** 2))
