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
import numpy as np
from kernel import PolyKernel, RbfKernel
import timing as t


class SingularMatrixException(ArithmeticError):
    def __init__(self, *args, **kwargs):
        super(SingularMatrixException, self).__init__(*args, **kwargs)


class NoOppositeClassLeftException(ArithmeticError):
    def __init__(self, *args, **kwargs):
        super(NoOppositeClassLeftException, self).__init__(*args, **kwargs)


class SVMBase:
    _tol = 1e-10

    def __init__(self, kernel_class, window_size):
        self._vectors = None
        self._kernel_class = kernel_class
        self._kernel = None
        self._b = None
        self._total = None
        self._correct = None
        self._logfile = None
        self._k = None
        self._window_size = window_size
        self._len = 0

    @property
    def x(self):
        """
        X is the matrix of input vectors, m x n, m vectors of n elements each
        """
        return self._vectors.x

    @property
    def y(self):
        """
        Y is the one-column matrix of classes, m x 1
        """
        return self._vectors.y

    @property
    def l(self):
        """
        Lambdas, vector, m x 1
        """
        return self._vectors.l

    @property
    def C(self):
        """
        Regularization constant, scalar
        """
        return self._vectors.C

    @property
    def support_set(self):
        return self._vectors.support_set

    @property
    def rest_set(self):
        return self._vectors.rest_set

    @property
    def k(self):
        return self._k[:self._len, :self._len]

    def sv_len(self):
        return self._vectors.sv_len()

    def initialize(self, vectors, *args, **kwargs):
        raise NotImplementedError('Base class method not implemented')

    @staticmethod
    def approx_equals(l, x, tol=_tol):
        """
        Compute a mask of booleans
        """
        return (l - x < tol) & (l - x > -tol)

    @staticmethod
    def masked_min(x, mask):
        masked = torch.where(mask)
        x1 = x
        x1[masked] = torch.max(x) + 1e12
        res, pos = torch.min(x1, 0)
        if pos in masked[0]:
            print('WARN: minimum found among masked, maybe rest set is empty')

        return res, pos

    @staticmethod
    def masked_max(x, mask):
        masked = torch.where(mask)
        x1 = x
        x1[masked] = torch.min(x) - 1e12
        res, pos = torch.max(x1, 0)
        if pos in masked[0]:
            print('WARN: maximum found among masked, maybe rest set is empty')

        return res, pos

    @staticmethod
    def test_masked_min():
        print('=== MaskedMin test_unit() begin ===')
        x = torch.tensor([1, -2, -4, 3], dtype=torch.float)
        mask = torch.tensor([False, False, True, True])
        res, pos = SVM.masked_min(x, mask)
        assert (res, pos) == (-2, 1)
        print('=== MaskedMin test_unit() end ===')

    def not_yet_learned(self, i):
        # SV, lambda > new_C anyway (do an assert), plain increase C
        # RV, lambda = 0: plain increase C
        # EV, lambda < new_C: continue learning upwards

        # check whether it migrated into the support set
        if i in self.support_set:
            pos = self.support_set.index(i)
            assert (self.l[pos] > 0) and (self.l[pos] <= self.C[pos]), 'a support vector should have lambda in between [0, C]'
            return False

        # determine real position
        assert i in self.rest_set, 'vector not found neither in support nor in rest set'
        pos = len(self.support_set) + self.rest_set.index(i)
        h_pos = self.h(pos)

        if (h_pos > 0 or self.approx_equals(h_pos, 0)) and self.approx_equals(self.l[pos], 0) or \
           (h_pos < 0 and self.approx_equals(self.l[pos], self.C[pos])):
            return False

        return True

    def not_yet_unlearned(self, i):
        # SV, lambda still < new C: plain decrease of C, no problems
        # SV, lambda > new C: move into last position and unlearn
        # RV, lambda = 0: plainly decrease C
        # EV, lambda > new_C: move into last position and unlearn

        # an unlearned vector should not be a support vector; if it is, its lambda should be close to C
        if i in self.support_set:
            pos = self.support_set.index(i)
            if self.approx_equals(self.l[pos], self.C[pos]):
                return False

            assert self.l[pos] > self.C[pos], 'a SV should have lambda greater than its C to be unlearned'
            return True

        # determine real position
        assert i in self.rest_set, 'vector not found in rest set'
        pos = len(self.support_set) + self.rest_set.index(i)
        h_pos = self.h(pos)

        # below, for unlearning, remember we 'unlearn' a vector's lambda only to its previously-set C
        if (h_pos < 0 and self.approx_equals(self.l[pos], self.C[pos])) or \
           (h_pos > 0 or self.approx_equals(h_pos, 0)) and self.approx_equals(self.l[pos], 0):
            return False

        return True

    def get_pos(self, i):
        return self.support_set.index(i) \
            if i in self.support_set \
            else len(self.support_set) + self.rest_set.index(i)

    def detect_similar(self, q):
        sim = (q[:-1] - q[-1,:].repeat(q.shape[0] - 1).view((q.shape[0] - 1, q.shape[1]))).sum(axis=1)
        res = self.approx_equals(sim, 0.0, tol=1e-4)

        if torch.any(res):
            return True

        return False

    def init_statistics(self):
        self._total = 0
        self._correct = 0
        self._logfile = None

    def collect_statistics(self, file, v, x_set, y_set):
        pred_set = self.g(x_set)
        x, y, pred = x_set[0], y_set[0], pred_set[0]
        print(f'== vector {v} predicted {pred} real {y}')

        self._total += 1
        self._correct += int(pred > 0 and y > 0 or pred < 0 and y < 0)
        print(f'== stats so far {self._correct}/{self._total} or {self._correct/self._total*100}%')

        inst_acc = ((y_set * pred_set) > 0).sum().float() / len(y_set)
        print(f'== instant accuracy {inst_acc * 100}%')

        inst_stats = ', '.join([str(x) for x in ((y_set * pred_set) > 0).int().tolist()])
        print(f'{v}, {inst_stats}', file=file)
        file.flush()

    def update_kernel(self):
        """
        Need to be called after every update of the support vectors
        """
        self._kernel.set_x(self.x)

    def compute_w(self):
        if self._kernel_class.__name__ != PolyKernel.__name__:
            raise RuntimeError(f'compute_w() can be called only for PolyKernel, not for {self._kernel_class.__name__}')

        n = self.sv_len()
        w = (self.l.unsqueeze(1) * self.y * self.x).sum(axis=0)

        return w

    def compute_kernel(self):
        self._k = torch.zeros((self._window_size, self._window_size), dtype=self._vectors.x.dtype, device=self._vectors.x.device)
        kernel = self._kernel()
        n = kernel.shape[0]
        self._k[:n, :n] = kernel
        self._len = n

    def compute_q(self):
        """
        Computes sum_i ( y_i y_j x_i x_j ), for all vectors i in the set
        :return: Q
        """
        return self.k * self.y.matmul(self.y.transpose(0, 1))

    def g(self, x, legit=False):
        if not legit:
            raise RuntimeError('costly method called')

        kern = self._kernel(x.unsqueeze(0)).squeeze(0) if len(x.shape) == 1 else self._kernel(x)
        res = kern.matmul(self.l.unsqueeze(1) * self.y) + self._b

        # compute g() in two ways, should generate the same result
        if self._kernel_class.__name__ == PolyKernel.__name__:
            res_w = x.matmul(self.compute_w()) + self._b
            if len(x.shape) == 1:
                assert self.approx_equals(res_w, res)
            else:
                assert torch.all(self.approx_equals(res_w, res.squeeze(1)))

        return res if len(x.shape) == 1 else res.squeeze(1)

    def h(self, x, y=None, legit=False):
        """
        Computes h() for a specific given vector
        :param x: the vector, or its associated index if y is None
        :param y: vector label or None
        :return:
        """
        if y is None:
            i = x
            # res1 = self.g(self.x[i]) * self.y[i][0] - 1
            res = (self.k[i].matmul(self.l.unsqueeze(1) * self.y) + self._b) * self.y[i][0] - 1
            # assert res1 == res
            return res
        else:
            return self.g(x, legit) * y - 1

    def compute_h(self, begin=0, end=None):
        """
        Computes h() for all entries x starting from position i to position j (end)
        :param i: start position (usually first rest vector)
        :param j: end position (usually last if default value is given)
        :return:
        """
        end = self.x.shape[0] if end is None else end
        # res1 = (self._kernel(self.x[begin:end]).matmul(self.l.unsqueeze(1) * self.y) + self._b) * self.y[begin:end] - 1.0
        res = (self.k[begin:end].matmul(self.l.unsqueeze(1) * self.y) + self._b) * self.y[begin:end] - 1.0
        # assert torch.all(res1 == res)

        if self._kernel_class.__name__ == PolyKernel.__name__:
            w = self.compute_w()
            res1 = (self.x[begin:end].matmul(w.reshape(-1, 1)) + self._b) * self.y[begin:end] - 1.0
            assert torch.all(self.approx_equals(res1, res, tol=1e-5)), 'the two methods of computing h() should generate the same result'

        return res.reshape(-1)

    def _extend_kernel(self):
        line = self._kernel(self.x[-1])
        k_line = torch.zeros_like(self._k[0])
        k_line[:self._len + 1] = line
        self._k[self._len] = k_line
        self._k[:, self._len] = k_line
        self._len += 1

    def _rearange_kernel(self, index):
        self._k = self._k[index][:, index]

    def from_support(self, pos):
        index = self._vectors.from_support(pos)
        if index is not None:
            self._rearange_kernel(index)

    def into_support(self, pos_in_rest):
        index = self._vectors.into_support(pos_in_rest)
        if index is not None:
            self._rearange_kernel(index)

    def move_to_end(self, original_pos):
        index = self._vectors.move_to_end(original_pos)
        self._rearange_kernel(index)

    def append(self, pos, x, y, c_init):
        self._vectors.append(pos, x, y, c_init)
        self._kernel.set_x(self.x)
        self._extend_kernel()

    def remove(self):
        self._vectors.remove()
        self._len -= 1


class SVM(SVMBase):
    def __init__(self, kernel_class, window_size):
        super(SVM, self).__init__(kernel_class, window_size)
        self.debug = False
        self.cycles = 0

        # timers
        self.t_migrate = t.Timing()
        self.t_updatek = t.Timing()
        self.t_move = t.Timing()
        self.t_q = t.Timing()
        self.t_learn = t.Timing()
        self.t_hdr = t.Timing()
        self.t_h = t.Timing()
        self.t_loop = t.Timing()
        self.t_end = t.Timing()

    def reset_timers(self):
        self.t_migrate.reset()
        self.t_updatek.reset()
        self.t_move.reset()
        self.t_q.reset()
        self.t_learn.reset()
        self.t_hdr.reset()
        self.t_h.reset()
        self.t_loop.reset()
        self.t_end.reset()

    def print_timers(self):
        self.cycles += 1
        if self.cycles % 100 == 0:
            print(f't_migrate: {self.t_migrate.get_mean()}')
            print(f't_updatek: {self.t_updatek.get_mean()}')
            print(f't_move: {self.t_move.get_mean()}')
            print(f't_q: {self.t_q.get_mean()}')
            print(f't_learn: {self.t_learn.get_mean()}')
            print(f't_hdr: {self.t_hdr.get_mean()}')
            print(f't_h: {self.t_h.get_mean()}')
            print(f't_loop: {self.t_loop.get_mean()}')
            print(f't_end: {self.t_end.get_mean()}')

    def initialize(self, vectors, *args, **kwargs):
        """
        Initialize using first two entries in x and y
        """
        self._vectors = vectors

        assert len(self.x[0].shape) == 1, 'x0 is not a vector'
        assert len(self.x[1].shape) == 1, 'x1 is not a vector'
        assert self.x[0].shape == self.x[1].shape, 'x0 and x1 have different shapes'
        assert torch.abs(self.x[0] - self.x[1]).sum() != 0, 'x0 and x1 have to be different'
        assert torch.abs(self.y[0]) == 1, 'y0 can be only +1 or -1'
        assert torch.abs(self.y[1]) == 1, 'y1 can be only +1 or -1'
        assert self.y[0] == -self.y[1], 'y0 and y1 should have opposed signs'

        # we make the notation that the first vectors in _x would be the support vectors
        # after them are all of the rest vectors
        # the last one is the newly added vector (x_c)

        # initialize kernel by calling its constructor
        self._kernel = self._kernel_class(*args, **kwargs)
        self._kernel.set_x(self.x)

        # support vectors are the ones in x[:n_support]
        assert self.sv_len() == 2

        self.compute_kernel()

        q = self.compute_q()
        n = self.sv_len()

        # we're interested only in the first two lines and columns
        q = q[:n, :n]

        # initialize lambdas
        self.l[0] = 2 / q.sum()
        self.l[1] = 2 / q.sum()

        self._b = (q[0, 0] - q[1, 1]) * self.y[1] / q.sum()

        # check that the g(x) is exactly +1 or -1 for those two points
        assert self.g(self.x[0], legit=True).cpu() - self.y[0][0].cpu() < 1e-6, 'checking first class failed'
        assert self.g(self.x[1], legit=True).cpu() - self.y[1][0].cpu() < 1e-6, 'checking second class failed'

        # check that h(x) is zero for both vectors
        assert self.h(0).cpu() < 1e-15, 'checking first slack failed'
        assert self.h(1).cpu() < 1e-15, 'checking second slack failed'

        # check that sum(l_i y_i) = 0
        assert self.l[:n].dot(self.y[:2].view(-1)) < 1e-16

    def compute_beta(self, q):
        """
        Computation of beta, (1 + SV) x 1 column
        (inverse of a symmetric positive definite is involved, perhaps Cholesky is faster)
        :return: beta
        """
        # number of support vectors
        n = self.sv_len()

        # compute Q_ss
        q_ss = q[:n, :n]

        # concatenate columns for first n lines, and then the last line
        upper = torch.cat((self.y[:n], q_ss), 1)
        lower = torch.cat((torch.tensor([[0.0]], dtype=torch.double).to(self._vectors.x.device), self.y[:n].t()), 1)
        first = torch.cat((upper, lower), 0)

        # invert it, and multiply with column vector of [Q_sc and y_c]
        # the c vector is considered on the last line
        a = torch.inverse(first)
        b = torch.cat((q[:n, -1], self.y[-1]), 0)
        beta = - a.matmul(b)

        return beta

    def compute_gamma(self, q, beta):
        """
        Computes gamma, (#rest + 1) x 1 column vector
        :param beta: beta
        :return: gamma
        """
        # number of support vectors
        n = self.sv_len()

        # form the matrix from y_(r,c) and Q_(r,c)s
        # and compute gamma = matrix x beta + Q_(r,c)c (c is last column)
        first = torch.cat((self.y[n:], q[n:, :n]), 1)
        gamma = first.matmul(beta.reshape(-1, 1)) + q[n:, -1].reshape(-1, 1)

        return gamma.reshape(-1)

    def limits_support(self, beta, incremental=True):
        beta_s = beta[1:]
        n = self.sv_len()

        if incremental:
            limits = torch.where(beta_s >= 0.0, (self.C[:n] - self.l[:n]) / beta_s, - self.l[:n] / beta_s)
            res = torch.min(limits, 0)
        else:
            limits = torch.where(beta_s <= 0.0, (self.C[:n] - self.l[:n]) / beta_s, - self.l[:n] / beta_s)
            res = torch.max(limits, 0)

        #print(f'minimum: {res}')

        # returns the minimum delta l_c, the position in support and the corresponding beta
        return res[0], res[1], beta_s[res[1]]

    def limits_rest(self, gamma, incremental=True):
        n = self.sv_len()
        h_r = self.compute_h(begin=n)
        limits = - h_r / gamma

        # from these limits, we filter out:
        if incremental:
            # * the positive gamma for other vectors (lambda_r = 0), their h_r will increase even more, and
            # * the negative gamma for error vectors (lambda_r = C), their h_r will decrease even more
            mask = (gamma > 0) & ( (h_r > 0) | self.approx_equals(h_r, 0.0) ) | \
                   (gamma < 0) & ( (h_r < 0) | self.approx_equals(h_r, 0.0) )
        else:
            # * the negative gamma for other vectors (lambda_r = 0), their h_r will increase even more, and
            # * the positive gamma for error vectors (lambda_r = C), their h_r will decrease even more
            mask = (gamma < 0) & ( (h_r > 0) | self.approx_equals(h_r, 0.0) ) | \
                   (gamma > 0) & ( (h_r < 0) | self.approx_equals(h_r, 0.0) )

            # in addition, the last rest vector should be masked since we want it out
            mask[-1] = True

        return self.masked_min(limits, mask) if incremental else self.masked_max(limits, mask)

    def update_state(self, beta, delta_lc):
        """
        Updates lambdas and recomputes hyperplane afterwards
        :param beta:
        :param delta_lc:
        :return:
        """
        n = self.sv_len()
        update = beta * delta_lc
        #print(f'update: {update}')

        # update support lambdas
        self.l[:n] += update[1:]

        # update learned vector (last) lambda
        self.l[-1] += delta_lc

        # update plane
        self._b += update[0]

    def _learn(self):
        # compute Q
        self.t_q.start()
        q = self.compute_q()
        self.t_q.stop()

        # compute limit for lambda_c
        beta = self.compute_beta(q)
        gamma = self.compute_gamma(q, beta)
        #print(f'beta: {beta}')
        #print(f'gamma: {gamma}')
        if self.approx_equals(gamma[-1], 0, tol=1e-4):
            raise SingularMatrixException('singular matrix detected')

        # compute limits and find minimum
        delta_lc_supp, pos_supp, beta_s = self.limits_support(beta)
        delta_lc_rest, pos_rest = self.limits_rest(gamma)
        delta_lc, pos, in_support = (delta_lc_supp, pos_supp, True) if delta_lc_supp < delta_lc_rest \
                               else (delta_lc_rest, pos_rest, False)

        # limit lambda_c by C
        delta_last = self.C[-1] - self.l[-1]
        delta_lc, last_limited = (delta_lc, False) if delta_last > delta_lc else (delta_last, True)

        # update lambdas and plane
        # print(f'lambda increase: {delta_lc}')
        self.update_state(beta, delta_lc)

        # if limit is determined by last, there is no movement
        if not last_limited:
            if self.approx_equals(delta_lc_rest, delta_lc_supp):
                # move the support vector
                self._vectors.l[pos_supp] = 0.0 if beta_s < 0 else self.C[pos]
                self.t_migrate.start()
                self.from_support(pos_supp)
                self.t_migrate.stop()
                self.t_migrate.start()
                self.into_support(pos_rest)
                self.t_migrate.stop()
            elif in_support:
                # depending on sign of beta, we move it from support
                # as error vector (beta > 0) or as other vector (beta < 0)
                self._vectors.l[pos] = 0.0 if beta_s < 0 else self.C[pos]
                self.t_migrate.start()
                self.from_support(pos)
                self.t_migrate.stop()
            else:
                self.t_migrate.start()
                self.into_support(pos)
                self.t_migrate.stop()

            self.t_updatek.start()
            self.update_kernel()
            self.t_updatek.stop()

        # self.print_sets()
        # self.print_h()

    def _unlearn(self):
        if self.y.squeeze(1)[:-1].sum().abs() == len(self.y.squeeze(1)[:-1]):
            # not possible to have remaining vectors have all the same class
            raise NoOppositeClassLeftException('removing essential vector')

        # compute Q
        q = self.compute_q()

        # compute limit for lambda_c
        beta = self.compute_beta(q)
        gamma = self.compute_gamma(q, beta)
        # print(f'beta: {beta}')

        # compute limits and find minimum
        delta_lc_supp, pos_supp, beta_s = self.limits_support(beta, incremental=False)
        delta_lc_rest, pos_rest = self.limits_rest(gamma, incremental=False)
        delta_lc, pos, in_support = (delta_lc_supp, pos_supp, True) if delta_lc_supp > delta_lc_rest \
            else (delta_lc_rest, pos_rest, False)

        # limit lambda_c by C, not zero
        delta_last = self.C[-1] - self.l[-1]
        #delta_last = - self.l[-1]
        delta_lc, last_limited = (delta_lc, False) if delta_last < delta_lc else (delta_last, True)

        # update lambdas and plane
        # print(f'lambda decrease: {delta_lc}')
        self.update_state(beta, delta_lc)

        # if limit is determined by last, there is no movement:
        # although there is a special case: what if you want to unlearn a SV that stays a SV afterwards?
        # prove that if you migrate a SV with a lower C than it has already, then it becomes an error vector
        if not last_limited:
            if in_support:
                # depending on sign of beta, we move it from support
                # as error vector (beta < 0) or as other vector (beta > 0)
                self._vectors.l[pos] = 0.0 if beta_s > 0 else self.C[pos]
                self.from_support(pos)
            else:
                self.into_support(pos)
            self.update_kernel()

        # self.print_sets()
        # self.print_h()

    def _potential_migrate_rest(self):
        """
        Move rest vectors with negative gradient to support set
        :return:
        """
        # self.print_sets()
        # self.print_h()

        # if the rest vector set length is zero, there is nothing to migate
        if len(self.rest_set) == 0:
            return

        n = self.sv_len()
        h_r = self.compute_h(begin=n)

        candidates = (h_r < 0) & self.approx_equals(self.l[n:], 0.)
        if candidates is None or not torch.any(candidates):
            return

        print('>> moving negative gradient and zero lambda rest vectors')
        candidates_pos = torch.where(candidates)[0]
        for i in range(len(candidates_pos) - 1, -1, -1):
            pos = candidates_pos[i]
            self.into_support(pos)
        self.update_kernel()

        # self.print_sets()
        # self.print_h()

    def _potential_migrate_last(self):
        # self.print_sets()
        # self.print_h()

        # if the rest vector set length is zero, there is nothing to migate
        if len(self.rest_set) == 0:
            return

        # if the gradient is zero, put it into support; otherwise, gradient should be negative
        # check whether it migrated into the support set
        pos = len(self.support_set) + len(self.rest_set) - 1
        h_pos = self.h(pos)

        if (h_pos < 0) and self.approx_equals(self.l[pos], self.C[pos]) or \
           (h_pos > 0 or self.approx_equals(h_pos, 0)) and self.approx_equals(self.l[pos], 0):
            # nothing to do
            return

        assert self.approx_equals(h_pos, 0) and (self.l[pos] > 0), 'invalid support vector'
        self.into_support(pos)
        self.update_kernel()

    def learn(self, i, C, revert=False):

        self.t_hdr.start()

        pos = self.get_pos(i)
        self.C[pos] = C

        # if support vector then C increased, there is nothing to do
        if i in self.support_set:
            assert self.l[pos] <= self.C[pos], 'C should only increase for support vectors'
            return

        # if it is a rest (other) vector, there is also no impact
        assert i in self.rest_set, 'vector should be either in support or in rest'
        if self.approx_equals(self.h(pos), 0, tol=1e-6):
            if revert:
                self.t_migrate.start()
                self.into_support(pos - self._vectors.sv_len())
                self.t_migrate.stop()

                self.t_updatek.start()
                self.update_kernel()
                self.t_updatek.stop()
            else:
                assert self.approx_equals(self.l[pos], 0)
                # find out the closest one support or error vector, remove it and push this instead
                #raise SingularMatrixException(f'removing non-essential vector {i}')
            return

        if pos < len(self.support_set) + len(self.rest_set) - 1:
            self.t_move.start()
            self.move_to_end(i)
            self.t_move.stop()

            self.t_updatek.start()
            self.update_kernel()
            self.t_updatek.stop()

        self.t_hdr.stop()

        # if its contribution is approximately identical to other vector's do not learn it
        # the two vectors will be too close, and that would cause numerical errors on migration

        self.t_h.start()
        h = self.compute_h()
        self.t_h.stop()

        similarity = self.approx_equals(h[:-1], h[-1], tol=1e-5)
        if torch.any(similarity):
            # TODO: remove old vector(s) and push the new one
            raise SingularMatrixException(f'contibution of vector {i} found to be too close with other one')

        self.t_loop.start()
        while self.not_yet_learned(i):
            self.t_learn.start()
            self._learn()
            self.t_learn.stop()

        self.t_loop.stop()

        self.t_end.start()
        self._potential_migrate_last()
        self._potential_migrate_rest()
        assert len(self._vectors.support_set) > 0, 'support set cannot be left without vectors'
        self.t_end.stop()

    def unlearn(self, i, C):
        pos = self.get_pos(i)
        self.C[pos] = C

        # if it is a support vector and C is still bigger enough, no impact, no removal
        if i in self.support_set and self.l[pos] <= self.C[pos] and self.C[pos] > 0:
            return

        # if it is a rest (other) vector, there is also no impact, no removal
        if i in self.rest_set and self.approx_equals(self.l[pos], self.C[pos]) and self.C[pos] > 0:
            return

        if pos < len(self.support_set) + len(self.rest_set) - 1 or i in self.support_set:
            self.move_to_end(i)
            self.update_kernel()
            pos = self.get_pos(i)
            assert pos == len(self.support_set) + len(self.rest_set) - 1

        while self.not_yet_unlearned(i):
            self._unlearn()

        # remove vector
        if self.approx_equals(self.C[pos], 0.):
            # print(f'>> removing  unlearned {i} vector')
            assert self.get_pos(i) == len(self.support_set) + len(self.rest_set) - 1
            self.remove()
            self.update_kernel()

        if len(self.rest_set) > 0:
            self._potential_migrate_last()
            self._potential_migrate_rest()
        assert len(self._vectors.support_set) > 1, 'support set cannot be left with just one vector'

    def prevent_unlearn(self, i):
        """
        Check whether this is the only of its class

        :param i: index in real set
        :return: boolean
        """
        pos = self.get_pos(i)
        return (self.y == self.y[pos]).sum() == 1

    def print_h(self, agressive_asserts=False):
        h = self.compute_h()
        print(f'>> ====[ h() for all ({len(h)}) vectors: ]====')

        for i in range(len(h)):
            print(f'>> h({i:>{5}}): {h[i]: <{15.8}} lambda: {self.l[i]: <{15.8}} y: {self.y[i].item(): <{6}} C: {self.C[i]: <{5.4}}')

        # check conditions for lambdas
        sum = self.l.dot(self.y.view(-1))
        assert self.approx_equals(sum.item(), 0, tol=1e-5), 'sum lambda_i * y_i should be zero'

        n = len(self.support_set)
        assert torch.all(self.approx_equals(h[:n], 0.0, tol=1e-3)), 'all gradients for support vectors should be zero'

        if agressive_asserts:
            test = torch.all(self.approx_equals(self.l[torch.where(h[n:] > 0)[0] + n], 0.0))
            assert test, 'positive gradients should have zero lambdas'

            negatives = torch.where((h[n:] < 0) & ~self.approx_equals(h[n:], 0.0, tol=1e-12))[0] + n
            test = torch.all(self.approx_equals(self.l[negatives], self.C[negatives]))
            assert test, 'negative gradients should have C lambdas'

            zeros = torch.where(self.approx_equals(h[n:], 0.0, tol=1e-12))[0]
            assert len(zeros) == 0, 'zero gradients should be support vectors instead'

    def print_sets(self):
        self._vectors.print_sets()

    def show_state(self, aggressive_asserts=True, force=False):
        if self.debug or force:
            self.print_sets()
            self.print_h(agressive_asserts=aggressive_asserts)
