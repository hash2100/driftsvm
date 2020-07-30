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


class Migration:
    """
    Migration class keeps count of vectors migrations between the sets (support and rest).

    Initial order of the vectors will change, thus the original index of the vector
    on a certain row is kept into its corresponding support_set() and rest_set()

    Support vectors are the ones in x()[:n_support], where n_support = sv_len()
    """
    def __init__(self, x, y, window_size, l=None, c=None, c_init=None):
        assert window_size > 0

        # allocate space for window_size vectors
        self._x = torch.zeros((window_size, x.shape[1]), dtype=x.dtype, device=x.device)
        self._y = torch.zeros((window_size, 1), dtype=x.dtype, device=y.device)
        self._l = torch.zeros(window_size, dtype=x.dtype, device=self._x.device)
        self._C = torch.zeros(window_size, dtype=x.dtype, device=self._x.device)

        # keeps an ordered list with support vectors
        # value at index i represents the position in the original set
        self._support_set = [0, 1]

        # also keep an ordered list with rest vectors
        # value at index i represents the position in the original set
        self._rest_set = [i for i in range(2, x.shape[0])]

        self._x[:x.shape[0]] = x
        self._y[:x.shape[0]] = y
        assert self._x.shape[0] == self._y.shape[0]

        # also initialize lambdas and C
        if l is not None:
            self._l[:x.shape[0]] = l

        if c is not None:
            self._C[:x.shape[0]] = c

        if c_init is not None:
            self._C = torch.ones(window_size, dtype=torch.double, device=self._x.device) * c_init

        # the number of actual positions used
        self._len = x.shape[0]
        self._window_size = window_size

    @property
    def x(self):
        return self._x[:self._len]

    @property
    def y(self):
        return self._y[:self._len]

    @property
    def l(self):
        return self._l[:self._len]

    @property
    def C(self):
        return self._C[:self._len]

    @property
    def support_set(self):
        return self._support_set

    @property
    def rest_set(self):
        return self._rest_set

    def sv_len(self):
        return len(self.support_set)

    def len(self):
        return self._len

    def print_sets(self):
        print(f'   Support set({len(self.support_set)}): {self.support_set}')
        print(f'   Rest set({len(self.rest_set)}): {self.rest_set}')

    def _from_support_index(self, pos, n_support):
        index = torch.arange(self._window_size)
        t = index[pos].clone()
        index[pos:n_support - 1] += 1
        index[n_support - 1] = t

        return index
        #return torch.cat((x[:pos], x[pos + 1:n_support], x[pos].unsqueeze(0), x[n_support:]), 0)

    def from_support(self, pos):
        """
        Migrates a vector from the support set to the beginning of the rest set.
        :param pos: position in the support set
        """
        # print(f'>> S --> R: support vector ({self.support_set[pos]}), pos={pos}, migrates to Rest')

        n_support = len(self._support_set)
        assert pos < n_support

        # if is the last in the support set, there is no need to move it in the matrix
        if pos == n_support - 1:
            self._rest_set = [self._support_set[-1]] + self._rest_set
            self._support_set = self._support_set[:-1]
            return

        # adjust matrix, labels, lambdas, C
        index = self._from_support_index(pos, n_support)
        self._x = self._x[index]
        self._y = self._y[index]
        self._l = self._l[index]
        self._C = self._C[index]

        # adjust counters
        self._rest_set = [self._support_set[pos]] + self._rest_set
        self._support_set = self._support_set[:pos] + self._support_set[pos + 1:]

        return index

    def _into_support_index(self, pos, n_support):
        index = torch.arange(self._window_size)
        t = index[pos].clone()
        index[n_support:pos+1] -= 1
        index[n_support] = t

        return index
        #return torch.cat((x[:n_support], x[pos].unsqueeze(0), x[n_support:pos], x[pos + 1:]), 0)

    def into_support(self, pos_in_rest):
        """
        Migrate a vector from the rest set to the end of the support set.
        :param pos_in_rest: position in the rest set (zero-based)
        """
        assert pos_in_rest < len(self._rest_set)

        # print(f'>> R --> S: rest vector ({self.rest_set[pos_in_rest]}), pos={pos_in_rest}, migrates to Support')

        n_support = len(self._support_set)
        pos = pos_in_rest + n_support

        # if is the first in the rest set, there is no need to move it in the matrix
        if pos_in_rest == 0:
            self._support_set = self._support_set + [self._rest_set[0]]
            self._rest_set = self._rest_set[1:]
            return

        # adjust matrix, labels, lambdas, C
        index = self._into_support_index(pos, n_support)
        self._x = self._x[index]
        self._y = self._y[index]
        self._l = self._l[index]
        self._C = self._C[index]

        # adjust counters
        self._support_set = self._support_set + [self._rest_set[pos_in_rest]]
        self._rest_set = self._rest_set[:pos_in_rest] + self._rest_set[pos_in_rest + 1:]

        return index

    def _move_to_end_index(self, pos):
        index = torch.arange(self._window_size)
        t = index[pos].clone()
        index[pos:self._len-1] += 1
        index[self._len-1] = t

        return index
        #return torch.cat((x[:pos], x[pos + 1:], x[pos].unsqueeze(0)), 0)

    def move_to_end(self, original_pos):
        n_support = len(self.support_set)

        if original_pos in self.support_set:
            pos = self.support_set.index(original_pos)
            self._rest_set = self._rest_set + [self._support_set[pos]]
            self._support_set = self._support_set[:pos] + self._support_set[pos + 1:]
        elif original_pos in self.rest_set:
            pos = self.rest_set.index(original_pos)
            self._rest_set = self._rest_set[:pos] + self._rest_set[pos + 1:] + [self._rest_set[pos]]
            pos += n_support
        else:
            assert False, f'Original_pos {original_pos} not found in either support_set or rest_set'

        # adjust matrix, labels, lambdas, C
        index = self._move_to_end_index(pos)
        self._x = self._x[index]
        self._y = self._y[index]
        self._l = self._l[index]
        self._C = self._C[index]

        return index

    def append(self, pos, x, y, c_init, l=0.0):
        assert x.shape == (self._x.shape[1],)
        assert y.shape == (1,)
        assert self._len < self._window_size

        self._x[self._len] = x
        self._y[self._len] = y
        self._l[self._len] = l
        self._C[self._len] = c_init

        self._rest_set.append(pos)
        self._len += 1

    def remove(self):
        self._len -= 1
        self._rest_set = self.rest_set[:-1]

    @staticmethod
    def test_unit():
        print("=== Migration test_unit() begin ===")

        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        y = torch.tensor([-1, 1, 1]).reshape((-1, 1))
        mig = Migration(x, y, window_size=10, l=torch.tensor([1, 2, 3]), c=torch.tensor([10, 20, 30]))

        mig.print_sets()
        i = 0
        print(f'>> moves vector {i} from Support into Rest')
        mig.from_support(i)
        assert mig.support_set == [1]
        assert mig.rest_set == [0, 2]
        assert torch.all(mig.x[0] == torch.tensor([3, 4]))
        assert torch.all(mig.x[1] == torch.tensor([1, 2]))
        assert torch.all(mig.x[2] == torch.tensor([5, 6]))
        assert torch.all(mig.y == torch.tensor([1, -1, 1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([2, 1, 3]))
        assert torch.all(mig.C == torch.tensor([20, 10, 30]))

        mig.print_sets()
        i = 0
        print(f'>> moves vector {i} from Rest into Support')
        mig.into_support(i)
        assert mig.support_set == [1, 0]
        assert mig.rest_set == [2]
        assert torch.all(mig.x[0] == torch.tensor([3, 4]))
        assert torch.all(mig.x[1] == torch.tensor([1, 2]))
        assert torch.all(mig.x[2] == torch.tensor([5, 6]))
        assert torch.all(mig.y == torch.tensor([1, -1, 1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([2, 1, 3]))
        assert torch.all(mig.C == torch.tensor([20, 10, 30]))

        mig.print_sets()
        i = 0
        print(f'>> moves vector {i} from Support into Rest')
        mig.from_support(i)
        assert mig.support_set == [0]
        assert mig.rest_set == [1, 2]
        assert torch.all(mig.x[0] == torch.tensor([1, 2]))
        assert torch.all(mig.x[1] == torch.tensor([3, 4]))
        assert torch.all(mig.x[2] == torch.tensor([5, 6]))
        assert torch.all(mig.y == torch.tensor([-1, 1, 1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([1, 2, 3]))
        assert torch.all(mig.C == torch.tensor([10, 20, 30]))

        mig.print_sets()
        i = 1
        print(f'>> moves vector {i} from Rest into Support')
        mig.into_support(i)
        assert mig.support_set == [0, 2]
        assert mig.rest_set == [1]
        assert torch.all(mig.x[0] == torch.tensor([1, 2]))
        assert torch.all(mig.x[1] == torch.tensor([5, 6]))
        assert torch.all(mig.x[2] == torch.tensor([3, 4]))
        assert torch.all(mig.y == torch.tensor([-1, 1, 1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([1, 3, 2]))
        assert torch.all(mig.C == torch.tensor([10, 30, 20]))

        mig.print_sets()
        i = 2
        print(f'>> moves to end original vector {i}')
        mig.move_to_end(i)
        assert mig.support_set == [0]
        assert mig.rest_set == [1, 2]
        assert torch.all(mig.x[0] == torch.tensor([1, 2]))
        assert torch.all(mig.x[1] == torch.tensor([3, 4]))
        assert torch.all(mig.x[2] == torch.tensor([5, 6]))
        assert torch.all(mig.y == torch.tensor([-1, 1, 1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([1, 2, 3]))
        assert torch.all(mig.C == torch.tensor([10, 20, 30]))

        mig.print_sets()
        i = 1
        print(f'>> moves to end original vector {i}')
        mig.move_to_end(i)
        assert mig.support_set == [0]
        assert mig.rest_set == [2, 1]
        assert torch.all(mig.x[0] == torch.tensor([1, 2]))
        assert torch.all(mig.x[1] == torch.tensor([5, 6]))
        assert torch.all(mig.x[2] == torch.tensor([3, 4]))
        assert torch.all(mig.y == torch.tensor([-1, 1, 1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([1, 3, 2]))
        assert torch.all(mig.C == torch.tensor([10, 30, 20]))

        mig.print_sets()
        i = 0
        print(f'>> moves to end original vector {i}')
        mig.move_to_end(i)
        assert mig.support_set == []
        assert mig.rest_set == [2, 1, 0]
        assert torch.all(mig.x[0] == torch.tensor([5, 6]))
        assert torch.all(mig.x[1] == torch.tensor([3, 4]))
        assert torch.all(mig.x[2] == torch.tensor([1, 2]))
        assert torch.all(mig.y == torch.tensor([1, 1, -1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([3, 2, 1]))
        assert torch.all(mig.C == torch.tensor([30, 20, 10]))

        mig.print_sets()
        i = 1
        print(f'>> moves vector {i} from Rest into Support')
        mig.into_support(i)
        assert mig.support_set == [1]
        assert mig.rest_set == [2, 0]
        assert torch.all(mig.x[0] == torch.tensor([3, 4]))
        assert torch.all(mig.x[1] == torch.tensor([5, 6]))
        assert torch.all(mig.x[2] == torch.tensor([1, 2]))
        assert torch.all(mig.y == torch.tensor([1, 1, -1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([2, 3, 1]))
        assert torch.all(mig.C == torch.tensor([20, 30, 10]))

        mig.print_sets()
        i = 0
        print(f'>> moves vector {i} from Rest into Support')
        mig.into_support(i)
        assert mig.support_set == [1, 2]
        assert mig.rest_set == [0]
        assert torch.all(mig.x[0] == torch.tensor([3, 4]))
        assert torch.all(mig.x[1] == torch.tensor([5, 6]))
        assert torch.all(mig.x[2] == torch.tensor([1, 2]))
        assert torch.all(mig.y == torch.tensor([1, 1, -1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([2, 3, 1]))
        assert torch.all(mig.C == torch.tensor([20, 30, 10]))

        mig.print_sets()
        print(f'>> appends a new vector (to Rest)')
        mig.append(3, torch.tensor([7, 8]), torch.tensor([-1]), l=4, c_init=40)
        assert mig.support_set == [1, 2]
        assert mig.rest_set == [0, 3]
        assert torch.all(mig.x[0] == torch.tensor([3, 4]))
        assert torch.all(mig.x[1] == torch.tensor([5, 6]))
        assert torch.all(mig.x[2] == torch.tensor([1, 2]))
        assert torch.all(mig.x[3] == torch.tensor([7, 8]))
        assert torch.all(mig.y == torch.tensor([1, 1, -1, -1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([2, 3, 1, 4]))
        assert torch.all(mig.C == torch.tensor([20, 30, 10, 40]))

        mig.print_sets()
        i = 0
        print(f'>> moves to end original vector {i}')
        mig.move_to_end(i)
        assert mig.support_set == [1, 2]
        assert mig.rest_set == [3, 0]
        assert torch.all(mig.x[0] == torch.tensor([3, 4]))
        assert torch.all(mig.x[1] == torch.tensor([5, 6]))
        assert torch.all(mig.x[2] == torch.tensor([7, 8]))
        assert torch.all(mig.x[3] == torch.tensor([1, 2]))
        assert torch.all(mig.y == torch.tensor([1, 1, -1, -1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([2, 3, 4, 1]))
        assert torch.all(mig.C == torch.tensor([20, 30, 40, 10]))

        mig.print_sets()
        print(f'>> removes the last vector from Rest')
        mig.remove()
        assert mig.support_set == [1, 2]
        assert mig.rest_set == [3]
        assert torch.all(mig.x[0] == torch.tensor([3, 4]))
        assert torch.all(mig.x[1] == torch.tensor([5, 6]))
        assert torch.all(mig.x[2] == torch.tensor([7, 8]))
        assert torch.all(mig.y == torch.tensor([1, 1, -1]).unsqueeze(1))
        assert torch.all(mig.l == torch.tensor([2, 3, 4]))
        assert torch.all(mig.C == torch.tensor([20, 30, 40]))

        mig.print_sets()
        print("=== Migration test_unit() end ===")
