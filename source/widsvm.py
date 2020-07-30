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

import numpy as np
import timing as t

from kernel import PolyKernel, RbfKernel
from migration import *
from svm import SVM, SingularMatrixException, NoOppositeClassLeftException

WINDOW_SIZE = 2300

def unit_test_migration():
    Migration.test_unit()
    print('\n')


def unit_test_svm():
    SVM.test_masked_min()
    print('\n')


def unit_test_polynomial():
    print('=== Polynomial test_unit() begin ===')

    # store Xes by rows
    vectors = Migration(x=torch.tensor([[0.0, 6.0], [4.0, -6.0]], dtype=torch.double),
                        y=torch.tensor([-1.0, 1.0], dtype=torch.double).reshape((-1, 1)),
                        c_init=0.5, window_size=10)

    svm = SVM(kernel_class=PolyKernel, window_size=10)
    svm.initialize(vectors)
    print(f'equation of the plane: {svm.compute_w().numpy()} * x + {svm._b.numpy()}')
    print(f'distance from [4, 4] to plane: {svm.g(torch.tensor([4.0, 4.0], dtype=torch.double), legit=True).numpy()}')

    svm.append(2, x=torch.tensor([5.0, 2.0], dtype=torch.double), y=torch.tensor([-1.0], dtype=torch.double), c_init=0.5)
    updated = svm.k
    recomputed = svm._kernel()
    assert torch.all(updated == recomputed)

    svm.show_state(False)

    assert svm.h(2) == svm.h(torch.tensor([5.0, 2.0], dtype=torch.double), -1, legit=True)

    # see the h() value for the third point
    # if it is in Others set, for lambda = 0, its h() should be > 0
    print(f'new point h(): {svm.h(2)}')

    # compute h for all vectors
    svm.print_h()
    svm.print_sets()
    g = svm.g(vectors.x, legit=True)
    h = svm.compute_h()
    gc = (h + 1) * svm.y.squeeze(1)
    print(f'g(0) = {g[0]}')
    print(f'g(1) = {g[1]}')
    print(f'g(2) = {g[2]}')
    assert g[0] == gc[0]
    assert g[1] == gc[1]
    assert g[2] == gc[2]

    # iterate as long as last vector does not migrate in support set or lambda_c != 0 or C
    svm.learn(i=2, C=0.5)
    svm.print_h()
    svm.print_sets()

    print(f'\n----------------\n')

    # now unlearn vector 2; put vector 2 on last position in rest vectors
    svm.move_to_end(2)
    svm.update_kernel()

    svm.print_sets()
    svm.print_h()

    # iterate as long as lambda_c > 0 (last vector)
    svm.unlearn(i=2, C=0.)
    svm.print_h()
    svm.print_sets()

    print('=== Polynomial test_unit() end ===')


def reinit(x_train, y_train, device, C, window_size):
    vectors = Migration(x=x_train.double().to(device), y=y_train.double().to(device).reshape(-1, 1), c_init=C, window_size=window_size)

    #svm = SVM(kernel_class=PolyKernel)
    svm = SVM(kernel_class=RbfKernel, window_size=WINDOW_SIZE)
    #svm.initialize(vectors, gamma=0.836) # electricity
    #svm.initialize(vectors, gamma=7.797753810882568) # circles
    svm.initialize(vectors, gamma=0.241477) # covertype

    print(f'distance from x[0] to plane: {svm.g(x_train[0].double().to(device), legit=True).cpu().numpy()}')
    print(f'distance from x[1] to plane: {svm.g(x_train[1].double().to(device), legit=True).cpu().numpy()}')
    print(svm.g(x_train.double().to(device), legit=True).cpu().numpy())
    print(f'y_train: {y_train}')

    # compute h for all vectors
    svm.print_h()
    svm.print_sets()

    svm.init_statistics()

    return svm, vectors

def main():
    C = 100. # electricity
    #C = 10. # circles

    device = torch.device('cpu')
    #device = torch.device('cuda')
    #d, m = torch.load('data/KDDCup1999.pt'), 2
    #d, m = torch.load('data/electricity.pt'), 2
    #d, m = torch.load('data/circles/circles.pt'), 2
    d, m = torch.load('data/covertype.pt'), 2
    x_train, y_train = d['x'][:m], d['y'][:m]

    svm, vectors = reinit(x_train, y_train, device, C, window_size=WINDOW_SIZE)

    # maximum number of vectors to be kept in the window
    #window_width = 125
    #window_width = 155 # electricity
    #window_width = 300 # covertype
    #window_width = 100  # circles
    #window_width = 1000
    window_width = WINDOW_SIZE

    svm.debug = False
    svm.init_statistics()
    set_len = 50

    file = open('running.csv', 'w')

    timer = t.Timing()

    for v in range(2, d['x'].shape[0]):
        # x_set, y_set = d['x'][v:v+set_len].double().to(device), d['y'][v:v+set_len].double().to(device)
        # x, y = x_set[0], y_set[0].unsqueeze(0)
        x, y = d['x'][v].double().to(device), d['y'][v].double().to(device).unsqueeze(0)

        if v > 3000:
            break

        if v == window_width:
            timer.reset()
            svm.reset_timers()
            print('Timers were reset')

        # learn vector v to 0.5
        if v % 100 == 0:
            print(f'>> adding {v} vector with {C/2}')
            if timer.count > 0:
                print(f'average time: {timer.get_mean()} s, std: {timer.get_std()} s')
        svm.append(v, x, y, c_init=C/2)
        # svm.update_kernel()
        svm.show_state(False)

        # before learning the vector, show how it is classified
        #svm.collect_statistics(file, v, x_set, y_set)

        # first, check if contribution of this vector is identical
        # with other's vector already added
        res = svm.detect_similar(svm.compute_q())
        if res:
            svm.remove()
            print(f'>> removing {v} vector as duplicate')
            svm.update_kernel()
            svm.show_state(False)
            continue

        last = svm.rest_set[-1]
        assert last == v
        i, c = last, C/2
        # print(f'>> learn {i} vector to {c}')
        try:
            timer.start()
            svm.learn(i, c)
            timer.stop()

        except SingularMatrixException as e:
            print(f'>> exception: {e}')
            print(f'>> removing {last} vector as singular')
            c = 0.0
            svm.unlearn(i, c)
            continue

        # svm.show_state()

        # prepare unlearning
        first = min(svm.support_set) if svm.rest_set is None or len(svm.rest_set) == 0 else min(min(svm.rest_set), min(svm.support_set))
        if len(svm.support_set) + len(svm.rest_set) >= window_width:
            i, c = first, C/2
            # print(f'>> unlearn {i} vector to {c}')
            if i not in svm.support_set and i not in svm.rest_set:
                print(f'>> vector {i} already unlearned')
                continue

            # timer.start()
            svm.unlearn(i, c)
            # timer.stop()
            # svm.show_state()

        # second phase of learning
        i, c = last, C
        # print(f'>> learn {i} vector to {c}')
        if i not in svm.support_set and i not in svm.rest_set:
            print(f'>> vector {i} already unlearned')
            continue
        try:
            timer.start()
            svm.learn(i, c)
            timer.stop()
        except SingularMatrixException as e:
            print(f'>> exception: {e}')
            print(f'>> removing {last} vector as singular')
            c = 0.0
            svm.unlearn(i, c)
            continue
        # svm.show_state()

        # second phase of unlearning
        if len(svm.support_set) + len(svm.rest_set) >= window_width:
            i, c = first, 0.0
            # print(f'>> unlearn {i} vector to {c}')

            if svm.prevent_unlearn(i):
                print(f'>> cannot unlearn {i}, it is the only of its class')
                continue

            if i not in svm.support_set and i not in svm.rest_set:
                print(f'>> vector {i} already unlearned')
                continue

            try:
                # timer.start()
                svm.unlearn(i, c)
                # timer.stop()
            except NoOppositeClassLeftException as e:
                c = C/2 # as previous
                svm.learn(i, c, revert=True)

        # svm.show_state()
        svm.print_timers()

    file.close()

    svm.show_state(force=True)


def main_tests():
    unit_test_migration()
    unit_test_svm()
    unit_test_polynomial()


if __name__ == '__main__':
    main()

