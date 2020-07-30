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


import time
import math as m

class Timing:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tic = None
        self.toc = None
        self.sum = 0
        self.sumsq = 0
        self.count = 0
        self.val = None

    def start(self):
        self.tic = time.perf_counter()

    def stop(self):
        self.toc = time.perf_counter()

        self.val = self.toc - self.tic
        self.sum += self.val
        self.sumsq += self.val * self.val
        self.count += 1

    def get_mean(self):
        return self.sum / self.count if self.count > 0 else 0

    def get_std(self):
        return m.sqrt(self.sumsq / self.count - (self.sum / self.count) ** 2) if self.count > 0 else 0
