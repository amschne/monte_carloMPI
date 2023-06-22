#!/usr/bin/env python

from mpi4py import MPI
import numpy as  np

class Parallel(object):
    def __init__(self, data_set):
        self._comm = MPI.COMM_WORLD
        self.size = self._comm.size
        self.rank = self._comm.rank
        
        self.working_set = self._map(data_set)

    def _partition(self, data_set, partitions):
        return np.array_split(data_set, partitions)

    def _reduce(self, answer, reduce_work_fn):
    
        answer_set = self._comm.gather(answer, root=0)
    
        if self.rank == 0:
            result = reduce_work_fn(answer_set)
        else:
            result = None
    
        return result

    def _map(self, data_set):
        if self.rank == 0:
            array_to_distribute = self._partition(data_set, self.size)
        else:
            array_to_distribute = None

        #print('[%s] array_to_distribute %s' % (rank, array_to_distribute))

        working_set = self._comm.scatter(array_to_distribute, root=0)
        
        return working_set
        
    def answer_and_reduce(self, answer, reduce_work_fn):
        return self._reduce(answer, reduce_work_fn)

"""
e.g.
    def reduce_work(self, answer_set):
        return sum(answer_set)

    def work(self, working_set):
        assert(false)
        return np.sum(working_set)
        
    def parallel_work(self, data_set):    
        print('[%s] num processors: %s' % (self.rank, self.size))
    
        working_set = map(data_set)

        answer = work(working_set)
        result = reduce(answer)
    
        print('[%s] working set: %s answer: %s result: %s' % (rank, working_set, answer, result))

        return result
"""

if __name__ == '__main__':
    FULL_DATA_SET = np.arange(100)
    
    serial_data = work(FULL_DATA_SET)
    print('serial data: %s' % serial_data)
    
    parallel_data = parallel_work(FULL_DATA_SET)
    print('parallel data: %s' % parallel_data)