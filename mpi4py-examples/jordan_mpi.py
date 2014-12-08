#!/usr/bin/env python

from mpi4py import MPI
import numpy as  np

FULL_DATA_SET = np.array(range(0, 100))

def reduce_work(answer_set):
    return sum(answer_set)

def work(working_set):
    return np.sum(working_set)

def partition(data_set, partitions):
    return np.array_split(data_set, partitions)

def reduce(comm, answer):
    rank = comm.rank
    
    answer_set = comm.gather(answer, root=0)
    
    if rank == 0:
        result = reduce_work(answer_set)
    else:
        result = None
    
    return result

def map(comm, data_set):
    size = comm.size
    rank = comm.rank
    
    if rank == 0:
        array_to_distribute = partition(FULL_DATA_SET, size)
    else:
        array_to_distribute = None

    #print('[%s] array_to_distribute %s' % (rank, array_to_distribute))

    working_set = comm.scatter(array_to_distribute, root=0)
    return working_set

def parallel_work(data_set):    
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    print('[%s] num processors: %s' % (rank, size))
    
    working_set = map(comm, data_set)

    answer = work(working_set)
    result = reduce(comm, answer)
    
    print('[%s] working set: %s answer: %s result: %s' % (rank, working_set, answer, result))

    return result


if __name__ == '__main__':
    serial_data = work(FULL_DATA_SET)
    print('serial data: %s' % serial_data)
    
    parallel_data = parallel_work(FULL_DATA_SET)
    print('parallel data: %s' % parallel_data)