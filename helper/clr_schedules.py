### Various Cyclic Learning Rate schedules implemented for Pytorch.
import numpy as np


def triangular_lr(step_size=2000,base_lr=10e-5,max_lr=10e-2):
    '''
    Triangular Cyclic Learning Rate Schedule
    '''

    scaler = lambda x:1

    lr_lambda = lambda it: base_lr + (max_lr-base_lr)*triangle(step_size,it)

    def triangle(step_size,it):
        cycle = np.floor(1+it/(2*step_size)) ## what is our cycle no.
        x = np.abs(it/step_size-2*cycle+1)
        return max(0,1-x)*scaler(cycle)

    return lr_lambda