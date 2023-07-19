# fucntion testing
import numpy as np
import math

def numpy_make_unique(arr):
    return np.unique(arr,axis=0)

def numpy_sort(arr, ax=0):
    return arr[arr[:,ax].argsort()]

    
def test():
    data = np.array([[         np.nan, 526.75675676],
            [ -8.12058804, 200.        ],
            [         np.nan, 420.        ]])
    
    print(numpy_make_unique(numpy_sort(data,ax=1)))


class apply_function(object):
    def __init__(self, function, *args):
        self.function = function
        self.args = args
        self.result = self.solve()
    def solve(self):
        return self.function(*self.args)

def test2():
    def add(a,b,c):
        return a+b+c

    def quadratic(a,b,c):
        return (-b + math.sqrt(b**2 - 4*a*c))/(2*a)

    def tensor_mult(a,b):

        assert a.shape == b.shape
        return np.tensordot(a,b,axes=0)

    adder = apply_function(add,1,2,3)
    quad = apply_function(quadratic,1,2,-3)
    tensor = apply_function(tensor_mult,np.array([1,2,3]),np.array([1,2,3]))

    funcs = [adder,quad,tensor]
    resuls = [i.result for i in funcs]
    print(resuls)



if __name__ == "__main__":
    test2()