"""Module providing mathematical vector functionality."""
import numpy as np

def kernel_or (vector_a: np.ndarray, vector_b: np.ndarray) -> int:
    '''
    Logical OR kernel function. Performs the sum of OR(a_i, b_i).

    Inputs
    - np.ndarray:   vector 1 with discrete values 0 or 1
    - np.ndarray:   vector 2 with discrete values 0 or 1

    Returns
    - Integer
    '''

    return np.sum(np.logical_or(vector_a, vector_b))


def kernel_and (vector_a: np.ndarray, vector_b: np.ndarray) -> int:
    '''
    Logical AND kernel function. Performs the sum of AND(a_i, b_i).

    Inputs
    - np.ndarray:   vector 1 with discrete values 0 or 1
    - np.ndarray:   vector 2 with discrete values 0 or 1

    Returns
    - Integer
    '''

    return np.sum(np.logical_and(vector_a, vector_b))


def kernel_not (vector_a: np.ndarray, vector_b: np.ndarray) -> int:
    '''
    Logical not kernel function. Performs the sum of NOT(a_i, b_i).

    Inputs
    - np.ndarray:   vector 1 with discrete values 0 or 1
    - np.ndarray:   vector 2 with discrete values 0 or 1

    Returns
    - Integer
    '''

    return np.sum(np.logical_not(vector_a, vector_b))


def kernel_xor (vector_a: np.ndarray, vector_b: np.ndarray) -> int:
    '''
    Logical xor kernel function. Performs the sum of XOR(a_i, b_i).

    Inputs
    - np.ndarray:   vector 1 with discrete values 0 or 1
    - np.ndarray:   vector 2 with discrete values 0 or 1

    Returns
    - Integer
    '''

    return np.sum(np.logical_xor(vector_a, vector_b))


def kernel_rbf (vector_a: np.ndarray, vector_b: np.ndarray, gamma: float) -> float:
    '''
    Radial bias kernel function.  Computes exp( -gamma ||a-b||^2 )

    Inputs
    - np.ndarray:   vector 1
    - np.ndarray:   vector 2
    - float:        gamma (typically a value between 10^-9 and 10^3)

    Returns
    - Float
    '''
    return np.sum(np.exp(-gamma * np.square(vector_a - vector_b)))


def kernel_gaussian (vector_a: np.ndarray, vector_b: np.ndarray, gamma: float) -> float:
    '''
    Radial bias kernel function.  Computes exp( -gamma ||a-b||^2 )

    Inputs
    - np.ndarray:   vector 1
    - np.ndarray:   vector 2
    - float:        gamma (typically a value between 10^-9 and 10^3)

    Returns
    - Float
    '''
    return kernel_rbf (vector_a=vector_a, vector_b=vector_b, gamma=gamma)


def kernel_dot (vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    '''
    Linear kernel function.  Performs the dot product of the two vectors

    Inputs
    - np.ndarray:   vector 1
    - np.ndarray:   vector 2

    Returns
    - Float
    '''
    return np.dot(vector_a, vector_b)


def kernel_linear (vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    '''
    Linear kernel function.  Performs the dot product of the two vectors

    Inputs
    - np.ndarray:   vector 1
    - np.ndarray:   vector 2

    Returns
    - Float
    '''
    return kernel_dot (vector_a=vector_a, vector_b=vector_b)


def kernel_polynomial (
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        gamma: float,
        shift: float = 0.0,
        degree: int = 1
    ) -> float:
    '''
    Polynomial kernel function.  Performs K(x, y) = (γ<x, y> + r)^d on the 
    two vectors.  When constant and degree are not set it reverts to a linear 
    kernel

    Inputs
    - np.ndarray:   vector 1
    - np.ndarray:   vector 2
    - float:        gamma (typically a value between 10^-9 and 10^3)
    - float:        constant shift "c", when c is greater than zero, it 
                    allows terms of lower order than d to influence the 
                    model.
    - int:          degree, controls the flexibility, typically 2 or 3.

    Returns
    - Float
    '''
    return np.sum (( gamma * np.dot (vector_a, vector_b) + shift ) ** degree)


def kernel_sigmoid (
        vector_a: np.ndarray,
        vector_b: np.ndarray,
        gamma: float,
        shift: float = 0.0,
    ) -> float:
    '''
    Sigmoid Kernel.  Performs tanh(gamma * <x, y> + shift) on the two 
    vectors.

    Inputs
    - np.ndarray:   vector 1
    - np.ndarray:   vector 2
    - float:        gamma (typically a value between 10^-9 and 10^3)
    - float:        constant shift "c", when c is greater than zero, it 
                    allows terms of lower order than d to influence the 
                    model.
    - int:          degree, controls the flexibility, typically 2 or 3.

    Returns
    - float
    '''
    return np.sum (np.tanh(gamma * np.dot(vector_a, vector_b) + shift))
