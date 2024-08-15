'''Module providing mathematical matrix functionality.'''
import numpy as np


class ConfusionMatrix():
    '''Class for a confusion matrix'''

    def __init__ (self, width: int = 1, height: int = 1) -> None:
        self._matrix = np.zeros((width, height))

    def __str__ (self) -> str:
        return str(self._matrix).replace("[", " ").replace("]", " ")

    @property
    def accuracy (self) -> float:
        '''
        Accuracy of the _matrix.

        Outputs
        Float:          Returns a value between 0 and 1 representing 
                        the accuracy of the matrix
        '''
        return ( np.trace(self._matrix) / np.sum(self._matrix))

    def __getitem__ (self, key: int) -> int:
        return self._matrix[key]

    def __setitem__ (self, key: int, value: int) -> None:
        self._matrix[key] = value

    def clear (self) -> None:
        '''
        Reset the confusion matrix count.
        '''
        self._matrix.fill(0)

    def reset (self) -> None:
        '''
        Reset the confusion matrix count.
        '''
        self.clear()