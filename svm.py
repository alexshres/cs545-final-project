'''Imports'''
from typing import Tuple, Callable
from confusion_matrix import ConfusionMatrix
from sklearn import svm


class SVM:
    '''
    SVM class
    '''
    def __init__(self, size: Tuple[int, int], kernel: Callable[..., float], shift: float = 1.0):
        self._confusion_matrix = ConfusionMatrix(size=size)
        self._kernel = kernel
        self._shift = shift
        self._classifier = svm.SVC(kernel=self._kernel, C=self._shift)

    @property
    def accuracy (self) -> float:
        '''
        Get the accuracy of the model
        '''
        return self._confusion_matrix.accuracy

    def __str__ (self) -> str:
        return str(self._confusion_matrix)

    def clear (self) -> None:
        '''
        Clear/Reset the model's accuracy and confusion matrix
        '''
        self._confusion_matrix.clear()

    def reset (self) -> None:
        '''
        Clear/Reset the model's accuracy and confusion matrix
        '''
        self.clear()

    def run (self) -> None:
        '''
        Run
        '''
        return

    def train (self, input_data, epochs: int = 1) -> None:
        '''
        train
        '''
        for _ in epochs:

            for line in input_data:

                print(line)

        return

    def test (self) -> None:
        '''
        test
        '''
        return
