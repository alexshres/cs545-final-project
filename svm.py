''' SVM class '''
from typing import Callable
from sklearn import svm
from confusion_matrix import ConfusionMatrix
from utils import parse_data_line

NORMALIZATION = 255.0

class SVM:
    '''
    SVM class
    '''
    def __init__(self, num_classes: int, kernel: Callable[..., float], shift: float = 1.0):
        self._confusion_matrix = ConfusionMatrix(size=(num_classes, num_classes))
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

    def run (self, test_data) -> None:
        '''
        Take inputs and create predictions using the SVM model. Populates a Confusion
          matrix and returns the accuracy after testing.
        
        Input
        np.ndarray:         testing data array

        Output
        float:              Accuracy of the model after testing.
        '''
        return self.test(test_data)

    def train(self, input_data):
        '''
        train
        '''
        # Extract targets and features from the input data.  Normalize the features.
        targets, features = parse_data_line(input_data, normalization=NORMALIZATION)

        # Train the classifier
        return self._classifier.fit(features, targets)

    def test (self, test_data) -> None:
        '''
        Take inputs and create predictions using the SVM model. Populates a Confusion
          matrix and returns the accuracy after testing.
        
        Input
        np.ndarray:         testing data array

        Output
        float:              Accuracy of the model after testing.
        '''

        # Extract targets and features from the input data.  Normalize the features.
        targets, features = parse_data_line(test_data, normalization=NORMALIZATION)

        # Input the data through the model, and predict the classification of the data.
        classifications = self._classifier.predict(features)

        # Populate the confusion matrix
        self._confusion_matrix.reset()
        for target, prediction in zip(targets, classifications):
            self._confusion_matrix[target][prediction] += 1

        return self._confusion_matrix.accuracy
