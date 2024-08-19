'''Module providing mathematical matrix functionality.'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class ConfusionMatrix():
    '''Class for a confusion matrix'''

    def __init__ (self, size: tuple) -> None:
        self._matrix = np.zeros(size, dtype=int)

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
        correct = np.trace(self._matrix)
        total = np.sum(self._matrix)
        if total == 0:
            return 0.0
        return correct / total
    
    @property
    def heatmap (self) -> None:
        '''
        Plot the confusion matrix using seaborn heatmap.
        '''
        plt.figure(figsize=(10, 7))
        # Create the heatmap directly from the numpy array
        sns.heatmap(self._matrix, annot=True, fmt='d', cmap='Blues', cbar=True, 
                    xticklabels=range(0, 9), yticklabels=range(0, 9))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig("svm_confusion_matrix.png")

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
    
    def plot (self, labels, data) -> None:
        '''
        Plot accuracy over data points.
        '''
        plt.figure(figsize=(10, 5))
        plt.plot(labels, data, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of Datapoints')
        plt.ylabel('Accuracy (%)')
        plt.title(f"SVM Accuracy Over Epochs")
        plt.grid(True)
        plt.savefig("svm_accuracy_over_time.png")