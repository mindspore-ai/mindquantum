from abc import ABC, abstractmethod
import numpy as np
import os

project_path = os.path.dirname(os.path.abspath(__file__))


class HybridModel(ABC):
    def __init__(self):
        super().__init__()
        self.origin_data = np.load(os.path.join(project_path, 'train.npy'),
                                   allow_pickle=True)[0]
        self.origin_x = self.origin_data['train_x']
        self.origin_y = self.origin_data['train_y']

    @abstractmethod
    def export_trained_parameters(self):
        """
        export trained parameters to hard disk.
        """
        pass

    @abstractmethod
    def load_trained_parameters(self):
        """
        load trained parameters and set it into hybrid netwok.
        """
        pass

    @abstractmethod
    def train(self):
        """
        train the hybrid network
        """
        pass

    @abstractmethod
    def predict(self, origin_test_x) -> float:
        """
        evaluate the accuracy of the test set based on the hybrid network.

        Args:
            origin_test_x: the test samples similar with origin_x

        Returns:
            list, the predict labels, similar with origin_y
        """
        pass
