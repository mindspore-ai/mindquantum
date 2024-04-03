from abc import ABC, abstractmethod
import numpy as np
import os
import sys

# sys.path.append("./src")

project_path = os.path.dirname(os.path.abspath(__file__))
# parent_path = os.path.dirname(project_path)
# print(parent_path)


class HybridModel(ABC):
    def __init__(self):
        super().__init__()
        # self.origin_data = np.load(os.path.join(project_path, 'train.npy'),
        #                            allow_pickle=True)[0]
        self.origin_x = np.load(os.path.join(project_path, 'new_train_x.npy'),
                                allow_pickle=True)
        self.origin_y = np.load(os.path.join(project_path, 'new_train_y.npy'),
                                allow_pickle=True)

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
