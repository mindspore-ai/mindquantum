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

    # @abstractmethod
    # def export_trained_parameters(self):
    #     """
    #     export trained parameters to hard disk.
    #     """
    #     pass

    # @abstractmethod
    # def load_trained_parameters(self):
    #     """
    #     load trained parameters and set it into hybrid netwok.
    #     """
    #     pass

    # @abstractmethod
    # def train(self):
    #     """
    #     train the hybrid network
    #     """
    #     pass

    # @abstractmethod
    # def predict(self, origin_test_x) -> float:
    #     """
    #     evaluate the accuracy of the test set based on the hybrid network.

    #     Args:
    #         origin_test_x: the test samples similar with origin_x

    #     Returns:
    #         list, the predict labels, similar with origin_y
    #     """
    #     pass
    
if __name__=='__main__':
    a=HybridModel()
    x=[]
    for i in a.origin_x:
        k=list(i.reshape((1,-1))[0].astype(np.int32))
        k= [str(j) for j in k]
        k = ''.join(k)
        if k not in x:
            x.append(k)
    a=[0]*16
    for i in x:
        for j in range(16):
            if i[j]=='0':
                a[j]+=1
                
        
    
