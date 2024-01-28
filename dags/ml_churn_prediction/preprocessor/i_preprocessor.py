from abc import ABC, abstractmethod

class IPreprocessor(ABC):
    @abstractmethod
    @staticmethod
    def preprocess(self, df):
        pass