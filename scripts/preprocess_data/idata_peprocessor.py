# ML preprocess pipeline interface

from abc import ABCMeta, abstractmethod

class IDataPreprocessor(metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def preprocess(data):
        pass
    
    @abstractmethod
    @staticmethod
    def save(data):
        pass