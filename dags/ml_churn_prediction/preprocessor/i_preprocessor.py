from abc import ABC, abstractmethod

class IPreprocessor(ABC):
    @abstractmethod
    def preprocess(self) -> None:
        pass