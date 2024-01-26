from abc import ABC, abstractmethod

class IPreprocessor(ABC):
    """Interface for preprocessor classes."""

    @abstractmethod
    @staticmethod
    def preprocess(df):
        """Preprocess the data."""
        pass