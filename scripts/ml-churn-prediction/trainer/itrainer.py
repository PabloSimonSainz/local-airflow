from abc import ABC, abstractmethod

class ITrainer(ABC):
    @abstractmethod
    def train_model() -> dict:
        """
        Train model
        """
        pass