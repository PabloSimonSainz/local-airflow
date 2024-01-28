from abc import ABC, abstractmethod

class IValidator(ABC):
    @abstractmethod
    def validate(seed:int, model_name:str) -> None:
        pass