from abc import ABC, abstractmethod

class IIngestor(ABC):
    """Interface for Ingestor classes"""

    @abstractmethod
    def bulk_csv_to_postgres(self) -> list:
        """Copy from csv to postgres table."""
        pass

    @abstractmethod
    def upsert_data_to_postgres(self, cols: list) -> None:
        """Upsert data from postgres table to another postgres table."""
        pass