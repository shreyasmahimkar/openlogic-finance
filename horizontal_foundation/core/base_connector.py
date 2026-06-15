from abc import ABC, abstractmethod


class BaseConnector(ABC):
    """Abstract base class representing an external data connector."""

    @abstractmethod
    def fetch(self, *args, **kwargs) -> dict:
        """Performs standard ingestion and returns metadata/results."""
        pass
