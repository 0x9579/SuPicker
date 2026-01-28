from dataclasses import dataclass, asdict, fields
from typing import TypeVar, Type

T = TypeVar("T", bound="Config")


@dataclass
class Config:
    """Base configuration class with serialization support."""

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Create config from dictionary."""
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)
