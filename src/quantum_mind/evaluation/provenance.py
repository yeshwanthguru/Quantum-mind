"""Model provenance records used by reproducible research experiments."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Optional

@dataclass(frozen=True)
class ModelProvenance:
    """Evidence and implementation metadata for one benchmark model."""
    model_id: str
    category: str
    original_reference: Optional[str] = None
    doi: Optional[str] = None
    original_algorithm: Optional[str] = None
    quantum_mind_extension: Optional[str] = None
    classical_baseline: Optional[str] = None
    validated_against: Optional[str] = None
    simulation_only: bool = True
    hardware_validated: bool = False
    human_data_validated: bool = False

    def to_dict(self):
        return asdict(self)

def validate_provenance(record):
    """Validate a provenance mapping and return a normalised dictionary."""
    if not isinstance(record, (ModelProvenance, dict)):
        raise TypeError("record must be ModelProvenance or a mapping")
    data = record.to_dict() if isinstance(record, ModelProvenance) else dict(record)
    missing = [key for key in ("model_id", "category") if not data.get(key)]
    if missing:
        raise ValueError(f"missing required provenance fields: {', '.join(missing)}")
    return data
