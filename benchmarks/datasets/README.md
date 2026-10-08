# HRI benchmark data

Each record contains an observed binary outcome y and one or more model
probabilities named p_<model>. Every model must receive the same held-out
records.

## Evidence levels

- human_collected: newly collected human/HRI observations.
- public_hri: public human or HRI dataset with documented provenance.
- aggregate: published aggregate observations.
- synthetic: generated data for debugging, recovery, and sensitivity analysis.

Synthetic data must never be presented as human-subject validation.

## Provenance requirements

Record dataset name, version/date, source, license/permission, inclusion and
exclusion criteria, train/validation/test split, and a SHA-256 hash where a
file is available. Do not commit private or identifiable participant data.

See schema.json for the machine-readable record schema.
