# Security policy

## Supported versions

Only the latest release of Quantum Mind receives fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Email **yeshwanth445@gmail.com** with a
description, the affected version and steps to reproduce. Reports are acknowledged within 7 days.

## Scope notes

- Quantum Mind runs locally and makes network calls only when a cloud backend is requested
  (`ibm:<device>`, `braket:<ARN>` in `quantum_mind.circuits.run`). Credentials are read by the providers' own
  SDKs (Qiskit Runtime saved account, AWS configuration); Quantum Mind never stores or logs them.
- Interactive HTML figures load Plotly's JavaScript from its CDN.
- Hardware runs are billed by the provider; check the provider's pricing before submitting jobs.
