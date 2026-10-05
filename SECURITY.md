# Security policy

## Supported versions

Only the latest release of `qlcog` receives fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Email **yeshwanth445@gmail.com** with a
description, the affected version and steps to reproduce. Reports are acknowledged within 7 days.

## Scope notes

- `qlcog` runs locally and makes network calls only when a cloud backend is requested
  (`ibm:<device>`, `braket:<ARN>` in `qlcog.circuits.run`). Credentials are read by the providers' own
  SDKs (Qiskit Runtime saved account, AWS configuration); `qlcog` never stores or logs them.
- Interactive HTML figures load Plotly's JavaScript from its CDN.
- Hardware runs are billed by the provider; check the provider's pricing before submitting jobs.
