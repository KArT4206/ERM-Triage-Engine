# Security policy

## Scope

This policy covers the ERM Triage Engine web application: the Flask server, its routes and the dashboard. It does not cover Hugging Face's hosting infrastructure or the OCTDL dataset.

## Data handling

Uploaded images are processed in memory for a single request and are not stored or logged. The server reads files only from fixed folders (`samples/`, a fixed list under `paper/`), so a request cannot reach other files.

## Reporting a vulnerability

Please report security issues privately to the repository owner instead of opening a public issue. Include a description, steps to reproduce and any relevant screenshots. Reports are handled on a best-effort basis; this is a research project with no dedicated security team.
