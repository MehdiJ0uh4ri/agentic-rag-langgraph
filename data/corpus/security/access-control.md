# Access Control Standard

## Identity
All human access is federated through the corporate IdP. Local accounts on production
systems are prohibited except for a single documented break-glass account per environment,
whose credentials are held in the offline vault and rotated every 90 days.

## Multi-factor authentication
Hardware security keys (FIDO2) are mandatory for all engineers with production access.
TOTP is accepted only for contractors during their first 30 days, after which a hardware
key must be enrolled. SMS-based second factors were retired on 2024-06-30.

## Privileged access
Production database access is granted just-in-time through the access broker for a maximum
session length of four hours. Every session is recorded and the recording is retained for
400 days. Standing production admin rights are not granted to individuals under any
circumstances.

## Review
Access reviews run quarterly. A reviewer who does not respond within ten working days
causes the reviewed grants to be revoked automatically.

## Incident thresholds
Any unauthorised production data access is a Severity 1 incident and must be declared
within 30 minutes of detection.
