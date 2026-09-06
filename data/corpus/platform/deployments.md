# Deployment Runbook

## Environments
There are three environments: dev, staging and production. Changes flow strictly
dev -> staging -> production. A change must soak in staging for at least two hours before
production promotion.

## Release windows
Production deployments are permitted Monday to Thursday, 09:00-16:00 CET. Fridays,
weekends and the last three business days of a quarter are frozen. Freeze exceptions
require an approved change record and on-call lead sign-off.

## Rollback
Every deployment must ship with a tested rollback path. If error rate exceeds 2% of
requests for five consecutive minutes, the deployment is rolled back automatically by the
progressive delivery controller. Manual rollback is initiated with `deployctl rollback
--release <id>` and takes effect within 90 seconds.

## Canary
New releases go to 5% of traffic for ten minutes, then 25% for ten minutes, then 100%.
The canary is aborted on any increase in p99 latency above 20% relative to baseline.

## Observability
Every service must emit RED metrics and expose /healthz and /readyz. Deployments without
a dashboard link in the change record are rejected by the pipeline.
