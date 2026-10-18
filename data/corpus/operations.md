# Nimbus Platform - Operations Runbook

## Backups

Nimbus takes an automatic snapshot of every database every 6 hours and retains snapshots for 14 days. Point-in-time recovery is available for the last 7 days with a granularity of one second. Restoring a snapshot creates a new database and never overwrites the original.

## Regions and availability

Nimbus runs in three regions: eu-west, us-east and ap-south. The availability SLA is 99.95 percent for the Business plan and 99.99 percent for the Enterprise plan. Cross-region replication is asynchronous with a typical lag below 5 seconds.

## Incident response

During a severity 1 incident the on-call engineer acknowledges within 15 minutes. Customers on the Enterprise plan receive status updates every 30 minutes until resolution. A post-incident report is published within 5 business days.

## Data deletion

When a project is deleted, its data enters a 30-day soft-delete window during which it can be restored. After that window the data is permanently erased and the deletion is irreversible.
