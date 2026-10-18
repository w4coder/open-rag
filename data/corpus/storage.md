# Nimbus Platform - Storage and Limits

## Storage tiers

Nimbus offers three storage tiers. The Hot tier keeps data on SSD with single-digit millisecond reads and costs 0.12 EUR per GB per month. The Warm tier costs 0.04 EUR per GB per month and has read latency of about 80 milliseconds. The Archive tier costs 0.004 EUR per GB per month and requires a restore request that takes up to 12 hours.

## Lifecycle policies

A lifecycle policy moves objects between tiers automatically. By default, objects not read for 30 days move from Hot to Warm, and objects not read for 180 days move from Warm to Archive. Policies can be customised per bucket with a JSON rule file.

## Rate limits

The default rate limit is 600 requests per minute per project. The Business plan raises it to 3000 requests per minute and the Enterprise plan to 20000 requests per minute. Batch endpoints count as one request regardless of the number of items, up to 500 items per batch.

## Object size

The maximum size of a single object is 5 TB. Objects larger than 100 MB should use multipart upload with parts of 16 MB.
