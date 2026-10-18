# Nimbus Platform - Authentication Guide

Nimbus is a fictional cloud data platform used as the demo corpus for open-rag.

## API keys

Every request to the Nimbus API must include an `X-Nimbus-Key` header. API keys are created in the console under Settings > Access. A key expires after 90 days and must then be rotated. Rotation does not interrupt traffic because the old key stays valid for 24 hours after a new key is issued.

## OAuth tokens

Service-to-service integrations should use OAuth 2.0 client credentials instead of API keys. Access tokens issued by Nimbus live for 15 minutes. Refresh tokens are not issued for the client credentials flow, so the client must request a new access token when the current one expires.

## Error codes

Error NB-4021 means the API key is expired or revoked. Error NB-4030 means the key is valid but lacks the required scope. Error NB-4290 means the rate limit has been exceeded; the response includes a `Retry-After` header expressed in seconds.

## Single sign-on

Enterprise plans support SAML 2.0 single sign-on. Group-to-role mapping is configured in the identity provider and synchronized every 10 minutes.
