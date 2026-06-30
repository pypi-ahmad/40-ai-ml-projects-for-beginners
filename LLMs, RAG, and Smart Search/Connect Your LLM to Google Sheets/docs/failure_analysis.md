# Failure Analysis Playbook

## Scenarios Covered
- Messy spreadsheets (mixed headers, sparse rows)
- Missing required columns
- Invalid dates / numbers
- Duplicate headers
- Merged-cell ingestion artifacts
- Large worksheets
- Permission errors (403 / SpreadsheetNotFound)
- Rate limits
- Credential file errors

## Handling Strategy
1. Detect issue via validation checks.
2. Emit structured error code and human-readable fix.
3. Apply retry only for transient API failures.
4. Fall back to cached data when safe.
5. Keep source sheet immutable by default.

## Example Error Mapping
- `SpreadsheetNotFound` -> Check sheet ID and sharing with service account.
- `403` -> Verify API scopes and IAM + sheet permissions.
- `429` -> Retry with exponential backoff + jitter.
- `invalid_grant` -> Refresh OAuth token or fix service account key.
