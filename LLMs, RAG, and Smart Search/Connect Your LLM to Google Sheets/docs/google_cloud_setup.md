# Google Cloud Setup (Beginner Friendly)

## 1) Create Google Cloud Project
1. Open Google Cloud Console.
2. Click project selector -> `New Project`.
3. Set name (for example `ai-spreadsheet-analytics`) and create.
4. Confirm active project in top bar.

## 2) Enable APIs
1. Go to `APIs & Services -> Library`.
2. Search and enable:
   - `Google Sheets API`
   - `Google Drive API`

## 3) Create Service Account
1. Go to `IAM & Admin -> Service Accounts`.
2. Click `Create Service Account`.
3. Add name + ID, continue.
4. Grant minimum role for read workflows:
   - `Viewer` (project)
5. Finish.

## 4) Create JSON Credentials
1. Open created service account.
2. Go `Keys -> Add Key -> Create New Key`.
3. Choose `JSON`.
4. Download key file.
5. Store outside repo or inside secret manager.

## 5) Share Sheet with Service Account
1. Open target Google Sheet.
2. Click `Share`.
3. Add service-account email (looks like `name@project.iam.gserviceaccount.com`).
4. Assign `Viewer` by default.
5. For write-back, use `Editor` only on destination sheet.

## 6) `.env` setup
```bash
cp .env.example .env
```
Set:
```text
GOOGLE_SERVICE_ACCOUNT_JSON=/abs/path/service_account.json
GOOGLE_SCOPES=https://www.googleapis.com/auth/spreadsheets.readonly,https://www.googleapis.com/auth/drive.readonly
```

## 7) Common Permission Errors
- `SpreadsheetNotFound`
  - Wrong sheet ID or sheet not shared with service account.
- `403 insufficient permissions`
  - Missing scope or missing sheet share permission.
- `invalid_grant`
  - Clock skew or invalid OAuth token.
- `The caller does not have permission`
  - API not enabled or wrong project key.

## 8) Security Best Practices
- Never commit JSON keys.
- Keep scopes minimal (`readonly` default).
- Use separate service account for dev/prod.
- Rotate keys regularly.
- Revoke leaked keys immediately in Service Account key page.
- Use environment variables and secret stores in deployment.
- Enable Cloud audit logs.

## 9) Avoid Credential Leaks in GitHub
- Keep `.env` ignored in `.gitignore`.
- Add credential filename patterns to `.gitignore`.
- Run secret scan before push.
- If leak happens:
  1. Revoke key now.
  2. Purge history if needed.
  3. Reissue new key.
