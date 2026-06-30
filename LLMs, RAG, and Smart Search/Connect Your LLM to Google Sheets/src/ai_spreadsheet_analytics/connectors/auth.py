"""Google authentication helpers."""

from __future__ import annotations

from pathlib import Path

import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from loguru import logger

from ai_spreadsheet_analytics.exceptions import CredentialError


def build_service_account_client(credentials_path: Path, scopes: list[str]) -> gspread.Client:
    """Build gspread client using service-account credentials.

    Args:
        credentials_path: Path to Google service account JSON.
        scopes: Requested OAuth scopes.

    Returns:
        Authenticated gspread client.

    Raises:
        CredentialError: When credentials are invalid or inaccessible.
    """
    if not credentials_path.exists():
        raise CredentialError(f"Service account file not found: {credentials_path}")
    try:
        creds = ServiceAccountCredentials.from_service_account_file(str(credentials_path), scopes=scopes)
        return gspread.authorize(creds)
    except Exception as exc:  # noqa: BLE001
        raise CredentialError(f"Failed to initialize service-account client: {exc}") from exc


def build_oauth_client(
    client_secret_path: Path,
    token_path: Path,
    scopes: list[str],
    local_server_port: int = 8080,
) -> gspread.Client:
    """Build gspread client using OAuth installed app flow.

    Args:
        client_secret_path: OAuth client secret JSON file.
        token_path: File used to persist refresh token.
        scopes: Requested OAuth scopes.
        local_server_port: Local callback port.

    Returns:
        Authenticated gspread client.

    Raises:
        CredentialError: If OAuth flow fails.
    """
    creds: Credentials | None = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            logger.info("Refreshed OAuth token")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes)
            creds = flow.run_local_server(port=local_server_port)
            logger.info("Generated new OAuth token")
        token_path.write_text(creds.to_json(), encoding="utf-8")

    if not creds:
        raise CredentialError("OAuth credentials unavailable after flow")

    try:
        return gspread.authorize(creds)
    except Exception as exc:  # noqa: BLE001
        raise CredentialError(f"Failed to authorize OAuth client: {exc}") from exc
