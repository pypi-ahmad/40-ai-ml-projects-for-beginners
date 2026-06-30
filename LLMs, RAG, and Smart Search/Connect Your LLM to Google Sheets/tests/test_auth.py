from __future__ import annotations

from pathlib import Path

import pytest

from ai_spreadsheet_analytics.connectors.auth import build_service_account_client
from ai_spreadsheet_analytics.exceptions import CredentialError


def test_service_account_missing_file_raises() -> None:
    with pytest.raises(CredentialError):
        build_service_account_client(Path("/tmp/does-not-exist.json"), ["scope"])
