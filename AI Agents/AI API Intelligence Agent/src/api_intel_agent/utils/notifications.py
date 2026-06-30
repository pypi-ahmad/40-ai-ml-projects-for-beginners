"""Notification helpers for email and Slack."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import httpx


def send_slack_notification(text: str) -> bool:
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook:
        return False
    try:
        response = httpx.post(webhook, json={"text": text}, timeout=20)
        return response.status_code < 300
    except Exception:
        return False


def send_email_notification(subject: str, body: str, to_email: str) -> bool:
    host = os.getenv("SMTP_HOST", "")
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASS", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    if not (host and user and password and to_email):
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = user
    message["To"] = to_email
    message.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(message)
        return True
    except Exception:
        return False
