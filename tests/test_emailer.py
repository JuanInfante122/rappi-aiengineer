"""Unit tests for report.emailer.send_report_email.

All tests use mocked smtplib so no real SMTP server is required.
Environment variables are injected via pytest monkeypatch.
"""

from __future__ import annotations

import smtplib
from datetime import date
from email.parser import Parser
from unittest.mock import MagicMock, patch

import pytest

from report.emailer import send_report_email

HTML = "<html><body>test report</body></html>"

SMTP_ENV = {
    "SMTP_HOST": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_USER": "user@example.com",
    "SMTP_PASSWORD": "secret",
}


def set_smtp_env(monkeypatch, overrides=None, exclude=None):
    """Configure SMTP environment variables for a test.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        overrides: Dict of env var overrides to apply on top of SMTP_ENV.
        exclude: List of env var keys to remove (simulate missing vars).
    """
    env = {**SMTP_ENV, **(overrides or {})}
    for k, v in env.items():
        if exclude and k in exclude:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


# ---------------------------------------------------------------------------
# EnvironmentError tests — missing required vars
# ---------------------------------------------------------------------------


def test_missing_smtp_host_raises(monkeypatch):
    """EnvironmentError raised when SMTP_HOST is absent."""
    set_smtp_env(monkeypatch, exclude=["SMTP_HOST"])
    with pytest.raises(EnvironmentError, match="SMTP_HOST"):
        send_report_email(HTML, "dest@example.com", "CO")


def test_missing_smtp_user_raises(monkeypatch):
    """EnvironmentError raised when SMTP_USER is absent."""
    set_smtp_env(monkeypatch, exclude=["SMTP_USER"])
    with pytest.raises(EnvironmentError, match="SMTP_USER"):
        send_report_email(HTML, "dest@example.com", "CO")


def test_missing_smtp_password_raises(monkeypatch):
    """EnvironmentError raised when SMTP_PASSWORD is absent."""
    set_smtp_env(monkeypatch, exclude=["SMTP_PASSWORD"])
    with pytest.raises(EnvironmentError, match="SMTP_PASSWORD"):
        send_report_email(HTML, "dest@example.com", "CO")


# ---------------------------------------------------------------------------
# STARTTLS path (port 587)
# ---------------------------------------------------------------------------


def test_starttls_path_calls_correct_methods(monkeypatch):
    """STARTTLS path: ehlo, starttls, login, and sendmail are all called."""
    set_smtp_env(monkeypatch)  # default port is 587

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    with patch("smtplib.SMTP", return_value=mock_instance) as mock_cls:
        send_report_email(HTML, "dest@example.com", "CO")
        mock_cls.assert_called_once()

    mock_instance.ehlo.assert_called_once()
    mock_instance.starttls.assert_called_once()
    mock_instance.login.assert_called_once()
    mock_instance.sendmail.assert_called_once()


def test_starttls_path_does_not_use_smtp_ssl(monkeypatch):
    """STARTTLS path: smtplib.SMTP_SSL must not be used on port 587."""
    set_smtp_env(monkeypatch)

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    with patch("smtplib.SMTP", return_value=mock_instance):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            send_report_email(HTML, "dest@example.com", "CO")
            mock_ssl.assert_not_called()


# ---------------------------------------------------------------------------
# SSL path (port 465)
# ---------------------------------------------------------------------------


def test_ssl_path_calls_correct_methods(monkeypatch):
    """SSL path: login and sendmail called; smtplib.SMTP must not be used."""
    set_smtp_env(monkeypatch, overrides={"SMTP_PORT": "465"})

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    with patch("smtplib.SMTP_SSL", return_value=mock_instance) as mock_ssl_cls:
        with patch("smtplib.SMTP") as mock_smtp_cls:
            send_report_email(HTML, "dest@example.com", "CO")
            mock_ssl_cls.assert_called_once()
            mock_smtp_cls.assert_not_called()

    mock_instance.login.assert_called_once()
    mock_instance.sendmail.assert_called_once()


# ---------------------------------------------------------------------------
# MIMEMultipart structure — subject and attachment filename
# ---------------------------------------------------------------------------


def test_email_subject_contains_country_and_date(monkeypatch):
    """Email subject contains the country code and today's date."""
    set_smtp_env(monkeypatch)
    country = "MX"
    today = date.today().strftime("%Y-%m-%d")

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    captured_calls: list = []

    def capture_sendmail(*args, **kwargs):
        captured_calls.append(args)

    mock_instance.sendmail.side_effect = capture_sendmail

    with patch("smtplib.SMTP", return_value=mock_instance):
        send_report_email(HTML, "dest@example.com", country)

    assert captured_calls, "sendmail was not called"
    raw_message = captured_calls[0][2]  # third arg is the message string
    parsed = Parser().parsestr(raw_message)

    assert country in parsed["Subject"], f"Country '{country}' not found in Subject"
    assert today in parsed["Subject"], f"Date '{today}' not found in Subject"


def test_attachment_filename_matches_pattern(monkeypatch):
    """Attachment Content-Disposition contains rappi_insights_{country}."""
    set_smtp_env(monkeypatch)
    country = "AR"

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    captured_calls: list = []

    def capture_sendmail(*args, **kwargs):
        captured_calls.append(args)

    mock_instance.sendmail.side_effect = capture_sendmail

    with patch("smtplib.SMTP", return_value=mock_instance):
        send_report_email(HTML, "dest@example.com", country)

    assert captured_calls, "sendmail was not called"
    raw_message = captured_calls[0][2]
    parsed = Parser().parsestr(raw_message)

    # Walk multipart to find attachment part
    attachment_found = False
    for part in parsed.walk():
        disposition = part.get("Content-Disposition", "")
        if "attachment" in disposition and f"rappi_insights_{country}" in disposition:
            attachment_found = True
            break

    assert attachment_found, (
        f"No attachment part found with 'rappi_insights_{country}' in Content-Disposition"
    )


# ---------------------------------------------------------------------------
# SMTP_FROM defaults to SMTP_USER when absent
# ---------------------------------------------------------------------------


def test_smtp_from_defaults_to_smtp_user(monkeypatch):
    """When SMTP_FROM is not set, sendmail From address equals SMTP_USER."""
    set_smtp_env(monkeypatch, exclude=["SMTP_FROM"])
    smtp_user = SMTP_ENV["SMTP_USER"]

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)

    captured_calls: list = []

    def capture_sendmail(*args, **kwargs):
        captured_calls.append(args)

    mock_instance.sendmail.side_effect = capture_sendmail

    with patch("smtplib.SMTP", return_value=mock_instance):
        send_report_email(HTML, "dest@example.com", "CO")

    assert captured_calls, "sendmail was not called"
    from_address = captured_calls[0][0]  # first arg to sendmail is the from address
    assert from_address == smtp_user, (
        f"Expected From={smtp_user!r}, got {from_address!r}"
    )
