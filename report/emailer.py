"""Email delivery for the Rappi Operations HTML executive report.

Sends the generated HTML report as an attachment via SMTP. The report uses
Plotly JavaScript charts that do not render inside email clients, so it is
delivered as a .html attachment rather than an inline HTML body. The email
body contains a plain-text summary so recipients know what the file contains
before opening it.

SMTP configuration is read from environment variables so no credentials are
baked into source code:

    SMTP_HOST      SMTP server hostname  (e.g. smtp.gmail.com)
    SMTP_PORT      SMTP port             (587 for STARTTLS, 465 for SSL)
    SMTP_USER      Sender email address  (e.g. reports@yourdomain.com)
    SMTP_PASSWORD  SMTP password or app-specific password
    SMTP_FROM      Display name + address shown in From header (optional,
                   defaults to SMTP_USER)
"""

from __future__ import annotations

import os
import smtplib
from datetime import date
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_report_email(
    html_content: str,
    recipient: str,
    country: str,
) -> None:
    """Send the HTML executive report to a recipient via SMTP.

    Attaches the HTML file so recipients can open it in a browser where
    Plotly charts render correctly. Includes a plain-text summary body so
    the email is readable even without opening the attachment.

    Args:
        html_content: Full HTML string produced by generate_html_report().
        recipient: Destination email address.
        country: Country code used in the subject line and filename.

    Raises:
        EnvironmentError: If any required SMTP_* env var is missing.
        smtplib.SMTPException: If the SMTP connection or send fails.
    """
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    missing = [
        var for var, val in [
            ("SMTP_HOST", smtp_host),
            ("SMTP_USER", smtp_user),
            ("SMTP_PASSWORD", smtp_password),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Email not configured. Missing environment variables: {', '.join(missing)}. "
            "Add them to your .env file."
        )

    today = date.today().strftime("%Y-%m-%d")
    subject = f"Reporte Semanal de Operaciones Rappi — {country} — {today}"
    filename = f"rappi_insights_{country}_{today}.html"

    body = (
        f"Hola,\n\n"
        f"Adjunto encontrarás el reporte ejecutivo de insights operacionales "
        f"para {country} generado el {today}.\n\n"
        f"El reporte incluye:\n"
        f"  • Resumen ejecutivo con los hallazgos críticos de la semana\n"
        f"  • Detalle por categoría: anomalías, tendencias, benchmarking, "
        f"correlaciones y oportunidades\n"
        f"  • Gráficos interactivos de Plotly (requiere abrir el archivo .html en un browser)\n\n"
        f"Abre el archivo adjunto en tu navegador para ver los gráficos interactivos.\n\n"
        f"— Rappi Ops Intelligence"
    )

    msg = MIMEMultipart()
    msg["From"] = smtp_from
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    attachment = MIMEApplication(html_content.encode("utf-8"), _subtype="html")
    attachment.add_header("Content-Disposition", "attachment", filename=filename)
    msg.attach(attachment)

    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, recipient, msg.as_string())
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, recipient, msg.as_string())
