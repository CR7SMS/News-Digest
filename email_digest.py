"""Render the digest as a mobile-friendly HTML email (inline CSS only, no
external images) plus a plain-text alternative, and send it via Gmail SMTP.

Sending needs three environment variables (from .env locally, or GitHub
secrets in Actions):
  DIGEST_EMAIL_FROM   the Gmail address that sends the digest
  DIGEST_EMAIL_TO     recipient(s), comma-separated
  GMAIL_APP_PASSWORD  a Gmail app password for DIGEST_EMAIL_FROM (not your normal password)
If any are missing, sending is skipped with a message (the HTML is still saved
by run_digest.py).
"""

import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from html import escape

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
AXES = [("scale", "Scale"), ("relevance", "Relevance"), ("major_lab", "Major lab"),
        ("preference", "Preference"), ("substance", "Substance")]


def subject_for(when: datetime) -> str:
    return f"AI News Digest - {when:%a, %b} {when.day}, {when.year}"


def _date_line(when: datetime) -> str:
    return f"{when:%A, %B} {when.day}, {when.year}"


def _badge_colors(score: float) -> tuple[str, str]:
    if score >= 7:
        return "#e6f4ea", "#1e6b34"
    if score >= 5:
        return "#fff4e0", "#8a5300"
    return "#eef0f3", "#4a5260"


def _axes_line(s: dict) -> str:
    return " · ".join(f"{label} {s[key]:.0f}" for key, label in AXES)


def render_html(stories: list[dict], when: datetime) -> str:
    items = []
    for i, s in enumerate(stories, start=1):
        bg, fg = _badge_colors(s["overall"])
        items.append(f"""
        <tr><td style="padding:18px 0;border-top:1px solid #e6e8eb;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
            <td valign="top" style="width:28px;font:600 15px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#8a919c;">{i}.</td>
            <td valign="top">
              <a href="{escape(s['url'], quote=True)}" style="font:600 17px/1.35 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a4fbf;text-decoration:none;">{escape(s['title'])}</a>
              <div style="margin-top:6px;font:13px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#5b6270;">
                {escape(s['outlet'])}
                &nbsp;<span style="display:inline-block;padding:1px 8px;border-radius:10px;background:{bg};color:{fg};font-weight:700;font-size:12px;">{s['overall']:.1f} / 10</span>
              </div>
              <div style="margin-top:8px;font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#222831;">{escape(s['reason'])}</div>
              <div style="margin-top:6px;font:12px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#8a919c;">{escape(_axes_line(s))}</div>
            </td>
          </tr></table>
        </td></tr>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(subject_for(when))}</title></head>
<body style="margin:0;padding:0;background:#f4f5f7;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f4f5f7;">
  <tr><td align="center" style="padding:16px 12px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;background:#ffffff;border-radius:10px;">
      <tr><td style="padding:22px 20px 6px 20px;">
        <div style="font:700 22px/1.3 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#111418;">AI News Digest</div>
        <div style="margin-top:4px;font:14px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#5b6270;">{_date_line(when)} &middot; top {len(stories)} stories, scored 0-10 by an LLM judge</div>
      </td></tr>
      <tr><td style="padding:4px 20px 8px 20px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">{''.join(items)}
        </table>
      </td></tr>
      <tr><td style="padding:12px 20px 20px 20px;border-top:1px solid #e6e8eb;font:12px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#8a919c;">
        Overall = average of five 0-10 axes (scale, relevance to AI evaluation, major-lab involvement, reader preference, substance). Paywalled or unreadable articles are skipped.
      </td></tr>
    </table>
  </td></tr>
</table>
</body></html>
"""


def render_text(stories: list[dict], when: datetime) -> str:
    lines = [f"AI News Digest - {_date_line(when)}", ""]
    for i, s in enumerate(stories, start=1):
        lines += [
            f"{i}. {s['title']}",
            f"   {s['outlet']} | overall {s['overall']:.1f}/10",
            f"   {s['reason']}",
            f"   {_axes_line(s)}",
            f"   {s['url']}",
            "",
        ]
    lines.append("Overall = average of five 0-10 axes. Paywalled or unreadable articles are skipped.")
    return "\n".join(lines)


def send_email(subject: str, html: str, text: str) -> bool:
    sender = os.getenv("DIGEST_EMAIL_FROM", "").strip()
    recipients = [r.strip() for r in os.getenv("DIGEST_EMAIL_TO", "").split(",") if r.strip()]
    password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()

    missing = [name for name, val in [("DIGEST_EMAIL_FROM", sender), ("DIGEST_EMAIL_TO", recipients),
                                      ("GMAIL_APP_PASSWORD", password)] if not val]
    if missing:
        print(f"Email NOT sent: missing environment variable(s) {', '.join(missing)}. "
              "The digest HTML was saved to output/ instead.")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=60) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)
    print(f"Email sent to {len(recipients)} recipient(s): {subject!r}")
    return True
