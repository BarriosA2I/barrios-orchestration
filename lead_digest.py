"""
Daily Lead Digest Email Service
Runs at 5am EST (10:00 UTC) via Render Cron
Sends new leads to alienation2innovation@gmail.com
"""

import os
import asyncio
import asyncpg
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECIPIENT = "alienation2innovation@gmail.com"
SENDER_EMAIL = "alienation2innovation@gmail.com"  # Single Sender Verified in SendGrid
SENDER_NAME = "Barrios A2I Leads"


async def get_new_leads():
    """Get leads from last 24 hours that haven't been notified"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL not set")
        return []

    conn = await asyncpg.connect(database_url)
    try:
        leads = await conn.fetch('''
            SELECT id, name, email, company, phone, message, source, created_at
            FROM leads
            WHERE created_at > NOW() - INTERVAL '24 hours'
            AND notified = FALSE
            ORDER BY created_at DESC
        ''')
        return leads
    finally:
        await conn.close()


async def mark_leads_notified(lead_ids):
    """Mark leads as notified"""
    if not lead_ids:
        return

    database_url = os.environ.get('DATABASE_URL')
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute('''
            UPDATE leads SET notified = TRUE WHERE id = ANY($1::uuid[])
        ''', lead_ids)
        logger.info(f"Marked {len(lead_ids)} leads as notified")
    finally:
        await conn.close()


def build_email_html(leads):
    """Build professional HTML email digest"""
    now = datetime.now(timezone.utc)

    lead_rows = ""
    for lead in leads:
        created = lead['created_at'].strftime('%I:%M %p') if lead['created_at'] else 'N/A'
        lead_rows += f"""
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 15px; color: #00CED1; font-weight: bold;">{lead['name'] or 'Not provided'}</td>
            <td style="padding: 15px;"><a href="mailto:{lead['email']}" style="color: #fff;">{lead['email']}</a></td>
            <td style="padding: 15px; color: #888;">{lead['company'] or '-'}</td>
            <td style="padding: 15px; color: #888;">{lead['phone'] or '-'}</td>
            <td style="padding: 15px; color: #666; font-size: 12px;">{lead['source']}</td>
            <td style="padding: 15px; color: #666; font-size: 12px;">{created}</td>
        </tr>
        """
        if lead['message']:
            lead_rows += f"""
            <tr style="border-bottom: 2px solid #222;">
                <td colspan="6" style="padding: 10px 15px 20px 15px; color: #aaa; font-style: italic; background: #0d0d0d;">
                    "{lead['message'][:200]}{'...' if len(lead['message'] or '') > 200 else ''}"
                </td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; background: #000; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="max-width: 900px; margin: 0 auto; padding: 30px 20px;">

            <!-- Header -->
            <div style="text-align: center; margin-bottom: 40px;">
                <h1 style="color: #00CED1; font-size: 28px; margin: 0; letter-spacing: 2px;">
                    BARRIOS A2I
                </h1>
                <p style="color: #666; margin: 10px 0 0 0; font-size: 14px;">
                    Daily Lead Intelligence Report
                </p>
            </div>

            <!-- Stats Card -->
            <div style="background: linear-gradient(135deg, #0a1628 0%, #0d0d0d 100%); border: 1px solid #00CED1; border-radius: 12px; padding: 30px; margin-bottom: 30px; text-align: center;">
                <div style="font-size: 48px; color: #00CED1; font-weight: bold;">{len(leads)}</div>
                <div style="color: #888; font-size: 16px; text-transform: uppercase; letter-spacing: 1px;">
                    New Lead{'s' if len(leads) != 1 else ''} in the Last 24 Hours
                </div>
                <div style="color: #444; font-size: 12px; margin-top: 10px;">
                    {now.strftime('%B %d, %Y')} | 5:00 AM EST
                </div>
            </div>

            <!-- Leads Table -->
            <div style="background: #111; border-radius: 12px; overflow: hidden; border: 1px solid #222;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #1a1a1a;">
                            <th style="padding: 15px; text-align: left; color: #00CED1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Name</th>
                            <th style="padding: 15px; text-align: left; color: #00CED1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Email</th>
                            <th style="padding: 15px; text-align: left; color: #00CED1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Company</th>
                            <th style="padding: 15px; text-align: left; color: #00CED1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Phone</th>
                            <th style="padding: 15px; text-align: left; color: #00CED1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Source</th>
                            <th style="padding: 15px; text-align: left; color: #00CED1; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {lead_rows}
                    </tbody>
                </table>
            </div>

            <!-- Action Tips -->
            <div style="background: #0a1628; border: 1px solid #1a3a5c; border-radius: 12px; padding: 25px; margin-top: 30px;">
                <h3 style="color: #00CED1; margin: 0 0 15px 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">
                    Quick Response Protocol
                </h3>
                <ul style="color: #888; margin: 0; padding-left: 20px; line-height: 1.8;">
                    <li>Respond within <strong style="color: #00CED1;">5 minutes</strong> for 21x higher conversion</li>
                    <li>Leads with company names are typically higher value</li>
                    <li>Check message content for specific pain points to address</li>
                </ul>
            </div>

            <!-- Footer -->
            <div style="text-align: center; margin-top: 40px; padding-top: 30px; border-top: 1px solid #222;">
                <p style="color: #444; font-size: 12px; margin: 0;">
                    Barrios A2I | Alienation 2 Innovation<br>
                    Automated by NEXUS Brain
                </p>
            </div>

        </div>
    </body>
    </html>
    """
    return html


def build_no_leads_html():
    """Email when there are no new leads"""
    now = datetime.now(timezone.utc)
    return f"""
    <!DOCTYPE html>
    <html>
    <body style="margin: 0; padding: 40px; background: #000; color: #fff; font-family: sans-serif; text-align: center;">
        <h1 style="color: #00CED1;">BARRIOS A2I</h1>
        <p style="color: #666;">Daily Lead Report | {now.strftime('%B %d, %Y')}</p>
        <div style="background: #111; padding: 40px; border-radius: 12px; margin: 30px auto; max-width: 400px;">
            <p style="font-size: 24px; margin: 0;">No new leads</p>
            <p style="color: #888; margin: 15px 0 0 0;">No new leads in the last 24 hours</p>
        </div>
        <p style="color: #444; font-size: 12px;">Time to boost that marketing!</p>
    </body>
    </html>
    """


async def send_email(subject: str, html_content: str):
    """Send email via SendGrid"""
    api_key = os.environ.get('SENDGRID_API_KEY')
    if not api_key:
        logger.error("SENDGRID_API_KEY not set")
        return False

    sg = SendGridAPIClient(api_key)
    message = Mail(
        from_email=Email(SENDER_EMAIL, SENDER_NAME),
        to_emails=To(RECIPIENT),
        subject=subject,
        html_content=Content("text/html", html_content)
    )

    try:
        response = sg.send(message)
        logger.info(f"Email sent! Status: {response.status_code}")
        return response.status_code in [200, 201, 202]
    except Exception as e:
        logger.error(f"SendGrid error: {e}")
        return False


async def run_digest():
    """Main digest function"""
    logger.info("=" * 50)
    logger.info("Starting daily lead digest...")
    logger.info(f"Time: {datetime.now(timezone.utc).isoformat()}")

    # Get new leads
    leads = await get_new_leads()
    logger.info(f"Found {len(leads)} new leads")

    # Build email
    if leads:
        subject = f"{len(leads)} New Lead{'s' if len(leads) != 1 else ''} - {datetime.now().strftime('%b %d')}"
        html = build_email_html(leads)
    else:
        subject = f"No New Leads - {datetime.now().strftime('%b %d')}"
        html = build_no_leads_html()

    # Send email
    success = await send_email(subject, html)

    if success and leads:
        # Mark leads as notified
        lead_ids = [lead['id'] for lead in leads]
        await mark_leads_notified(lead_ids)

    logger.info("Digest complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_digest())
