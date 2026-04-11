"""Shared browser page utilities for the horse racing scraper.

Consolidates ad blocking, bot detection, and career stats parsing
that was previously duplicated across run.py, entries.py, and backfill.py.
"""

import asyncio
import re
import logging
from typing import Dict

from playwright.async_api import Page

log = logging.getLogger("scraper")


# ─── Ad/tracker blocking ─────────────────────────────────────
# Blocking these cuts Equibase page load from ~3s to ~0.7-1s.

BLOCKED_FRAGMENTS = (
    "doubleclick", "googletagmanager", "google-analytics", "googlesyndication",
    "googleadservices", "adservice.google", "criteo", "pubmatic", "rubicon",
    "adnxs", "adsrvr", "fuseplatform", "bloodhorse", "uniconsent", "amazon-adsystem",
    "gumgum", "casalemedia", "sodar", "safeframe", "ingage.tech", "media.net",
    "richaudience", "kueezrtb", "cootlogix", "lijit", "33across", "openrtb",
    "pbxai", "unrulymedia", "optable.co", "dns-finder", "adtrafficquality",
    "servenobid", "smartadserver", "hbopenbid", "pagead", "ad-delivery",
    "cmp.uniconsent", "scorecardresearch", "taboola", "outbrain", "yieldmo",
    "analytics.google", "recaptcha", "gstatic.com/recaptcha", "html-load.cc",
    "/pagead/", "/cm.g.doubleclick", "fonts.googleapis", "fonts.gstatic",
)

BLOCKED_RESOURCE_TYPES = {"image", "font", "media", "stylesheet"}


async def setup_page_blocking(page: Page):
    """Intercept and abort ads/trackers/heavy resources on this page.

    Typically provides ~3x page load speedup on Equibase pages.
    """
    async def handler(route):
        req = route.request
        try:
            if req.resource_type in BLOCKED_RESOURCE_TYPES:
                await route.abort()
                return
            if any(frag in req.url for frag in BLOCKED_FRAGMENTS):
                await route.abort()
                return
            await route.continue_()
        except Exception:
            pass
    await page.route("**/*", handler)


# ─── Bot detection ────────────────────────────────────────────

BOT_PHRASES = [
    "security check", "captcha", "i am human",
    "pardon our interruption",
]


def is_bot_blocked(text: str) -> bool:
    """Check if page text indicates bot detection."""
    text_lower = text.lower()
    return any(p in text_lower for p in BOT_PHRASES)


# ─── Career stats parsing ────────────────────────────────────

async def parse_career_stats(page: Page) -> Dict:
    """Parse career stats from a loaded Equibase horse profile page.

    Tries DOM tables first (Starts|Firsts|Seconds|Thirds|Earnings header),
    falls back to body-text regex.

    Returns dict with num_past_starts/wins/seconds/thirds, or {} on failure.
    """
    # Attempt 1: DOM tables with canonical header
    try:
        data = await page.evaluate(
            r"""
            () => {
                const tables = Array.from(document.querySelectorAll('table'))
                    .filter(t => {
                        const h = t.rows[0];
                        if (!h) return false;
                        const txt = Array.from(h.cells).map(c => c.innerText.trim()).join('|');
                        return txt === 'Starts|Firsts|Seconds|Thirds|Earnings';
                    });
                return tables.map(t =>
                    Array.from(t.rows).map(r =>
                        Array.from(r.cells).map(c => c.innerText.trim())
                    )
                );
            }
            """
        )
        # Second table is career totals; first is current-year stats
        career_row = None
        if data and len(data) >= 2 and len(data[1]) > 1:
            career_row = data[1][1]
        elif data and len(data) == 1 and len(data[0]) > 1:
            career_row = data[0][1]
        if career_row and len(career_row) >= 4:
            try:
                return {
                    "num_past_starts":  int(re.sub(r"[^\d]", "", career_row[0] or "0") or "0"),
                    "num_past_wins":    int(re.sub(r"[^\d]", "", career_row[1] or "0") or "0"),
                    "num_past_seconds": int(re.sub(r"[^\d]", "", career_row[2] or "0") or "0"),
                    "num_past_thirds":  int(re.sub(r"[^\d]", "", career_row[3] or "0") or "0"),
                }
            except ValueError:
                pass
    except Exception:
        pass

    # Attempt 2: body-text regex
    try:
        body = await page.evaluate("() => document.body.innerText || ''")
        m = re.search(
            r"CAREER\s+STATISTICS\*?[\s\S]{0,200}?"
            r"Starts\s+Firsts\s+Seconds\s+Thirds\s+Earnings\s+"
            r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
            body, re.I,
        )
        if m:
            return {
                "num_past_starts":  int(m.group(1)),
                "num_past_wins":    int(m.group(2)),
                "num_past_seconds": int(m.group(3)),
                "num_past_thirds":  int(m.group(4)),
            }
    except Exception:
        pass

    return {}


async def search_and_parse_career(page: Page, horse_name: str) -> Dict:
    """Search Equibase for a horse by name and parse career stats.

    Navigates to equibase.com homepage, submits the search form,
    handles disambiguation (picks first TB match), and parses the
    resulting profile page.

    Returns dict with num_past_starts/wins/seconds/thirds, or {} on failure.
    """
    try:
        await page.goto("https://www.equibase.com/",
                        wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(0.5)
        await page.fill("input[name='searchInput']", horse_name, timeout=5000)
        try:
            await page.click(
                "form button[type='submit'], form input[type='submit']",
                timeout=3000,
            )
        except Exception:
            await page.press("input[name='searchInput']", "Enter")
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
        await asyncio.sleep(0.6)

        # Disambiguation fallback: if we didn't land on a profile page,
        # pick the first TB horse link
        if "refno=" not in page.url:
            profile_href = await page.evaluate(
                r"""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href*="type=Horse"]'))
                        .filter(a => /refno=\d+/i.test(a.href));
                    if (!links.length) return null;
                    const tb = links.find(a => /registry=T(?:&|$)/i.test(a.href)) || links[0];
                    return tb.href;
                }
                """
            )
            if not profile_href:
                return {}
            href = profile_href.replace("&amp;", "&")
            if not href.startswith("http"):
                href = "https://www.equibase.com" + href
            await page.goto(href, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(0.6)

        return await parse_career_stats(page)
    except Exception as e:
        log.debug(f"Career search failed for {horse_name!r}: {e}")
        return {}
