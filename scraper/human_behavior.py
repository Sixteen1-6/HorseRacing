"""
Human-like browser behavior simulation for anti-detection.
"""

import asyncio
import random
import logging

from playwright.async_api import Page

log = logging.getLogger("scraper")


class HumanBehavior:
    """Simulates realistic human browsing patterns."""

    @staticmethod
    async def random_delay(min_s=1.5, max_s=3.0, long_pause=True):
        """Wait a random amount with occasional longer pauses."""
        if long_pause and random.random() < 0.10:
            delay = random.uniform(8.0, 15.0)
        else:
            delay = random.uniform(min_s, max_s)
        await asyncio.sleep(delay)

    @staticmethod
    async def type_like_human(page: Page, selector: str, text: str):
        """Type text with variable speed like a real person."""
        await page.click(selector)
        for char in text:
            await page.keyboard.type(char, delay=random.randint(50, 200))
            # Occasional brief pause mid-word
            if random.random() < 0.05:
                await asyncio.sleep(random.uniform(0.3, 0.8))

    @staticmethod
    async def random_scroll(page: Page):
        """Scroll the page randomly like a person reading."""
        scroll_amount = random.randint(200, 600)
        direction = random.choice([1, 1, 1, -1])  # mostly scroll down
        await page.evaluate(f"window.scrollBy(0, {scroll_amount * direction})")
        await asyncio.sleep(random.uniform(0.5, 1.5))

    @staticmethod
    async def random_mouse_move(page: Page):
        """Move mouse to a random position on the page."""
        x = random.randint(100, 1200)
        y = random.randint(100, 700)
        await page.mouse.move(x, y)
        await asyncio.sleep(random.uniform(0.1, 0.3))

    @staticmethod
    async def hover_before_click(page: Page, selector: str):
        """Hover over element briefly before clicking (human behavior)."""
        try:
            elem = page.locator(selector).first
            await elem.hover()
            await asyncio.sleep(random.uniform(0.3, 0.8))
            await elem.click()
        except Exception:
            await page.click(selector)

    @staticmethod
    async def browse_distraction(page: Page):
        """
        Occasionally visit an unrelated page (mimics tabbed browsing).
        Called randomly ~5% of the time between scrape targets.
        Saves and restores the original page URL.
        """
        distractions = [
            "https://www.google.com",
            "https://www.weather.com",
            "https://news.ycombinator.com",
        ]
        saved_url = page.url
        url = random.choice(distractions)
        log.debug(f"Distraction browse: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(random.uniform(2, 5))
        # Restore previous page so caller's state is preserved
        if saved_url and not saved_url.startswith("about:"):
            try:
                await page.goto(saved_url, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                pass
