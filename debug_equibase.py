"""
Quick diagnostic: visit Equibase summary URL and dump page structure.
Run with: python debug_equibase.py

This does NOT scrape data — it just shows you what the page looks like
so we can fix the parsing logic.
"""
import asyncio
import json
from playwright.async_api import async_playwright


# The exact URL the scraper would hit for KEE on 2023-04-07
SUMMARY_URL = "https://www.equibase.com/static/chart/summary/KEE040723USA-EQB.html"
CHART_INDEX_URL = "https://www.equibase.com/premium/eqbPDFChartPlusIndex.cfm?tid=KEE&dt=04/07/2023&ctry=USA"


async def dump_page(page, label):
    """Dump diagnostic info about a page."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"URL: {page.url}")
    print(f"Title: {await page.title()}")

    # Get all tables and their structure
    table_info = await page.evaluate("""
        () => {
            const tables = document.querySelectorAll('table');
            return Array.from(tables).map((t, i) => {
                const rows = Array.from(t.querySelectorAll('tr'));
                return {
                    index: i,
                    className: t.className,
                    id: t.id,
                    totalRows: rows.length,
                    totalCols: rows[0] ? rows[0].cells.length : 0,
                    headerRow: rows[0] ? Array.from(rows[0].cells).map(c => c.innerText.trim()) : [],
                    secondRow: rows[1] ? Array.from(rows[1].cells).map(c => c.innerText.trim()) : [],
                    thirdRow: rows[2] ? Array.from(rows[2].cells).map(c => c.innerText.trim()) : [],
                    parentId: t.parentElement ? t.parentElement.id : '',
                    parentClass: t.parentElement ? t.parentElement.className : '',
                };
            });
        }
    """)

    print(f"\nTables found: {len(table_info)}")
    for t in table_info:
        print(f"\n  Table #{t['index']} (class='{t['className']}', id='{t['id']}')")
        print(f"    Parent: class='{t['parentClass']}', id='{t['parentId']}'")
        print(f"    Size: {t['totalRows']} rows x {t['totalCols']} cols")
        print(f"    Header:  {t['headerRow']}")
        print(f"    Row 1:   {t['secondRow']}")
        print(f"    Row 2:   {t['thirdRow']}")

    # Get all links on page
    links = await page.evaluate("""
        () => {
            return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                href: a.href,
                text: a.innerText.trim().substring(0, 50),
            })).filter(l => l.href.includes('equibase') || l.href.includes('.pdf') ||
                           l.href.includes('chart') || l.text.toLowerCase().includes('race'));
        }
    """)

    print(f"\nRelevant links ({len(links)}):")
    for link in links[:20]:
        print(f"  [{link['text'][:40]}] → {link['href'][:100]}")

    # Get page text (first 500 chars)
    body_text = await page.inner_text("body")
    print(f"\nPage text (first 500 chars):")
    print(f"  {body_text[:500]}")

    # Check for race-related headings / divs
    race_headers = await page.evaluate(r"""
        () => {
            const elements = document.querySelectorAll('h1, h2, h3, h4, h5, .race, [class*="race"], [id*="race"]');
            return Array.from(elements).map(el => ({
                tag: el.tagName,
                class: el.className,
                id: el.id,
                text: el.innerText.trim().substring(0, 100),
            }));
        }
    """)

    print(f"\nRace-related headings/elements ({len(race_headers)}):")
    for el in race_headers[:15]:
        print(f"  <{el['tag']} class='{el['class']}' id='{el['id']}'> {el['text']}")


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)  # visible so you can see
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
        )
        page = await context.new_page()

        # Test 1: Summary URL
        print("\n" + "#"*60)
        print("  TEST 1: Summary URL")
        print("#"*60)
        try:
            await page.goto(SUMMARY_URL, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(3)
            await dump_page(page, "Summary Page")
        except Exception as e:
            print(f"  FAILED: {e}")

        # Test 2: Chart Index URL
        print("\n" + "#"*60)
        print("  TEST 2: Chart Index URL")
        print("#"*60)
        try:
            await page.goto(CHART_INDEX_URL, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(3)
            await dump_page(page, "Chart Index Page")
        except Exception as e:
            print(f"  FAILED: {e}")

        # Test 3: Click a PDF link and check what we get
        pdf_links = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                .filter(a => a.href.includes('.pdf') || a.href.includes('PDF'))
                .map(a => ({href: a.href, text: a.innerText.trim()}))
        """)

        if pdf_links:
            print("\n" + "#"*60)
            print("  TEST 3: PDF Link Check")
            print("#"*60)
            test_url = pdf_links[0]["href"]
            print(f"  Testing PDF link: {test_url}")
            try:
                response = await page.request.get(test_url, timeout=15000)
                print(f"  Status: {response.status}")
                print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")
                body = await response.body()
                print(f"  Body size: {len(body)} bytes")
                print(f"  First 50 bytes: {body[:50]}")
                is_pdf = body[:5].startswith(b"%PDF")
                print(f"  Is actual PDF: {is_pdf}")
                if not is_pdf:
                    print(f"  Body as text (first 200): {body[:200].decode('utf-8', errors='replace')}")
            except Exception as e:
                print(f"  FAILED: {e}")

        print("\n\nDone! Copy the output above and share it with Claude.")
        input("Press Enter to close browser...")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
