"""Quick test: can we download Equibase PDFs directly without a browser?"""
import urllib.request

urls = [
    # Historical (2023)
    "https://www.equibase.com/static/chart/2023/usa/kee/20230407-usa-kee-1-d.standard.pdf",
    # Recent (2026)
    "https://www.equibase.com/static/chart/pdf/KEE040326USA1.pdf",
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

for url in urls:
    print(f"\nTesting: {url}")
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = resp.read(200)
        print(f"  Status: {resp.status}")
        print(f"  Content-Type: {resp.headers.get('Content-Type')}")
        print(f"  Size: {resp.headers.get('Content-Length', 'unknown')} bytes")
        print(f"  Is PDF: {data[:5] == b'%PDF-'}")
        print(f"  RESULT: SUCCESS")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  RESULT: FAILED")

print("\nIf both say SUCCESS, the scraper can download PDFs without a browser!")
print("If they fail, we need a different approach.")
