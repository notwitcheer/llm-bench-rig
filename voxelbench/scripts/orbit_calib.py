import os, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
html = Path(sys.argv[1]).resolve(); px = float(sys.argv[2]); secs = float(sys.argv[3])
CHROME = os.path.expanduser("~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--use-angle=gl-egl", "--ignore-gpu-blocklist"], executable_path=CHROME,
                          chromium_sandbox=False, env={**os.environ, "LD_LIBRARY_PATH": os.path.expanduser("~/voxelbench/lib")})
    pg = b.new_context(viewport={"width": 1920, "height": 1080}).new_page()
    pg.goto(html.as_uri(), wait_until="load"); pg.wait_for_timeout(1500)
    th0 = pg.evaluate("typeof ctrl !== 'undefined' ? ctrl.theta : null")
    pg.mouse.move(300, 540); pg.mouse.down(); t0 = time.time()
    while True:
        k = min(1.0, (time.time() - t0) / secs); pg.mouse.move(300 + px * k, 540)
        if k >= 1: break
        pg.wait_for_timeout(16)
    pg.mouse.up(); pg.wait_for_timeout(1500)
    th1 = pg.evaluate("typeof ctrl !== 'undefined' ? ctrl.theta : null")
    print({"theta0": th0, "theta1": th1, "turns": (abs(th1 - th0) / 6.283185) if th0 is not None else None, "px": px, "secs": secs})
    b.close()
