#!/usr/bin/env python3
"""Load the built site in headless Chromium, run the model on WASM, and check it against the PyTorch reference.

    pip install playwright && python -m playwright install --with-deps chromium
    python tests/browser_smoke.py            # expects dist/ (run scripts/prepare_site.mjs first)

The site is served WITHOUT COOP/COEP headers on purpose, like GitHub Pages does, so this also proves that the
service worker in web/coi-sw.js makes the page cross-origin isolated.
"""
import json, os, socketserver, sys, threading, time, http.server, functools
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / os.environ.get("SMOKE_DIST", "dist")
VARIANT = os.environ.get("SMOKE_VARIANT", "q8e8")
MIN_AGREE_FRACTION = float(os.environ.get("SMOKE_MIN_AGREE", "0.8"))
MAX_DRIFT = float(os.environ.get("SMOKE_MAX_DRIFT", "0.15"))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".js": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".json": "application/json"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache"); super().end_headers()
    def log_message(self, *a): pass


def main():
    if not DIST.exists():
        sys.exit("dist/ missing; run scripts/prepare_site.mjs first")
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(DIST)))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page()
        logs = []
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"[:300]))
        pg.on("pageerror", lambda e: logs.append("[pageerror] " + str(e)[:300]))
        pg.goto(f"http://127.0.0.1:{port}/index.html")
        for _ in range(30):  # the service worker registers, then the page reloads once
            time.sleep(1)
            try:
                if pg.evaluate("self.crossOriginIsolated"): break
            except Exception:
                pass
        isolated = pg.evaluate("self.crossOriginIsolated")
        print("cross-origin isolated via service worker:", isolated)
        assert isolated, "service worker did not make the page cross-origin isolated"

        pg.wait_for_function("document.querySelectorAll('#variant option').length > 0 && !document.querySelector('#loadBtn').disabled", timeout=30000)
        pg.select_option("#variant", VARIANT)
        pg.select_option("#backend", "wasm")
        pg.click("#loadBtn")
        st = ""
        for _ in range(900):
            time.sleep(1)
            st = pg.inner_text("#status")
            if st.startswith("Ready") or st.startswith("Could not"): break
        print("status:", st)
        assert st.startswith("Ready"), st

        pg.click("#verifyBtn")
        v = None
        for _ in range(900):
            time.sleep(1)
            v = pg.evaluate("window.__verify || null")
            if v: break
        print("verify:", json.dumps(v))
        assert v, "comparison did not finish"
        assert v["agree"] / v["total"] >= MIN_AGREE_FRACTION, f"only {v['agree']}/{v['total']} answers match"
        assert v["worst"] <= MAX_DRIFT, f"largest probability difference {v['worst']:.3f} exceeds {MAX_DRIFT}"
        errs = [l for l in logs if l.startswith("[pageerror]")]
        assert not errs, errs
        pg.screenshot(path=str(ROOT / "build" / "smoke.png"), full_page=True) if (ROOT / "build").exists() else None
        b.close()
    print("browser smoke test passed")


if __name__ == "__main__":
    main()
