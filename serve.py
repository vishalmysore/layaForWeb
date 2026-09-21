"""Serve the built site locally.

    python serve.py                 # serves dist/ on http://localhost:8000 with COOP/COEP headers
    python serve.py 9000            # another port
    python serve.py --no-coi        # omit the headers, to mimic GitHub Pages (the page then uses its service worker)
"""
import http.server, socketserver, sys, os
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
COI = "--no-coi" not in sys.argv
PORT = int(args[0]) if args else 8000
ROOT = Path(__file__).resolve().parent / "dist"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".onnx": "application/octet-stream", ".json": "application/json"}

    def end_headers(self):
        if COI:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not ROOT.exists():
        sys.exit("dist/ not found. Run: python scripts/build_model.py && npm ci && node scripts/prepare_site.mjs")
    os.chdir(ROOT)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Serving dist/ on http://localhost:{PORT}  (COOP/COEP headers: {'on' if COI else 'off'})  Ctrl+C to stop")
        httpd.serve_forever()
