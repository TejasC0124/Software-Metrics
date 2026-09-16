from __future__ import annotations

import argparse
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from src.service import SPBEService


STYLE = """
:root { color-scheme: light; --ink: #172033; --muted: #64748b; --paper: #f6f8fc;
        --panel: #ffffff; --line: #dbe3ef; --blue: #2563eb; --blue-soft: #eff6ff;
        --green: #047857; --amber: #b45309; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: Arial, sans-serif; }
main { width: min(1040px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 64px; }
.eyebrow { color: var(--blue); font-weight: 700; font-size: .8rem; letter-spacing: .09em; text-transform: uppercase; }
h1 { margin: 9px 0 10px; font-size: clamp(2rem, 5vw, 3.35rem); line-height: 1.05; }
.lead { max-width: 700px; color: var(--muted); line-height: 1.6; font-size: 1.05rem; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 26px; margin-top: 26px; box-shadow: 0 8px 28px rgba(15,23,42,.05); }
.form-grid, .grid { display: grid; gap: 16px; grid-template-columns: repeat(3, 1fr); }
label { display: grid; gap: 7px; color: #334155; font-size: .9rem; font-weight: 700; }
input { width: 100%; border: 1px solid #b9c7d8; border-radius: 9px; padding: 12px; color: var(--ink); font: inherit; }
input:focus { outline: 3px solid #bfdbfe; border-color: var(--blue); }
button { margin-top: 20px; border: 0; border-radius: 9px; padding: 12px 18px; background: var(--blue); color: white; font: inherit; font-weight: 700; cursor: pointer; }
button:hover { background: #1d4ed8; }
.error { padding: 14px; border-radius: 10px; background: #fff1f2; color: #9f1239; margin-top: 20px; }
.result-title { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }
.pill { border-radius: 999px; padding: 6px 10px; background: var(--blue-soft); color: #1d4ed8; font-size: .82rem; font-weight: 700; }
.pill.success { background: #ecfdf5; color: var(--green); }
.metric { border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
.metric small { color: var(--muted); display: block; margin-bottom: 8px; }
.metric strong { font-size: 1.4rem; }
.bar { height: 8px; background: #e8eef6; border-radius: 999px; overflow: hidden; margin-top: 10px; }
.bar > span { display: block; height: 100%; background: var(--blue); border-radius: inherit; }
h2, h3 { margin: 0 0 12px; }
.focus { color: var(--muted); line-height: 1.5; }
li { margin: 9px 0; line-height: 1.45; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: .92rem; }
th, td { text-align: left; padding: 11px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 700; }
.note { color: var(--muted); line-height: 1.5; font-size: .9rem; }
@media (max-width: 720px) { main { width: min(100% - 24px, 1040px); padding-top: 26px; } .form-grid, .grid { grid-template-columns: 1fr; } .panel { padding: 19px; } }
"""


def score_card(label: str, score: float) -> str:
    percentage = round(score * 100)
    return f"""
    <div class="metric"><small>{html.escape(label)}</small><strong>{percentage}%</strong>
      <div class="bar"><span style="width:{percentage}%"></span></div></div>"""


def render_result(result: dict[str, object]) -> str:
    analysis = result["analysis"]
    intervention = result["intervention"]
    profile = result["profile"]
    is_correct = analysis["is_correct"]
    mapping = analysis["weak_mapping"] or "No weak bridge detected"
    status = "Correct response" if is_correct else "Targeted support selected"
    status_class = "success" if is_correct else ""
    cards = "".join(
        score_card(label.title(), score)
        for label, score in analysis["modality_scores"].items()
        if score is not None
    )
    bridge_rows = "".join(
        "<tr><td>{}</td><td>{}%</td><td>{}</td></tr>".format(
            html.escape(bridge.replace("_", " - ").title()),
            round(details["strength"] * 100),
            details["weak_events"],
        )
        for bridge, details in profile["bridges"].items()
    )
    activities = "".join(
        f"<li>{html.escape(activity)}</li>" for activity in intervention["activities"]
    )
    return f"""
    <section class="panel" aria-live="polite">
      <div class="result-title"><div><div class="eyebrow">Analysis complete</div>
        <h2>{html.escape(analysis['target'])} → {html.escape(analysis['response'])}</h2></div>
        <span class="pill {status_class}">{status}</span></div>
      <div class="grid">{cards}</div>
      <div style="margin-top:24px"><h3>Detected focus</h3>
        <p class="focus"><strong>{html.escape(mapping.replace('_', ' - ').title())}</strong><br>
        {html.escape(intervention['focus'])}</p></div>
      <div style="margin-top:24px"><h3>{html.escape(intervention['title'])}</h3><ol>{activities}</ol></div>
      <div style="margin-top:28px"><h3>Learner Cross-Modal Confusion Profile</h3>
        <p class="note">{profile['attempts']} recorded attempt(s) for {html.escape(result['learner_id'])}.</p>
        <table><thead><tr><th>Bridge</th><th>Current strength</th><th>Weak events</th></tr></thead>
        <tbody>{bridge_rows}</tbody></table></div>
    </section>"""


def render_page(result: dict[str, object] | None = None, error: str | None = None) -> str:
    result_html = render_result(result) if result else ""
    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>SPBE Review Demo</title>
    <style>{STYLE}</style></head><body><main>
    <div class="eyebrow">SPBE review demo</div><h1>Find the weak learning bridge.</h1>
    <p class="lead">Enter a target word and the learner's typed response. The prototype compares meaning, sound, and spelling, then recommends the next activity.</p>
    <form class="panel" method="post" action="/analyse"><div class="form-grid">
    <label>Target word<input name="target" value="said" maxlength="60" required></label>
    <label>Learner response<input name="response" value="side" maxlength="60" required></label>
    <label>Learner ID<input name="learner" value="review_demo" maxlength="60" required></label>
    </div><button type="submit">Analyse response</button>{error_html}</form>{result_html}
    <p class="note">Prototype only: it is an educational support tool, not a clinical diagnosis. Visual scoring currently uses spelling similarity; speech and trained handwriting models are planned next.</p>
    </main></body></html>"""


def create_handler(service: SPBEService):
    class SPBEHandler(BaseHTTPRequestHandler):
        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path != "/":
                self._send_html(render_page(error="Page not found."), HTTPStatus.NOT_FOUND)
                return
            self._send_html(render_page())

        def do_POST(self) -> None:
            if self.path != "/analyse":
                self._send_html(render_page(error="Page not found."), HTTPStatus.NOT_FOUND)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            form = parse_qs(body)
            try:
                result = service.analyse_attempt(
                    form.get("target", [""])[0],
                    form.get("response", [""])[0],
                    form.get("learner", [""])[0],
                )
                self._send_html(render_page(result=result))
            except ValueError as error:
                self._send_html(render_page(error=str(error)), HTTPStatus.BAD_REQUEST)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return SPBEHandler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local SPBE review demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--profile-file",
        type=Path,
        default=Path("outputs") / "learner_profiles.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    server = ThreadingHTTPServer((args.host, args.port), create_handler(SPBEService(args.profile_file)))
    print(f"SPBE review demo running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSPBE review demo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
