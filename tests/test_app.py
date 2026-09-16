import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app import create_handler
from src.service import SPBEService


class AppTestCase(unittest.TestCase):
    def test_dashboard_serves_form_and_analysis_result(self):
        with tempfile.TemporaryDirectory() as directory:
            service = SPBEService(Path(directory) / "profiles.json")
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base_url, timeout=10) as response:
                    homepage = response.read().decode("utf-8")
                request = Request(
                    f"{base_url}/analyse",
                    data=urlencode(
                        {"target": "said", "response": "side", "learner": "web_test"}
                    ).encode("utf-8"),
                    method="POST",
                )
                with urlopen(request, timeout=10) as response:
                    result_page = response.read().decode("utf-8")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=10)

        self.assertIn("Find the weak learning bridge", homepage)
        self.assertIn("Detected focus", result_page)
        self.assertIn("Meaning to sound reinforcement", result_page)


if __name__ == "__main__":
    unittest.main()
