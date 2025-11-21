import requests
from typing import Optional

PRINTER_IP = "<your printer IP>"   # <-- your Qidi's IP
url = f"http://{PRINTER_IP}:7125/printer/objects/query?toolhead=position"


def get_z_height(timeout: float = 3.0) -> Optional[float]:
    """Return current Z height from the printer or None on error.

    timeout: seconds to wait for the HTTP request. Returns None if the
    request fails or the response doesn't contain the expected data.
    """
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["result"]["status"]["toolhead"]["position"][2]
    except Exception:
        # Network error, timeout, bad JSON, or unexpected response shape

        return None

