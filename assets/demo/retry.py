"""Retry failed API requests with exponential backoff."""
import time

def retry_request(request, attempts=3):
    for attempt in range(attempts):
        try:
            return request()
        except ConnectionError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
