import time

from queue import Queue

from threading import Thread


class Timeout:

    def __init__(self, seconds=5):
        """
        This class is used to timeout tests.
        Args:
            seconds: The timeout in seconds.
        """

        self.seconds = seconds

        self._exception_queue = Queue()
        self._finished = False
        self._timeout_thread = None

    def _timeout(self):
        time.sleep(self.seconds)
        if not self._finished:
            self._exception_queue.put(True)
        else:
            self._exception_queue.put(False)

    def start(self):
        self._timeout_thread = Thread(target=self._timeout)
        self._timeout_thread.start()

    def stop(self):
        self._finished = True
        self._exception_queue.put(False)

    def check(self):
        if not self._exception_queue.empty():
            exception = self._exception_queue.get(block=False)
            if exception:
                raise TimeoutError(f"Timeout of {self.seconds} seconds exceeded.")
