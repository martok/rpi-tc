import time


class Stopwatch:
    """
    Simple timing class used to record time of capturing frames
    """

    def __init__(self):
        self.start_time = None
        self.duration = None

    def start(self):
        if self.start_time is not None:
            # Already started
            return
        self.running = True
        self.start_time = time.time()

    def stop(self):
        # Stop and return elapsed time
        self.duration = time.time() - self.start_time
        self.start_time = None
        return self.duration

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()