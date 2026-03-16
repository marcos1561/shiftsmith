class ProgressCounter:
    def __init__(self, start: int, end: int, update_freq: int = 10):
        if end < start:
            raise ValueError("End value must be greater than or equal to start value.")
        self.start = start
        self.end = end
        self.current = start
        self.update_freq = update_freq
        self.last_shown_percent = -1

    def increment(self, step: int = 1):
        self.current = min(self.current + step, self.end)

    def progress(self) -> float:
        if self.end == self.start:
            return 1.0
        return (self.current - self.start) / (self.end - self.start)

    def is_complete(self) -> bool:
        return self.current >= self.end

    def show(self):
        percent = self.progress() * 100
        # Calculate which frequency bucket we're in
        current_bucket = int(percent / self.update_freq)
        last_bucket = int(self.last_shown_percent / self.update_freq)
        
        # Only show if we've crossed a frequency threshold or reached completion
        if current_bucket > last_bucket or self.is_complete():
            print(self.__str__())
            self.last_shown_percent = percent

    def __str__(self):
        percent = self.progress() * 100
        return f"Progress: {self.current}/{self.end} ({percent:.2f}%)"