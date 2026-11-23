"""Bounded timestamp sampling for rolling clips, matching uniform bin centers."""
from bisect import bisect_left
from collections import deque
import math


class TemporalWindow:
    def __init__(self, steps=16, seconds=5.0):
        if steps < 1 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("steps and seconds must be positive")
        self.steps = steps
        self.seconds = seconds
        self.period = seconds / steps
        # Two retained frames per target bin limit tensor memory independently of FPS.
        self.frames = deque(maxlen=2 * steps + 2)
        self.started = None
        self.bucket = None

    def add(self, timestamp, value):
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.frames and timestamp < self.frames[-1][0]:
            raise ValueError("timestamps must be nondecreasing")
        if not self.frames or timestamp - self.frames[-1][0] >= self.seconds:
            self.frames.clear()
            self.started = timestamp
            self.bucket = None
        bucket = math.floor((timestamp - self.started) / (self.period / 2))
        if bucket == self.bucket:
            self.frames[-1] = (timestamp, value)
        else:
            self.frames.append((timestamp, value))
        self.bucket = bucket
        cutoff = timestamp - self.seconds
        while len(self.frames) > 1 and self.frames[1][0] < cutoff:
            self.frames.popleft()

    @property
    def coverage(self):
        return min(self.seconds, self.frames[-1][0] - self.started) if self.frames else 0.0

    def sample(self):
        if not self.frames or self.coverage + 1e-8 < self.seconds - self.period / 2:
            return None
        items = list(self.frames)
        times = [item[0] for item in items]
        end = times[-1] + self.period / 2
        start = end - self.seconds
        selected = []
        for i in range(self.steps):
            target = start + (i + .5) * self.period
            right = bisect_left(times, target)
            candidates = {max(0, right - 1), min(len(items) - 1, right)}
            index = min(candidates, key=lambda j: (abs(times[j] - target), j))
            selected.append(items[index][1])
        return selected
