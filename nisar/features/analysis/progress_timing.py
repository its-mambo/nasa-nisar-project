"""Coarse remaining-time estimates from measured work and same-server runs."""
from statistics import median
from time import monotonic

STAGES = ("read", "detect", "measure", "render")


class RunTiming:
    def __init__(self, pixels=1, history=(), clock=monotonic):
        self.pixels = max(1, pixels)
        self.history, self.clock = list(history), clock
        self.stage, self.stage_started = None, clock()
        self.durations = {}
        self.completed = self.total = None

    def update(self, stage, completed=None, total=None):
        now = self.clock()
        if stage != self.stage:
            if self.stage in STAGES:
                self.durations[self.stage] = max(0, now-self.stage_started)
            self.stage, self.stage_started = stage, now
        self.completed, self.total = completed, total

    def profile(self):
        durations = dict(self.durations)
        if self.stage in STAGES:
            durations[self.stage] = max(0, self.clock()-self.stage_started)
        return {"pixels":self.pixels, "durations":durations}

    def estimate(self):
        if self.stage not in STAGES:
            return None
        elapsed = max(0, self.clock()-self.stage_started)
        remaining = None
        basis = "recent runs on this server"
        # Counts are actual detector tiles or written result images, never fake percentages.
        if (self.total and self.completed is not None and 0 < self.completed < self.total
                and elapsed >= 1):
            remaining = elapsed/self.completed * (self.total-self.completed)
            basis = "work completed in this stage"

        def historical(stage):
            samples = []
            for profile in self.history:
                duration = profile["durations"].get(stage)
                if duration is None:
                    continue
                ratio = self.pixels/profile["pixels"]
                # Far larger/smaller jobs are poor comparisons: don't invent an ETA.
                if .25 <= ratio <= 4:
                    samples.append(duration*ratio)
            return median(samples) if samples else None

        if remaining is None:
            duration = historical(self.stage)
            if duration is not None and elapsed < duration:
                remaining = duration-elapsed
        if remaining is None:
            return None
        future = [historical(s) for s in STAGES[STAGES.index(self.stage)+1:]]
        whole_run = all(value is not None for value in future)
        if whole_run:
            remaining += sum(future)
        # A range reflects stage variability and different ROI content/hardware load.
        return {"scope":"analysis" if whole_run else "stage", "basis":basis,
                "lower_s":round(max(1,remaining*.6)), "upper_s":round(max(2,remaining*1.6))}
