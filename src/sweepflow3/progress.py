"""Small progress logger shared by data preparation and training."""
from __future__ import annotations

from functools import wraps
import os
import time

_active = []


def event(label, state="INFO", **fields):
    if os.environ.get("SWEEPFLOW_PROGRESS", "1").lower() in {"0", "off", "false"}:
        return
    values = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    print(f"[SweepFlow][rank {os.environ.get('RANK', '0')}][{label}] {state} {values}", flush=True)


class Progress:
    def __init__(self, label, total=None, *, initial=0, unit="items", eta=True,
                 visible=True, **fields):
        self.label, self.total, self.done = label, total, initial
        self.unit, self.visible, self.fields = unit, visible, fields
        self.closed = False
        self.started = self.last_output = time.monotonic()

    def __enter__(self):
        _active.append(self)
        self._emit("START")
        return self

    def _emit(self, state, **extra):
        if self.visible:
            count = f"{self.done:g}/{self.total:g}" if self.total is not None else None
            event(self.label, state, **{self.unit: count}, elapsed_s=round(time.monotonic() - self.started, 1),
                  **(self.fields | extra))
        self.last_output = time.monotonic()

    def detail(self, **fields):
        self.fields.update(fields)

    def advance(self, n=1, *, reused=False, force=False, **fields):
        self.done += n
        self.fields.update(fields)
        if force or time.monotonic() - self.last_output >= 10 or (self.total is not None and self.done >= self.total):
            self._emit("PROGRESS")

    def close(self, state="DONE", **extra):
        if self.closed:
            return
        self.closed = True
        if self in _active:
            _active.remove(self)
        self._emit(state, **extra)

    def __exit__(self, exc_type, exc, tb):
        self.close("FAILED" if exc_type else "DONE", exception=repr(exc) if exc else None)
        return False


def activity(label, **fields):
    return Progress(label, **fields)


def detail(**fields):
    if _active:
        _active[-1].detail(**fields)


def reused(**fields):
    detail(**fields)


def iterate(items, label, *, total=None, unit="items", eta=True, **fields):
    if total is None:
        try:
            total = len(items)
        except TypeError:
            pass
    with Progress(label, total, unit=unit, **fields) as progress:
        for item in items:
            yield item
            progress.advance()


def reported(label):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with activity(label):
                return function(*args, **kwargs)
        return wrapped
    return decorate
