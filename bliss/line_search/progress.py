"""Optional, parent-process progress reporting for a line search.

The renderer is deliberately isolated from the scientific calculation: ordinary
display errors disable it, while exceptions in the calculation still propagate.
"""

from __future__ import annotations

from numbers import Integral
import os
import threading
from time import monotonic


_HEARTBEAT_INTERVAL = 1.0
_PHASE_FORMAT = "{desc} | elapsed: {elapsed}"
_SIMULATION_FORMAT = (
    "{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%{postfix}"
    " | elapsed: {elapsed} | estimated time remaining: {remaining}"
)


def _load_tqdm():
    # Keep notebook discovery and stream access out of disabled runs and workers.
    from tqdm.auto import tqdm

    return tqdm


class SearchProgress:
    """A single, optional progress display owned by the creating process.

    Only the simulation phase has a known total and an estimated remaining time.
    Other phases display their label and elapsed time without a percentage.
    """

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()
        self._bar = None
        self._entered = False
        self._closed = False
        self._simulations_active = False
        self._total = None
        self._completed = 0
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._started_at = None

    def __enter__(self):
        with self._lock:
            if self._entered or self._closed:
                return self
            self._entered = True
            if not self.enabled or os.getpid() != self._owner_pid:
                self.enabled = False
                return self
            self._started_at = monotonic()
            try:
                self._bar = _load_tqdm()(
                    total=None,
                    desc="Preparing",
                    unit="sim",
                    bar_format=_PHASE_FORMAT,
                    dynamic_ncols=True,
                    mininterval=0.2,
                    miniters=1,
                    leave=True,
                )
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat,
                    name="BLiSS-progress",
                    daemon=True,
                )
                self._heartbeat_thread.start()
            except Exception:
                self._disable_locked()
        return self

    def _usable_locked(self):
        return (
            self.enabled
            and not self._closed
            and self._bar is not None
            and os.getpid() == self._owner_pid
        )

    def _disable_locked(self):
        self.enabled = False
        self._stop.set()
        bar, self._bar = self._bar, None
        if bar is not None:
            try:
                bar.close()
            except Exception:
                pass

    def _heartbeat(self):
        while not self._stop.wait(_HEARTBEAT_INTERVAL):
            with self._lock:
                if not self._usable_locked():
                    return
                try:
                    self._bar.refresh()
                except Exception:
                    self._disable_locked()
                    return

    def phase(self, label):
        """Display an uncounted phase, with a fresh elapsed-time clock."""
        with self._lock:
            if not self._usable_locked():
                return
            self._simulations_active = False
            try:
                # tqdm.reset(None) preserves the old total unless cleared first.
                self._bar.total = None
                self._bar.bar_format = _PHASE_FORMAT
                self._bar.set_description_str(str(label), refresh=False)
                self._bar.set_postfix_str("", refresh=False)
                self._bar.reset(total=None)
            except Exception:
                self._disable_locked()

    def start_simulations(self, total):
        """Start the simulation clock before worker startup and count completion."""
        with self._lock:
            if not self._usable_locked():
                return
            if not isinstance(total, Integral) or isinstance(total, bool) or total < 0:
                self._disable_locked()
                return
            self._total = int(total)
            self._completed = 0
            self._simulations_active = self._total > 0
            try:
                if self._total:
                    self._bar.bar_format = _SIMULATION_FORMAT
                    self._bar.set_description_str("Simulations", refresh=False)
                    self._bar.set_postfix_str(
                        f"pending: {self._total}", refresh=False
                    )
                    self._bar.reset(total=self._total)
                else:
                    self._bar.total = None
                    self._bar.bar_format = _PHASE_FORMAT
                    self._bar.set_description_str(
                        "Simulations: 0/0, pending: 0", refresh=False
                    )
                    self._bar.set_postfix_str("", refresh=False)
                    self._bar.reset(total=None)
            except Exception:
                self._disable_locked()

    def advance(self, count=1):
        """Record completed simulations, including calls from callback threads."""
        with self._lock:
            if not self._usable_locked() or not self._simulations_active:
                return
            if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
                self._disable_locked()
                return
            if not count:
                return
            self._completed += int(count)
            try:
                pending = max(self._total - self._completed, 0)
                self._bar.set_postfix_str(f"pending: {pending}", refresh=False)
                self._bar.update(int(count))
            except Exception:
                self._disable_locked()

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            self._stop.set()
            if (
                self.enabled
                and self._bar is not None
                and os.getpid() == self._owner_pid
            ):
                try:
                    if exc_type is None:
                        label = "Completed"
                    elif issubclass(exc_type, KeyboardInterrupt):
                        label = "Interrupted"
                    else:
                        label = "Error"
                    if self._total is not None:
                        # Uncounted phases reset tqdm's n; restore the actual
                        # count so notebook widgets agree with the final label.
                        self._bar.total = self._total
                        self._bar.n = self._completed
                        pending = max(self._total - self._completed, 0)
                        label += (
                            f" | simulations: {self._completed}/{self._total}"
                            f", pending: {pending}"
                        )
                    elapsed = max(int(monotonic() - self._started_at), 0)
                    hours, rest = divmod(elapsed, 3600)
                    minutes, seconds = divmod(rest, 60)
                    duration = f"{minutes:02d}:{seconds:02d}"
                    if hours:
                        duration = f"{hours}:{duration}"
                    label += f" | total duration: {duration}"
                    self._bar.bar_format = "{desc}"
                    self._bar.set_description_str(label, refresh=False)
                    self._bar.set_postfix_str("", refresh=False)
                    self._bar.refresh()
                    self._bar.close()
                    if exc_type is not None and hasattr(self._bar, "container"):
                        # Notebook close() infers success from n == total, even
                        # when a later fit or save failed after all simulations.
                        self._bar.display(bar_style="danger")
                except Exception:
                    self._disable_locked()
                self._bar = None
        # Never join while holding the lock needed by the heartbeat.
        heartbeat = self._heartbeat_thread
        if heartbeat is not None and heartbeat is not threading.current_thread():
            try:
                heartbeat.join(timeout=0.25)
            except Exception:
                pass
        return False
