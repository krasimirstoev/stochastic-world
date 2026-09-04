import json

from .storage import COMPACT_EVENTS


_EVENT_INSERT = (
    "INSERT INTO events(simulation_id,day,sequence,actor_id,target_id,event_type,success,data) "
    "VALUES(?,?,?,?,?,?,?,?)"
)


class BufferedEventMixin:
    """Buffer high-volume event rows and log lines until the daily commit."""

    def __init__(self, *args, **kwargs):
        self._event_rows = []
        self._event_log_lines = []
        super().__init__(*args, **kwargs)

    def event(self, day, sequence, event_type, actor=None, target=None, success=None, **data):
        if self.event_mode == "compact" and event_type not in COMPACT_EVENTS:
            return
        actor_id = actor.id if actor else None
        target_id = target.id if target else None
        payload = (
            json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            if data else None
        )
        self._event_rows.append((
            self.simulation_id,
            day,
            sequence,
            actor_id,
            target_id,
            event_type,
            success,
            payload,
        ))
        self._event_log_lines.append(
            f"day={day:04d} seq={sequence:09d} type={event_type} "
            f"actor={actor_id} target={target_id} data={data}\n"
        )

    def _flush_event_buffer(self):
        if self._event_rows:
            self.conn.executemany(_EVENT_INSERT, self._event_rows)
            self._event_rows.clear()
        if self._event_log_lines:
            self.log_fh.writelines(self._event_log_lines)
            self._event_log_lines.clear()

    def commit_day(self):
        self._flush_event_buffer()
        return super().commit_day()

    def finish(self, collapse_day=None):
        self._flush_event_buffer()
        return super().finish(collapse_day)
