"""Forensic evidence clip recording, alert audit logging, and session telemetry."""
import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


SESSION_EXTRA_COLUMNS = [
    ("ended", "TEXT"),
    ("status", "TEXT DEFAULT 'running'"),
    ("source", "TEXT"),
    ("model", "TEXT"),
    ("alerts_count", "INTEGER DEFAULT 0"),
]

EVENT_EXTRA_COLUMNS = [
    ("ended", "TEXT"),
    ("source", "TEXT"),
    ("model", "TEXT"),
    ("threshold", "REAL"),
    ("mean_score", "REAL"),
    ("severity", "TEXT DEFAULT 'medium'"),
    ("frames", "INTEGER DEFAULT 0"),
    ("alert_frames", "INTEGER DEFAULT 0"),
    ("clip_fps", "REAL"),
    ("pre_roll", "REAL DEFAULT 0"),
    ("peak_offset", "REAL DEFAULT 0"),
    ("file_size", "INTEGER DEFAULT 0"),
    ("sha256", "TEXT"),
    ("thumbnail", "TEXT"),
    ("bbox_json", "TEXT"),
    ("timeline_json", "TEXT"),
    ("notes", "TEXT DEFAULT ''"),
    ("reviewed_at", "TEXT"),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _ensure_columns(db, table, extra_columns):
    existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
    for col_name, col_def in extra_columns:
        if col_name not in existing:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")


def connect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "history.sqlite", timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "id TEXT PRIMARY KEY, started TEXT, config TEXT, summary TEXT)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "id TEXT PRIMARY KEY, session_id TEXT, started TEXT, clip TEXT, "
        "duration REAL, peak REAL, review TEXT DEFAULT 'unreviewed')"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS alert_logs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session_id TEXT, event_id TEXT, "
        "level TEXT, code TEXT, message TEXT, score REAL, meta TEXT)"
    )
    _ensure_columns(db, "sessions", SESSION_EXTRA_COLUMNS)
    _ensure_columns(db, "events", EVENT_EXTRA_COLUMNS)
    db.execute("CREATE INDEX IF NOT EXISTS idx_events_started ON events(started DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_logs_ts ON alert_logs(ts DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_logs_event ON alert_logs(event_id)")
    db.commit()
    return db


def append_audit_log(root, db, *, level, code, message, session_id=None, event_id=None, score=None, meta=None, ts=None):
    """Ghi dong thoi vao bang SQLite alert_logs va file bat bien alert_logs.jsonl."""
    ts = ts or utc_now()
    meta_dict = meta or {}
    meta_str = json.dumps(meta_dict, ensure_ascii=False)
    db.execute(
        "INSERT INTO alert_logs(ts, session_id, event_id, level, code, message, score, meta) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, session_id, event_id, level, code, message, score, meta_str),
    )
    jsonl_path = Path(root) / "alert_logs.jsonl"
    record = {
        "ts": ts,
        "level": level,
        "code": code,
        "session_id": session_id,
        "event_id": event_id,
        "score": round(float(score), 4) if score is not None else None,
        "message": message,
        "meta": meta_dict,
    }
    try:
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def review_event(root, event_id, label, notes=None):
    """Cap nhat nhan kiem duyet va ghi log kiem toan."""
    if label not in ("true_alert", "false_alert", "unreviewed"):
        raise ValueError(f"Invalid review label: {label}")
    root = Path(root)
    db = connect(root)
    try:
        cur = db.execute(
            "SELECT session_id, peak, review, notes FROM events WHERE id=? AND duration IS NOT NULL",
            (event_id,),
        )
        row = cur.fetchone()
        if row is None:
            return False
        session_id, peak, old_review, old_notes = row
        reviewed_at = utc_now() if label != "unreviewed" else None
        final_notes = old_notes if notes is None else str(notes).strip()[:1000]
        db.execute(
            "UPDATE events SET review=?, notes=?, reviewed_at=? WHERE id=? AND duration IS NOT NULL",
            (label, final_notes, reviewed_at, event_id),
        )
        label_vn = {
            "true_alert": "Đánh nhau thật (True Alert)",
            "false_alert": "Báo nhầm (False Alert)",
            "unreviewed": "Chưa kiểm duyệt",
        }[label]
        append_audit_log(
            root,
            db,
            level="INFO",
            code="EVENT_REVIEWED",
            message=f"Cập nhật trạng thái bằng chứng {event_id[:8]} → {label_vn}",
            session_id=session_id,
            event_id=event_id,
            score=peak,
            meta={"old_review": old_review, "new_review": label, "notes": final_notes},
        )
        db.commit()
        return True
    finally:
        db.close()


def _file_sha256_and_size(path):
    path = Path(path)
    if not path.is_file():
        return 0, None
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def open_evidence_writer(path, fps, frame_size):
    """Mo VideoWriter uu tien H.264 (avc1) de phat truc tiep tren trinh duyet HTML5."""
    w, h = int(frame_size[0]), int(frame_size[1])
    even_w = max(2, w - (w % 2))
    even_h = max(2, h - (h % 2))
    target_size = (even_w, even_h)

    candidates = []
    if os.name == "nt":
        candidates.append((cv2.CAP_MSMF, "H264"))
    candidates.extend([(None, "avc1"), (None, "mp4v")])

    for api_pref, fourcc_name in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = (
            cv2.VideoWriter(str(path), api_pref, fourcc, float(fps), target_size)
            if api_pref is not None
            else cv2.VideoWriter(str(path), fourcc, float(fps), target_size)
        )
        if writer.isOpened():
            return writer, target_size, fourcc_name
        writer.release()

    return None, target_size, None


def _fit_frame(frame, target_size):
    tw, th = target_size
    h, w = frame.shape[:2]
    if w == tw and h == th:
        return frame
    return frame[:th, :tw]


def _compact_detections(detections):
    if not detections:
        return {"zone": None, "zone_kind": None, "persons": []}
    zone = detections.get("zone")
    zone_clean = [round(float(v), 4) for v in zone[:4]] if zone and len(zone) >= 4 else None
    persons_clean = []
    for p in (detections.get("persons") or [])[:12]:
        box = p.get("box")
        if box and len(box) >= 4:
            persons_clean.append(
                {
                    "box": [round(float(v), 4) for v in box[:4]],
                    "conf": round(float(p.get("conf", 1.0)), 3),
                    "involved": bool(p.get("involved", False)),
                }
            )
    return {
        "zone": zone_clean,
        "zone_kind": detections.get("zone_kind"),
        "persons": persons_clean,
    }


def _classify_severity(peak, alert_seconds):
    if peak >= 0.85 or alert_seconds >= 5.0:
        return "critical"
    if peak >= 0.65 or alert_seconds >= 2.5:
        return "high"
    return "medium"


class Monitor:
    def __init__(self, root, config, max_clip_seconds=15, post_seconds=3, pre_seconds=2.5):
        self.root = Path(root)
        self.db = connect(root)
        self.id = uuid.uuid4().hex
        self.config = dict(config or {})
        self.source_label = str(self.config.get("source", "camera"))
        self.model_label = str(self.config.get("model", "unknown"))
        self.threshold = (
            float(self.config["threshold"])
            if self.config.get("threshold") is not None
            else None
        )
        self.start = time.perf_counter()
        self.frames = self.predictions = 0
        self.alerts_count = 0
        self.inference_total = 0.0
        self.latencies = deque(maxlen=10000)
        self.inferences = deque(maxlen=10000)
        self.frame_times = deque(maxlen=60)

        self.writer = None
        self.writer_size = None
        self.max_clip_seconds = max_clip_seconds
        self.post_seconds = post_seconds
        self.pre_seconds = pre_seconds
        self.pre_buffer = deque(maxlen=90)
        self.last_snapshot = self.start
        self.prev_alert = False
        self.prev_pending = False

        started_iso = self.utc()
        self.db.execute(
            "INSERT INTO sessions(id, started, config, summary, status, source, model, alerts_count) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                self.id,
                started_iso,
                json.dumps(self.config, ensure_ascii=False),
                "{}",
                "running",
                self.source_label,
                self.model_label,
                0,
            ),
        )
        append_audit_log(
            self.root,
            self.db,
            level="INFO",
            code="SESSION_START",
            message=f"Khởi động phiên giám sát (Mô hình: {self.model_label.upper()}, Nguồn: {self.source_label})",
            session_id=self.id,
            meta={
                "source": self.source_label,
                "model": self.model_label,
                "threshold": self.threshold,
                "device": self.config.get("device"),
            },
            ts=started_iso,
        )
        self.db.commit()

    @staticmethod
    def utc():
        return utc_now()

    def _estimate_fps(self):
        if len(self.frame_times) >= 3:
            span = self.frame_times[-1] - self.frame_times[0]
            if span >= 0.05:
                return max(2.0, min(60.0, (len(self.frame_times) - 1) / span))
        nominal = self.config.get("source_fps")
        if nominal and 1.0 <= float(nominal) <= 120.0:
            return min(60.0, float(nominal))
        return 15.0

    def _append_timeline_point(self, frame_index, score, alert, pending, det):
        t_sec = round(frame_index / max(self.clip_fps, 1.0), 3)
        # Keep timeline dense enough (~8 Hz) for smooth bounding box tracking without bloating DB
        if (
            self.clip_timeline
            and (t_sec - self.clip_timeline[-1]["t"]) < 0.11
            and self.clip_timeline[-1]["a"] == int(bool(alert))
            and score < self.peak
        ):
            return
        persons_compact = [
            [*p["box"], p["conf"], 1 if p["involved"] else 0]
            for p in det.get("persons", [])
        ]
        self.clip_timeline.append(
            {
                "t": t_sec,
                "s": round(float(score), 4),
                "a": 1 if alert else 0,
                "p": 1 if pending else 0,
                "z": det.get("zone"),
                "k": det.get("zone_kind"),
                "b": persons_compact,
            }
        )

    def _write_clip_frame(self, frame, score, alert, pending, det):
        fitted = _fit_frame(frame, self.writer_size)
        self.writer.write(fitted)
        idx = self.clip_frames
        self.clip_frames += 1
        if alert:
            self.clip_alert_frames += 1
            self.alert_score_sum += float(score)
        self.all_score_sum += float(score)

        has_boxes = bool(det.get("zone") or det.get("persons"))
        if self.peak_frame is None or score >= self.peak:
            self.peak = max(self.peak, float(score))
            self.peak_frame = fitted.copy()
            self.peak_frame_idx = idx
            if has_boxes or not (self.peak_bbox.get("zone") or self.peak_bbox.get("persons")):
                self.peak_bbox = det
        elif has_boxes and not (self.peak_bbox.get("zone") or self.peak_bbox.get("persons")):
            self.peak_bbox = det

        self._append_timeline_point(idx, score, alert, pending, det)

    def _start_clip(self, now, frame, score, det):
        self.event_id = uuid.uuid4().hex
        self.clip_name = self.event_id + ".mp4"
        self.thumb_name = self.event_id + ".jpg"
        self.clip_fps = self._estimate_fps()
        clip_path = self.root / self.clip_name
        writer, target_size, codec = open_evidence_writer(
            clip_path, self.clip_fps, (frame.shape[1], frame.shape[0])
        )
        if writer is None:
            raise RuntimeError("Khong tao duoc clip bang chung; kiem tra codec va thu muc output")

        self.writer = writer
        self.writer_size = target_size
        self.clip_codec = codec
        self.clip_start = now
        self.clip_started_iso = self.utc()
        self.clip_frames = 0
        self.clip_alert_frames = 0
        self.alert_score_sum = 0.0
        self.all_score_sum = 0.0
        self.peak = float(score)
        self.peak_frame = None
        self.peak_frame_idx = 0
        self.peak_bbox = det
        self.clip_timeline = []

        # Flush pre-roll buffer so clip captures lead-up before alert confirmation
        pre_items = [item for item in self.pre_buffer if now - item[0] <= self.pre_seconds]
        self.pre_buffer.clear()
        for _, p_frame, p_score, p_pending, p_det in pre_items:
            self._write_clip_frame(p_frame, p_score, False, p_pending, p_det)
        self.clip_pre_roll = round(len(pre_items) / max(self.clip_fps, 1.0), 3)

        self.alerts_count += 1
        self.db.execute(
            "INSERT INTO events("
            "id, session_id, started, clip, duration, peak, review, "
            "source, model, threshold, clip_fps, pre_roll"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.event_id,
                self.id,
                self.clip_started_iso,
                self.clip_name,
                None,
                float(score),
                "unreviewed",
                self.source_label,
                self.model_label,
                self.threshold,
                round(self.clip_fps, 2),
                self.clip_pre_roll,
            ),
        )
        self.db.execute(
            "UPDATE sessions SET alerts_count=? WHERE id=?",
            (self.alerts_count, self.id),
        )
        append_audit_log(
            self.root,
            self.db,
            level="CRITICAL",
            code="ALERT_CONFIRMED",
            message=(
                f"Phát hiện hành vi đánh nhau (Điểm: {score * 100:.1f}%) — "
                f"Đang ghi bằng chứng {self.clip_name}"
            ),
            session_id=self.id,
            event_id=self.event_id,
            score=score,
            meta={
                "clip": self.clip_name,
                "pre_roll_sec": self.clip_pre_roll,
                "clip_fps": round(self.clip_fps, 2),
                "codec": codec,
                "zone": det.get("zone"),
            },
            ts=self.clip_started_iso,
        )
        self.db.commit()

    def close_clip(self):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None

        ended_iso = self.utc()
        duration = round(self.clip_frames / max(self.clip_fps, 1.0), 3)
        alert_seconds = self.clip_alert_frames / max(self.clip_fps, 1.0)
        mean_score = (
            self.alert_score_sum / self.clip_alert_frames
            if self.clip_alert_frames > 0
            else self.all_score_sum / max(self.clip_frames, 1)
        )
        peak_offset = round(self.peak_frame_idx / max(self.clip_fps, 1.0), 3)
        severity = _classify_severity(self.peak, alert_seconds)

        thumb_saved = None
        if self.peak_frame is not None:
            thumb_path = self.root / self.thumb_name
            if cv2.imwrite(str(thumb_path), self.peak_frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                thumb_saved = self.thumb_name
            self.peak_frame = None

        clip_path = self.root / self.clip_name
        file_size, sha256_hex = _file_sha256_and_size(clip_path)

        self.db.execute(
            "UPDATE events SET "
            "duration=?, peak=?, ended=?, mean_score=?, severity=?, frames=?, "
            "alert_frames=?, clip_fps=?, pre_roll=?, peak_offset=?, file_size=?, "
            "sha256=?, thumbnail=?, bbox_json=?, timeline_json=? "
            "WHERE id=?",
            (
                duration,
                self.peak,
                ended_iso,
                round(mean_score, 4),
                severity,
                self.clip_frames,
                self.clip_alert_frames,
                round(self.clip_fps, 2),
                self.clip_pre_roll,
                peak_offset,
                file_size,
                sha256_hex,
                thumb_saved,
                json.dumps(self.peak_bbox, ensure_ascii=False),
                json.dumps(self.clip_timeline, separators=(",", ":")),
                self.event_id,
            ),
        )
        append_audit_log(
            self.root,
            self.db,
            level="WARN" if severity == "medium" else "CRITICAL",
            code="CLIP_SAVED",
            message=(
                f"Đã lưu clip bằng chứng {self.clip_name} "
                f"({duration:.1f}s, {self.clip_frames} frames, Peak {self.peak * 100:.1f}%)"
            ),
            session_id=self.id,
            event_id=self.event_id,
            score=self.peak,
            meta={
                "clip": self.clip_name,
                "duration": duration,
                "frames": self.clip_frames,
                "severity": severity,
                "file_size": file_size,
                "sha256": sha256_hex,
                "peak_offset": peak_offset,
            },
            ts=ended_iso,
        )
        self.db.commit()

    def record(self, frame, alert, score, inference_ms, processing_ms, detections=None, pending=False):
        now = time.perf_counter()
        self.frames += 1
        self.frame_times.append(now)
        self.latencies.append(processing_ms)
        if inference_ms is not None:
            self.inferences.append(inference_ms)
            self.inference_total += inference_ms
            self.predictions += 1

        det = _compact_detections(detections)

        if pending and not self.prev_pending and not alert:
            append_audit_log(
                self.root,
                self.db,
                level="WARN",
                code="ALERT_PENDING",
                message=f"Phát hiện dấu hiệu nghi vấn vượt ngưỡng ({score * 100:.1f}%), đang xác nhận...",
                session_id=self.id,
                score=score,
                meta={"zone": det.get("zone")},
            )
            self.db.commit()

        # Sustained alert splits cleanly into consecutive bounded segments
        if self.writer is not None and now - self.clip_start >= self.max_clip_seconds:
            self.close_clip()

        if alert:
            self.last_alert = now
            if self.writer is None:
                self._start_clip(now, frame, score, det)
        elif self.prev_alert and not alert:
            append_audit_log(
                self.root,
                self.db,
                level="INFO",
                code="ALERT_CLEARED",
                message=f"Cảnh báo kết thúc (Điểm hiện tại: {score * 100:.1f}%), tiếp tục ghi hậu kỳ {self.post_seconds}s",
                session_id=self.id,
                event_id=getattr(self, "event_id", None),
                score=score,
            )
            self.db.commit()

        if self.writer is not None:
            self._write_clip_frame(frame, score, alert, pending, det)
            if not alert and now - self.last_alert >= self.post_seconds:
                self.close_clip()
        else:
            if self.pre_seconds > 0:
                while self.pre_buffer and now - self.pre_buffer[0][0] > self.pre_seconds:
                    self.pre_buffer.popleft()
                self.pre_buffer.append((now, frame.copy(), float(score), bool(pending), det))

        self.prev_alert = bool(alert)
        self.prev_pending = bool(pending)

        if now - self.last_snapshot >= 2:
            self.snapshot()
            self.last_snapshot = now

    def snapshot(self):
        elapsed = max(time.perf_counter() - self.start, 0.001)
        summary = dict(
            elapsed_seconds=round(elapsed, 3),
            frames=self.frames,
            predictions=self.predictions,
            alerts_count=self.alerts_count,
            pipeline_fps=round(self.frames / elapsed, 2),
            predictions_per_second=round(self.predictions / elapsed, 2),
            inference_mean_ms=(
                round(self.inference_total / self.predictions, 3) if self.predictions else None
            ),
            inference_p95_ms=(
                round(float(np.percentile(self.inferences, 95)), 3) if self.inferences else None
            ),
            processing_p95_ms=(
                round(float(np.percentile(self.latencies, 95)), 3) if self.latencies else None
            ),
        )
        self.db.execute(
            "UPDATE sessions SET summary=?, alerts_count=? WHERE id=?",
            (json.dumps(summary), self.alerts_count, self.id),
        )
        self.db.commit()
        return summary

    def close(self):
        self.close_clip()
        summary = self.snapshot()
        ended_iso = self.utc()
        self.db.execute(
            "UPDATE sessions SET ended=?, status='completed' WHERE id=?",
            (ended_iso, self.id),
        )
        append_audit_log(
            self.root,
            self.db,
            level="INFO",
            code="SESSION_END",
            message=(
                f"Kết thúc phiên giám sát ({self.frames} frames, "
                f"{self.alerts_count} clip cảnh báo, {summary['pipeline_fps']:.1f} FPS)"
            ),
            session_id=self.id,
            meta=summary,
            ts=ended_iso,
        )
        self.db.commit()
        (self.root / (self.id + ".json")).write_text(
            json.dumps(dict(config=self.config, summary=summary, ended=ended_iso), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.db.close()
