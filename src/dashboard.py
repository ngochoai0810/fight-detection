"""Surveillance & Forensic Evidence Dashboard: python src/dashboard.py (localhost only)."""
import argparse
import csv
import html
import io
import json
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from monitoring import _file_sha256_and_size, connect, review_event, utc_now
from live_camera import LiveCamera


HEX_MEDIA_RE = re.compile(r"^/([a-f0-9]{32})\.(mp4|jpg)$")
HEX_ID_RE = re.compile(r"^[a-f0-9]{32}$")
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates/dashboard.html"


def _safe_json_loads(raw, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def load_dashboard_state(root):
    root = Path(root)
    db = connect(root)
    db.row_factory = sqlite3.Row
    try:
        session_rows = db.execute("SELECT * FROM sessions ORDER BY started DESC").fetchall()
        event_rows = db.execute("SELECT * FROM events ORDER BY started DESC").fetchall()
        log_rows = db.execute("SELECT * FROM alert_logs ORDER BY id DESC LIMIT 300").fetchall()
    finally:
        db.close()

    sessions_by_id = {}
    sessions = []
    for row in session_rows:
        d = dict(row)
        cfg = _safe_json_loads(d.get("config"), {})
        summary = _safe_json_loads(d.get("summary"), {})
        item = {
            "id": d["id"],
            "started": d.get("started"),
            "ended": d.get("ended"),
            "status": d.get("status") or ("completed" if d.get("ended") else "running"),
            "source": d.get("source") or str(cfg.get("source", "0")),
            "model": d.get("model") or str(cfg.get("model", "unknown")),
            "alerts_count": d.get("alerts_count") or summary.get("alerts_count", 0),
            "config": cfg,
            "summary": summary,
        }
        sessions_by_id[item["id"]] = item
        sessions.append(item)

    events = []
    total_bytes = 0
    total_duration = 0.0
    for row in event_rows:
        d = dict(row)
        sess = sessions_by_id.get(d.get("session_id"), {})
        sess_cfg = sess.get("config", {})
        clip_name = d.get("clip") or ""
        clip_path = root / clip_name if clip_name else None
        clip_exists = bool(clip_path and clip_path.is_file())
        actual_size = clip_path.stat().st_size if clip_exists else 0
        stored_size = d.get("file_size") or actual_size
        total_bytes += actual_size

        thumb_name = d.get("thumbnail") or (d["id"] + ".jpg")
        thumb_exists = (root / thumb_name).is_file()

        peak = float(d.get("peak") or 0.0)
        duration = d.get("duration")
        if duration is not None:
            duration = float(duration)
            total_duration += duration

        severity = d.get("severity")
        if not severity:
            if peak >= 0.85 or (duration and duration >= 8.0):
                severity = "critical"
            elif peak >= 0.65:
                severity = "high"
            else:
                severity = "medium"

        threshold = d.get("threshold")
        if threshold is None and sess_cfg.get("threshold") is not None:
            threshold = float(sess_cfg["threshold"])

        events.append(
            {
                "id": d["id"],
                "session_id": d.get("session_id") or "",
                "started": d.get("started"),
                "ended": d.get("ended"),
                "clip": clip_name,
                "clip_exists": clip_exists,
                "thumbnail": thumb_name if thumb_exists else None,
                "duration": duration,
                "peak": round(peak, 4),
                "mean_score": round(float(d["mean_score"]), 4) if d.get("mean_score") is not None else round(peak, 4),
                "threshold": round(float(threshold), 4) if threshold is not None else 0.5,
                "review": d.get("review") or "unreviewed",
                "severity": severity,
                "source": d.get("source") or sess.get("source") or "camera",
                "model": d.get("model") or sess.get("model") or "m3",
                "frames": int(d.get("frames") or 0),
                "alert_frames": int(d.get("alert_frames") or 0),
                "clip_fps": float(d.get("clip_fps") or 15.0),
                "pre_roll": float(d.get("pre_roll") or 0.0),
                "peak_offset": float(d.get("peak_offset") or 0.0),
                "file_size": int(stored_size or 0),
                "size_match": (not d.get("file_size")) or (actual_size == int(d["file_size"])),
                "sha256": d.get("sha256"),
                "bbox": _safe_json_loads(d.get("bbox_json"), {"zone": None, "zone_kind": None, "persons": []}),
                "timeline": _safe_json_loads(d.get("timeline_json"), []),
                "notes": d.get("notes") or "",
                "reviewed_at": d.get("reviewed_at"),
            }
        )

    logs = []
    for row in log_rows:
        d = dict(row)
        logs.append(
            {
                "id": d["id"],
                "ts": d.get("ts"),
                "session_id": d.get("session_id"),
                "event_id": d.get("event_id"),
                "level": d.get("level") or "INFO",
                "code": d.get("code") or "LOG",
                "message": d.get("message") or "",
                "score": d.get("score"),
                "meta": _safe_json_loads(d.get("meta"), {}),
            }
        )

    completed = [e for e in events if e["duration"] is not None]
    recording = [e for e in events if e["duration"] is None]
    reviewed = [e for e in completed if e["review"] != "unreviewed"]
    true_count = sum(1 for e in reviewed if e["review"] == "true_alert")
    false_count = sum(1 for e in reviewed if e["review"] == "false_alert")
    unreviewed_count = len(completed) - len(reviewed)

    false_ratio = f"{false_count / len(reviewed):.1%}" if reviewed else "Chưa có nhãn"
    precision_ratio = f"{true_count / len(reviewed):.1%}" if reviewed else "Chưa có nhãn"

    stats = {
        "total_events": len(events),
        "completed_events": len(completed),
        "active_recording": len(recording),
        "reviewed_count": len(reviewed),
        "unreviewed_count": unreviewed_count,
        "true_count": true_count,
        "false_count": false_count,
        "false_ratio": false_ratio,
        "precision_ratio": precision_ratio,
        "critical_count": sum(1 for e in events if e["severity"] == "critical"),
        "high_count": sum(1 for e in events if e["severity"] == "high"),
        "medium_count": sum(1 for e in events if e["severity"] == "medium"),
        "total_bytes": total_bytes,
        "total_duration": round(total_duration, 1),
        "sessions_count": len(sessions),
        "logs_count": len(logs),
    }

    return {
        "generated_at": utc_now(),
        "stats": stats,
        "sessions": sessions,
        "events": events,
        "logs": logs,
    }


def build_events_csv(state):
    buf = io.StringIO()
    buf.write("\ufeff")  # UTF-8 BOM for Excel Vietnamese compatibility
    writer = csv.writer(buf)
    writer.writerow(
        [
            "event_id",
            "session_id",
            "started_utc",
            "ended_utc",
            "source",
            "model",
            "severity",
            "peak_score",
            "mean_score",
            "threshold",
            "duration_sec",
            "pre_roll_sec",
            "peak_offset_sec",
            "frames",
            "clip_fps",
            "review_status",
            "reviewed_at_utc",
            "notes",
            "clip_file",
            "file_size_bytes",
            "sha256",
        ]
    )
    for e in state["events"]:
        writer.writerow(
            [
                e["id"],
                e["session_id"],
                e["started"] or "",
                e["ended"] or "",
                e["source"],
                e["model"],
                e["severity"],
                e["peak"],
                e["mean_score"],
                e["threshold"],
                e["duration"] if e["duration"] is not None else "",
                e["pre_roll"],
                e["peak_offset"],
                e["frames"],
                e["clip_fps"],
                e["review"],
                e["reviewed_at"] or "",
                e["notes"],
                e["clip"],
                e["file_size"],
                e["sha256"] or "",
            ]
        )
    return buf.getvalue().encode("utf-8")


def render_dashboard_html(state):
    st = state["stats"]
    models_txt = ", ".join(sorted({str(s.get("model", "")) for s in state["sessions"]}))
    ssr_summary = html.escape(
        f"{st['total_events']} clip · {st['reviewed_count']} đã kiểm tra · "
        f"{st['false_count']} clip báo nhầm · Tỷ lệ: {st['false_ratio']} · Models: {models_txt}"
    )
    safe_json = json.dumps(state, ensure_ascii=False).replace("</", "<\\/")
    return (
        TEMPLATE_PATH.read_text(encoding="utf-8").replace("__SSR_SUMMARY__", ssr_summary)
        .replace("__INITIAL_JSON__", safe_json)
        .encode("utf-8")
    )


def handler(root):
    root = Path(root)
    camera = LiveCamera(root)
    job = {"status": "idle"}
    job_lock = threading.Lock()
    project = Path(__file__).resolve().parent.parent

    def run_detection(job_id, source, model):
        output = root / f"{job_id}.mp4"
        log_path = root / f"{job_id}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, "-u", str(project / "src/inference.py"),
                     "--source", str(source), "--model", model,
                     "--weights", str(project / f"results/{model}/best.pt"),
                     "--yolo-weights", str(project / "yolo11n.pt"),
                     "--output-dir", str(root.resolve()), "--save", str(output.resolve()),
                     "--headless", "--no-voice"],
                    cwd=str(project), stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
            with job_lock:
                if result.returncode == 0 and output.is_file():
                    job.update(status="completed", video=f"/{job_id}.mp4")
                else:
                    job.update(status="failed", error=log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
        except Exception as exc:
            with job_lock:
                job.update(status="failed", error=str(exc))
        finally:
            source.unlink(missing_ok=True)

    class Dashboard(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format, *args):
            # Keep terminal output quiet during routine 3s polling & video range chunks
            pass

        def _check_same_origin(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            return origin in (None, f"http://{host}", f"https://{host}")

        def _send_json(self, obj, status=200):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _serve_media(self, file_path, content_type):
            if not file_path.is_file():
                self.send_error(404)
                return
            size = file_path.stat().st_size
            range_hdr = self.headers.get("Range")
            start, end = 0, size - 1
            status = 200

            if range_hdr and range_hdr.startswith("bytes=") and size > 0:
                m = re.match(r"^bytes=(\d*)-(\d*)$", range_hdr.strip())
                if m:
                    s_str, e_str = m.groups()
                    if s_str == "" and e_str != "":
                        suffix = int(e_str)
                        start = max(0, size - suffix)
                    else:
                        start = int(s_str) if s_str else 0
                        end = int(e_str) if e_str else size - 1
                    if start >= size or start > end:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    end = min(end, size - 1)
                    status = 206

            length = end - start + 1 if size > 0 else 0
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                if length > 0:
                    with file_path.open("rb") as fh:
                        fh.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = fh.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/api/camera/start", "/api/camera/stop"):
                if not self._check_same_origin():
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= length <= 8192:
                        raise ValueError("Yêu cầu không hợp lệ")
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError("Yêu cầu không hợp lệ")
                    if parsed.path.endswith("/stop"):
                        self._send_json(camera.stop())
                        return
                    with job_lock:
                        if job["status"] in ("uploading", "running"):
                            self._send_json({"error": "Hãy chờ phân tích video hoàn tất trước khi bật camera"}, 409)
                            return
                        state = camera.start(payload.get("model", "m1"), payload.get("source", 0), payload.get("threshold"), payload.get("window_seconds", 5.0))
                    self._send_json(state, 202)
                except (ValueError, TypeError) as exc:
                    self._send_json({"error": str(exc)}, 400)
                except RuntimeError as exc:
                    self._send_json({"error": str(exc)}, 409)
                return
            if parsed.path == "/api/detection":
                if not self._check_same_origin():
                    self.send_error(403)
                    return
                model = parse_qs(parsed.query).get("model", ["m3"])[0]
                if model not in ("m1", "m2", "m3"):
                    self._send_json({"error": "Mô hình không hợp lệ"}, 400)
                    return
                if not (project / f"results/{model}/best.pt").is_file():
                    self._send_json({"error": f"Thiếu results/{model}/best.pt"}, 400)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if not 0 < length <= 1024**3:
                    self._send_json({"error": "Video phải có dung lượng từ 1 byte đến 1 GB"}, 413)
                    return
                with job_lock:
                    if camera.is_active():
                        self._send_json({"error": "Hãy tắt camera trước khi phân tích video"}, 409)
                        return
                    if job["status"] in ("uploading", "running"):
                        self._send_json({"error": "Đang phân tích một video khác"}, 409)
                        return
                    job.clear()
                    job.update(id=uuid.uuid4().hex, status="uploading")
                    job_id = job["id"]
                source = root / f"{job_id}.upload"
                try:
                    with source.open("wb") as upload:
                        remaining = length
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("Tải video chưa hoàn tất")
                            upload.write(chunk)
                            remaining -= len(chunk)
                    with job_lock:
                        job.update(status="running")
                    threading.Thread(target=run_detection, args=(job_id, source, model), daemon=True).start()
                    self._send_json({"id": job_id, "status": "running"}, 202)
                except Exception as exc:
                    source.unlink(missing_ok=True)
                    with job_lock:
                        job.update(status="failed", error=str(exc))
                    self._send_json({"error": str(exc)}, 400)
                return
            if parsed.path not in ("/review", "/api/review"):
                self.send_error(404)
                return
            if not self._check_same_origin():
                self.send_error(403)
                return
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len > 8192:
                self.send_error(413)
                return
            raw_body = self.rfile.read(content_len).decode("utf-8", errors="replace")

            if parsed.path == "/api/review":
                payload = _safe_json_loads(raw_body, None)
                if not isinstance(payload, dict):
                    self.send_error(400)
                    return
                event_id = str(payload.get("id", ""))
                label = str(payload.get("review", ""))
                notes = payload.get("notes")
                if label not in ("true_alert", "false_alert", "unreviewed") or not HEX_ID_RE.fullmatch(event_id):
                    self.send_error(400)
                    return
                review_event(root, event_id, label, notes=notes)
                self._send_json(load_dashboard_state(root))
                return

            # Classic HTML form POST /review (backward compatible)
            data = parse_qs(raw_body)
            label = data.get("review", [""])[0]
            event_id = data.get("id", [""])[0]
            notes = data.get("notes", [None])[0]
            if label not in ("true_alert", "false_alert", "unreviewed"):
                self.send_error(400)
                return
            if HEX_ID_RE.fullmatch(event_id):
                review_event(root, event_id, label, notes=notes)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path

            if route in ("/assets/dashboard.css", "/assets/detection.js", "/assets/camera.js", "/assets/evidence.js"):
                asset = ASSETS_DIR / route.rsplit("/", 1)[-1]
                content_type = "text/css; charset=utf-8" if asset.suffix == ".css" else "text/javascript; charset=utf-8"
                self._serve_media(asset, content_type)
                return

            if route == "/api/camera":
                self._send_json(camera.snapshot())
                return

            if route == "/api/camera/stream":
                if camera.snapshot()["status"] not in ("starting", "running"):
                    self.send_error(409)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                sequence = -1
                try:
                    while True:
                        next_sequence, frame, status = camera.next_frame(sequence)
                        if status in ("stopped", "stopping", "failed"):
                            break
                        if frame is not None and next_sequence != sequence:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                            self.wfile.flush()
                        sequence = next_sequence
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return

            if re.fullmatch(r"/assets/fonts/(fonts\.css|dashboard-\d+\.(?:ttf|woff2))", route):
                asset = ASSETS_DIR / "fonts" / route.rsplit("/", 1)[-1]
                content_type = {".css": "text/css; charset=utf-8", ".ttf": "font/ttf", ".woff2": "font/woff2"}[asset.suffix]
                self._serve_media(asset, content_type)
                return

            if route == "/api/detection":
                with job_lock:
                    self._send_json(dict(job))
                return

            if route == "/":
                payload = render_dashboard_html(load_dashboard_state(root))
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if route == "/api/state":
                self._send_json(load_dashboard_state(root))
                return

            if route == "/api/verify":
                qs = parse_qs(parsed.query)
                event_id = qs.get("id", [""])[0]
                if not HEX_ID_RE.fullmatch(event_id):
                    self.send_error(400)
                    return
                db = connect(root)
                row = db.execute(
                    "SELECT clip, sha256, file_size FROM events WHERE id=?", (event_id,)
                ).fetchone()
                db.close()
                if not row or not row[0]:
                    self.send_error(404)
                    return
                clip_name, expected_sha, expected_size = row
                actual_size, actual_sha = _file_sha256_and_size(root / clip_name)
                match = bool(expected_sha and actual_sha and expected_sha == actual_sha)
                self._send_json(
                    {
                        "ok": actual_sha is not None,
                        "match": match,
                        "expected": expected_sha,
                        "actual": actual_sha,
                        "expected_size": expected_size,
                        "actual_size": actual_size,
                    }
                )
                return

            if route == "/api/export.csv":
                payload = build_events_csv(load_dashboard_state(root))
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="fight_alerts_log.csv"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if route == "/api/export.json":
                self._send_json(load_dashboard_state(root))
                return

            media_match = HEX_MEDIA_RE.fullmatch(route)
            if media_match:
                _, ext = media_match.groups()
                ctype = "video/mp4" if ext == "mp4" else "image/jpeg"
                self._serve_media(root / route.lstrip("/"), ctype)
                return

            self.send_error(404)

    Dashboard.camera = camera
    return Dashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fight Detection Surveillance Dashboard")
    parser.add_argument("--output-dir", default="runs/live")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    connect(root).close()
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    dashboard_handler = handler(root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), dashboard_handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard_handler.camera.stop(wait=True)
        server.server_close()
