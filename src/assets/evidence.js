window.__INITIAL_STATE__ = JSON.parse(document.getElementById('initialState').textContent);

(function () {
  let state = window.__INITIAL_STATE__;
  let selectedEventId = null;
  let filterReview = "all";
  let filterSeverity = "all";
  let filterModel = "all";
  let filterSearch = "";
  let filterLogLevel = "all";
  let liveSync = true;
  let rafId = null;

  const video = document.getElementById("evidenceVideo");
  const hudCanvas = document.getElementById("hudCanvas");
  const hudCtx = hudCanvas.getContext("2d");
  const tlCanvas = document.getElementById("timelineCanvas");
  const tlCtx = tlCanvas.getContext("2d");

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function fmtTime(iso) {
    if (!iso) return "--";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    return d.toLocaleString("vi-VN", { hour12: false });
  }

  function fmtBytes(bytes) {
    if (!bytes) return "0 KB";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function sevBadge(sev) {
    if (sev === "critical") return '<span class="badge badge-critical">Nghiêm trọng</span>';
    if (sev === "high") return '<span class="badge badge-high">Cao</span>';
    return '<span class="badge badge-medium">Trung bình</span>';
  }

  function revBadge(rev) {
    if (rev === "true_alert") return '<span class="badge badge-true">✓ Đánh nhau thật</span>';
    if (rev === "false_alert") return '<span class="badge badge-false">✕ Báo nhầm</span>';
    return '<span class="badge badge-unreviewed">● Chờ duyệt</span>';
  }

  function getSelectedEvent() {
    return state.events.find(e => e.id === selectedEventId) || null;
  }

  function renderKPIs() {
    const st = state.stats;
    const recBanner = document.getElementById("recBanner");
    if (st.active_recording > 0) {
      recBanner.classList.add("show");
      document.getElementById("recBannerText").textContent =
        `ĐANG GHI TRỰC TIẾP ${st.active_recording} CLIP BẰNG CHỨNG CẢNH BÁO ĐÁNH NHAU...`;
    } else {
      recBanner.classList.remove("show");
    }

    document.getElementById("kpiTotal").textContent = `${st.total_events}`;
    document.getElementById("kpiTotalSub").textContent =
      `${st.total_duration.toFixed(1)}s tổng thời lượng · ${fmtBytes(st.total_bytes)}`;

    document.getElementById("kpiUnreviewed").textContent = `${st.unreviewed_count}`;
    document.getElementById("kpiSeveritySub").textContent =
      `${st.critical_count} Nghiêm trọng · ${st.high_count} Cao · ${st.medium_count} TB`;

    document.getElementById("kpiFalseRate").textContent =
      st.reviewed_count > 0 ? st.false_ratio : "—";
    document.getElementById("kpiReviewSub").textContent =
      `${st.true_count} Đúng (${st.precision_ratio}) · ${st.false_count} Báo nhầm / ${st.reviewed_count} đã duyệt`;

    const latest = state.sessions[0];
    if (latest && latest.summary && latest.summary.pipeline_fps != null) {
      const sm = latest.summary;
      document.getElementById("kpiPerf").textContent = `${Number(sm.pipeline_fps).toFixed(1)} FPS`;
      const infMs = sm.inference_mean_ms != null ? `${Number(sm.inference_mean_ms).toFixed(1)}ms` : "--";
      document.getElementById("kpiPerfSub").textContent =
        `Mô hình ${String(latest.model).toUpperCase()} · Suy luận TB ${infMs} · ${latest.sessions_count || state.stats.sessions_count} phiên`;
    } else {
      document.getElementById("kpiPerf").textContent = "-- FPS";
      document.getElementById("kpiPerfSub").textContent = `${st.sessions_count} phiên giám sát đã ghi nhận`;
    }
  }

  function filteredEvents() {
    return state.events.filter(ev => {
      if (filterReview !== "all" && ev.review !== filterReview) return false;
      if (filterSeverity !== "all" && ev.severity !== filterSeverity) return false;
      if (filterModel !== "all" && String(ev.model).toLowerCase() !== filterModel) return false;
      if (filterSearch) {
        const q = filterSearch.toLowerCase();
        const hay = `${ev.id} ${ev.session_id} ${ev.source} ${ev.model} ${ev.notes}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  function renderAlertList() {
    const listEl = document.getElementById("alertList");
    const items = filteredEvents();
    document.getElementById("feedCountLabel").textContent = `${items.length} / ${state.events.length} clip`;

    if (!items.length) {
      listEl.innerHTML = `<div style="padding:28px 16px;text-align:center;color:var(--text-muted)">
        Không có cảnh báo nào khớp bộ lọc hiện tại.
      </div>`;
      return;
    }

    listEl.innerHTML = items.map(ev => {
      const sel = ev.id === selectedEventId ? "selected" : "";
      const z = ev.bbox && ev.bbox.zone;
      let miniZoneHtml = "";
      if (z && z.length === 4) {
        const left = Math.max(0, Math.min(95, z[0] * 100)).toFixed(1);
        const top = Math.max(0, Math.min(95, z[1] * 100)).toFixed(1);
        const width = Math.max(6, Math.min(100 - left, (z[2] - z[0]) * 100)).toFixed(1);
        const height = Math.max(8, Math.min(100 - top, (z[3] - z[1]) * 100)).toFixed(1);
        miniZoneHtml = `<div class="mini-zone" style="left:${left}%;top:${top}%;width:${width}%;height:${height}%"></div>`;
      }
      const thumbInner = ev.thumbnail
        ? `<img src="/${esc(ev.thumbnail)}" alt="snapshot" loading="lazy">${miniZoneHtml}`
        : `<div style="height:100%;display:grid;place-items:center;font-size:11px;color:var(--text-muted)">${ev.duration == null ? "● Đang ghi..." : "Clip MP4"}</div>${miniZoneHtml}`;

      const durText = ev.duration == null ? "Đang ghi..." : `${ev.duration.toFixed(1)}s`;
      return `<div class="alert-card ${sel}" data-id="${esc(ev.id)}">
        <div class="thumb-box">
          ${thumbInner}
          <span class="thumb-score">${(ev.peak * 100).toFixed(1)}%</span>
        </div>
        <div class="card-info">
          <div>
            <div class="card-top">
              <span class="card-time" title="UTC: ${esc(ev.started)}">${esc(fmtTime(ev.started))}</span>
              ${sevBadge(ev.severity)}
            </div>
            <div class="card-meta">
              Nguồn: <b>${esc(ev.source)}</b> · ${esc(String(ev.model).toUpperCase())} · Thời lượng: <b>${durText}</b>
            </div>
            ${ev.notes ? `<div style="font-size:11.5px;color:var(--cyan);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">📝 ${esc(ev.notes)}</div>` : ""}
          </div>
          <div class="card-bottom">
            ${revBadge(ev.review)}
            <span class="mono" style="font-size:11px;color:var(--text-dim)">#${esc(ev.id.slice(0, 8))}</span>
          </div>
        </div>
      </div>`;
    }).join("");

    listEl.querySelectorAll(".alert-card").forEach(card => {
      card.addEventListener("click", () => selectEvent(card.dataset.id, true));
    });
  }

  function selectEvent(eventId, loadVideoSource) {
    const prevId = selectedEventId;
    selectedEventId = eventId;
    const ev = getSelectedEvent();
    renderAlertList();

    const emptyEl = document.getElementById("viewportEmpty");
    const dlBtn = document.getElementById("btnDownloadClip");
    document.getElementById("hashStatusText").textContent = "";

    if (!ev) {
      emptyEl.style.display = "flex";
      video.removeAttribute("src");
      document.getElementById("playerHeaderId").textContent = "Chưa chọn clip";
      document.getElementById("playerSeverityBadge").innerHTML = "";
      document.getElementById("playerReviewBadge").innerHTML = "";
      clearCanvas();
      drawTimeline(null);
      return;
    }

    document.getElementById("playerHeaderId").textContent =
      `ID: ${ev.id.slice(0, 12)}… · Nguồn: ${ev.source} · Mô hình: ${String(ev.model).toUpperCase()}`;
    document.getElementById("playerSeverityBadge").innerHTML = sevBadge(ev.severity);
    document.getElementById("playerReviewBadge").innerHTML = revBadge(ev.review);

    document.getElementById("btnMarkTrue").classList.toggle("active", ev.review === "true_alert");
    document.getElementById("btnMarkFalse").classList.toggle("active", ev.review === "false_alert");
    document.getElementById("btnMarkUnrev").classList.toggle("active", ev.review === "unreviewed");

    if (prevId !== ev.id) {
      document.getElementById("inpNotes").value = ev.notes || "";
    }

    const shaShort = ev.sha256 ? `${ev.sha256.slice(0, 16)}…${ev.sha256.slice(-8)}` : "Đang tạo...";
    document.getElementById("eventMetaGrid").innerHTML = `
      <div class="meta-item"><div class="k">Nguồn / Mô hình / Ngưỡng</div><div class="v">${esc(ev.source)} · ${esc(String(ev.model).toUpperCase())} · ≥${(ev.threshold * 100).toFixed(1)}%</div></div>
      <div class="meta-item"><div class="k">Điểm Đỉnh / Trung bình</div><div class="v" style="color:var(--red);font-weight:900">${(ev.peak * 100).toFixed(2)}% / ${(ev.mean_score * 100).toFixed(2)}%</div></div>
      <div class="meta-item"><div class="k">Bắt đầu → Kết thúc (UTC)</div><div class="v">${esc(ev.started ? ev.started.slice(11, 19) : "--")} → ${esc(ev.ended ? ev.ended.slice(11, 19) : "Đang ghi")}</div></div>
      <div class="meta-item"><div class="k">Thời lượng / Đệm trước / Đỉnh</div><div class="v">${ev.duration != null ? ev.duration.toFixed(2) + "s" : "--"} · Pre ${ev.pre_roll.toFixed(1)}s · Peak @${ev.peak_offset.toFixed(2)}s</div></div>
      <div class="meta-item"><div class="k">Số Frame / FPS / Dung lượng</div><div class="v">${ev.frames} frames @ ${ev.clip_fps.toFixed(1)} FPS · ${fmtBytes(ev.file_size)}</div></div>
      <div class="meta-item"><div class="k">Mã băm toàn vẹn SHA-256</div><div class="v" title="${esc(ev.sha256 || "")}">${esc(shaShort)}</div></div>
    `;

    if (ev.duration == null || !ev.clip_exists) {
      emptyEl.style.display = "flex";
      emptyEl.innerHTML = `<div style="font-size:28px;margin-bottom:8px">⏺</div>
        <div style="font-weight:800;color:var(--red)">Clip bằng chứng đang được ghi…</div>
        <div style="font-size:12.5px;margin-top:4px">Video sẽ phát được ngay sau khi đợt cảnh báo kết thúc hoặc đóng phiên.</div>`;
      dlBtn.style.pointerEvents = "none";
      dlBtn.style.opacity = "0.5";
    } else {
      emptyEl.style.display = "none";
      dlBtn.style.pointerEvents = "auto";
      dlBtn.style.opacity = "1";
      dlBtn.href = "/" + encodeURIComponent(ev.clip);
      if (loadVideoSource && video.getAttribute("data-clip") !== ev.clip) {
        video.setAttribute("data-clip", ev.clip);
        video.src = "/" + encodeURIComponent(ev.clip);
        video.playbackRate = parseFloat(document.getElementById("selSpeed").value || "1");
      }
    }
    drawHudOverlay();
    drawTimeline(ev);
  }

  /* Interpolate detection box & score at current video playback timestamp */
  function sampleTelemetry(ev, currentTime) {
    const tl = ev.timeline || [];
    if (!tl.length) {
      return {
        t: currentTime,
        s: ev.peak,
        a: 1,
        p: 0,
        z: ev.bbox ? ev.bbox.zone : null,
        k: ev.bbox ? ev.bbox.zone_kind : null,
        b: ev.bbox && ev.bbox.persons
          ? ev.bbox.persons.map(p => [...p.box, p.conf || 1, p.involved ? 1 : 0])
          : []
      };
    }
    if (currentTime <= tl[0].t) return tl[0];
    if (currentTime >= tl[tl.length - 1].t) return tl[tl.length - 1];

    let lo = 0, hi = tl.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (tl[mid].t <= currentTime) lo = mid;
      else hi = mid;
    }
    const p0 = tl[lo], p1 = tl[hi];
    const span = Math.max(0.001, p1.t - p0.t);
    const alpha = Math.max(0, Math.min(1, (currentTime - p0.t) / span));

    let z = p0.z;
    if (p0.z && p1.z && p0.z.length === 4 && p1.z.length === 4 && span < 0.8) {
      z = p0.z.map((v, i) => v + (p1.z[i] - v) * alpha);
    } else if (!p0.z && p1.z && alpha > 0.5) {
      z = p1.z;
    }

    return {
      t: currentTime,
      s: p0.s + (p1.s - p0.s) * alpha,
      a: alpha < 0.5 ? p0.a : p1.a,
      p: alpha < 0.5 ? p0.p : p1.p,
      z: z,
      k: p0.k || p1.k,
      b: alpha < 0.5 ? (p0.b || []) : (p1.b || [])
    };
  }

  function clearCanvas() {
    hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
  }

  function videoContentRect() {
    const cw = hudCanvas.clientWidth || 1;
    const ch = hudCanvas.clientHeight || 1;
    const vw = video.videoWidth || 16;
    const vh = video.videoHeight || 9;
    const scale = Math.min(cw / vw, ch / vh);
    const rw = vw * scale;
    const rh = vh * scale;
    return { x: (cw - rw) / 2, y: (ch - rh) / 2, w: rw, h: rh };
  }

  function drawCornerBracketBox(ctx, x, y, w, h, strokeColor, lineWidth, cornerLen, fillColor, dashed) {
    if (w <= 2 || h <= 2) return;
    ctx.save();
    if (fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fillRect(x, y, w, h);
    }
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    if (dashed) ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    const cl = Math.max(8, Math.min(cornerLen, w * 0.28, h * 0.28));
    ctx.lineWidth = lineWidth + 2;
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y);
    // Top-right
    ctx.moveTo(x + w - cl, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cl);
    // Bottom-left
    ctx.moveTo(x, y + h - cl); ctx.lineTo(x, y + h); ctx.lineTo(x + cl, y + h);
    // Bottom-right
    ctx.moveTo(x + w - cl, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cl);
    // Mid-edge tactical crosshair ticks
    const tick = 5;
    ctx.moveTo(x + w / 2, y - tick); ctx.lineTo(x + w / 2, y + tick);
    ctx.moveTo(x + w / 2, y + h - tick); ctx.lineTo(x + w / 2, y + h + tick);
    ctx.moveTo(x - tick, y + h / 2); ctx.lineTo(x + tick, y + h / 2);
    ctx.moveTo(x + w - tick, y + h / 2); ctx.lineTo(x + w + tick, y + h / 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawTag(ctx, x, y, text, bgColor, borderColor) {
    ctx.save();
    ctx.font = "700 11.5px 'Segoe UI', system-ui, sans-serif";
    const padX = 8, h = 21;
    const tw = ctx.measureText(text).width + padX * 2;
    const tx = Math.max(4, Math.min(hudCanvas.clientWidth - tw - 4, x));
    const ty = y - h - 4 < 4 ? y + 4 : y - h - 4;
    ctx.fillStyle = bgColor;
    ctx.fillRect(tx, ty, tw, h);
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(tx, ty, tw, h);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, tx + padX, ty + 14.5);
    ctx.restore();
  }

  function drawHudOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const cw = hudCanvas.clientWidth || 1;
    const ch = hudCanvas.clientHeight || 1;
    if (hudCanvas.width !== Math.round(cw * dpr) || hudCanvas.height !== Math.round(ch * dpr)) {
      hudCanvas.width = Math.round(cw * dpr);
      hudCanvas.height = Math.round(ch * dpr);
    }
    hudCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    hudCtx.clearRect(0, 0, cw, ch);

    const ev = getSelectedEvent();
    if (!ev || ev.duration == null || !ev.clip_exists) return;

    const showZone = document.getElementById("chkZone").checked;
    const showPersons = document.getElementById("chkPersons").checked;
    const spotlight = document.getElementById("chkSpotlight").checked;

    const rect = videoContentRect();
    const tNow = video.currentTime || 0;
    const sample = sampleTelemetry(ev, tNow);
    const isAlert = sample.a === 1 || sample.s >= ev.threshold;
    const isPending = !isAlert && sample.p === 1;

    // Optional Spotlight dimming outside primary detected behavior zone
    if (spotlight && sample.z && sample.z.length === 4) {
      const zx = rect.x + sample.z[0] * rect.w;
      const zy = rect.y + sample.z[1] * rect.h;
      const zw = (sample.z[2] - sample.z[0]) * rect.w;
      const zh = (sample.z[3] - sample.z[1]) * rect.h;
      hudCtx.save();
      hudCtx.fillStyle = "rgba(3, 6, 14, 0.54)";
      hudCtx.beginPath();
      hudCtx.rect(rect.x, rect.y, rect.w, rect.h);
      hudCtx.rect(zx, zy, zw, zh);
      hudCtx.fill("evenodd");
      hudCtx.restore();
    }

    // Individual Person Detection Borders
    if (showPersons && sample.b && sample.b.length) {
      sample.b.forEach((pb, idx) => {
        const px = rect.x + pb[0] * rect.w;
        const py = rect.y + pb[1] * rect.h;
        const pw = (pb[2] - pb[0]) * rect.w;
        const ph = (pb[3] - pb[1]) * rect.h;
        const conf = pb[4] != null ? pb[4] : 1.0;
        const involved = pb[5] === 1;

        if (involved && isAlert) {
          drawCornerBracketBox(hudCtx, px, py, pw, ph, "#fb7185", 1.5, 14, "rgba(244,63,94,0.06)", false);
          drawTag(hudCtx, px, py + ph + 24, `ĐỐI TƯỢNG #${idx + 1} · ${(conf * 100).toFixed(0)}%`, "rgba(159, 18, 57, 0.88)", "#fb7185");
        } else {
          drawCornerBracketBox(hudCtx, px, py, pw, ph, "rgba(56, 189, 248, 0.85)", 1.2, 12, null, true);
          drawTag(hudCtx, px, py + ph + 24, `NGƯỜI #${idx + 1} · ${(conf * 100).toFixed(0)}%`, "rgba(12, 74, 110, 0.85)", "#38bdf8");
        }
      });
    }

    // Primary Behavior / Conflict Zone Border Line
    if (showZone && sample.z && sample.z.length === 4) {
      const zx = rect.x + sample.z[0] * rect.w;
      const zy = rect.y + sample.z[1] * rect.h;
      const zw = (sample.z[2] - sample.z[0]) * rect.w;
      const zh = (sample.z[3] - sample.z[1]) * rect.h;

      if (isAlert) {
        drawCornerBracketBox(hudCtx, zx, zy, zw, zh, "#ef4444", 2.2, 26, "rgba(239, 68, 68, 0.10)", false);
        const kindTxt = sample.k === "interaction" ? "KHOANH VÙNG ĐÁNH NHAU" : "PHÁT HIỆN HÀNH VI BẠO LỰC";
        drawTag(hudCtx, zx, zy, `⚠ ${kindTxt} · ${(sample.s * 100).toFixed(1)}%`, "rgba(185, 28, 28, 0.94)", "#fca5a5");
      } else if (isPending) {
        drawCornerBracketBox(hudCtx, zx, zy, zw, zh, "#f59e0b", 1.8, 22, "rgba(245, 158, 11, 0.07)", false);
        drawTag(hudCtx, zx, zy, `◌ NGHI VẤN XÔ XÁT · ${(sample.s * 100).toFixed(1)}%`, "rgba(180, 83, 9, 0.92)", "#fcd34d");
      } else {
        drawCornerBracketBox(hudCtx, zx, zy, zw, zh, "rgba(56, 189, 248, 0.75)", 1.4, 18, null, true);
        drawTag(hudCtx, zx, zy, `◎ VÙNG THEO DÕI · ${(sample.s * 100).toFixed(1)}%`, "rgba(15, 23, 42, 0.86)", "#38bdf8");
      }
    }

    // Top-right compact CCTV status stamp inside video area
    hudCtx.save();
    const stampW = 224, stampH = 28;
    const sx = rect.x + rect.w - stampW - 10;
    const sy = rect.y + 10;
    if (rect.w > 260 && rect.h > 80) {
      hudCtx.fillStyle = "rgba(9, 14, 25, 0.82)";
      hudCtx.fillRect(sx, sy, stampW, stampH);
      hudCtx.strokeStyle = isAlert ? "#ef4444" : "#334155";
      hudCtx.strokeRect(sx, sy, stampW, stampH);
      hudCtx.fillStyle = isAlert ? "#f87171" : "#94a3b8";
      hudCtx.font = "700 11.5px Consolas, monospace";
      const phase = tNow < ev.pre_roll ? "PRE-ROLL" : (isAlert ? "FIGHT ALERT" : "MONITOR");
      hudCtx.fillText(`${phase} | T+${tNow.toFixed(2)}s | ${(sample.s * 100).toFixed(1)}%`, sx + 9, sy + 18);
    }
    if (isAlert) {
      hudCtx.strokeStyle = "rgba(239, 68, 68, 0.85)";
      hudCtx.lineWidth = 3;
      hudCtx.strokeRect(rect.x + 1.5, rect.y + 1.5, rect.w - 3, rect.h - 3);
    }
    hudCtx.restore();
  }

  function drawTimeline(ev) {
    const dpr = window.devicePixelRatio || 1;
    const w = tlCanvas.clientWidth || 600;
    const h = tlCanvas.clientHeight || 66;
    if (tlCanvas.width !== Math.round(w * dpr) || tlCanvas.height !== Math.round(h * dpr)) {
      tlCanvas.width = Math.round(w * dpr);
      tlCanvas.height = Math.round(h * dpr);
    }
    tlCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    tlCtx.clearRect(0, 0, w, h);

    if (!ev || !ev.duration) {
      document.getElementById("timelineReadout").textContent = "0.00s / 0.00s · Điểm: --%";
      return;
    }

    const dur = Math.max(0.1, video.duration || ev.duration);
    const tl = ev.timeline || [];
    const padY = 8;
    const plotH = h - padY * 2;

    // Pre-roll shaded zone
    if (ev.pre_roll > 0) {
      const preW = Math.min(w, (ev.pre_roll / dur) * w);
      tlCtx.fillStyle = "rgba(56, 189, 248, 0.08)";
      tlCtx.fillRect(0, 0, preW, h);
    }

    // Alert active segments highlight
    for (let i = 0; i < tl.length; i++) {
      if (tl[i].a === 1) {
        const x1 = (tl[i].t / dur) * w;
        const nextT = i + 1 < tl.length ? tl[i + 1].t : dur;
        const x2 = Math.max(x1 + 2, (nextT / dur) * w);
        tlCtx.fillStyle = "rgba(239, 68, 68, 0.18)";
        tlCtx.fillRect(x1, 0, x2 - x1, h);
      }
    }

    // Threshold dashed line
    const thY = padY + (1 - Math.max(0, Math.min(1, ev.threshold))) * plotH;
    tlCtx.save();
    tlCtx.strokeStyle = "#886300";
    tlCtx.setLineDash([4, 4]);
    tlCtx.beginPath();
    tlCtx.moveTo(0, thY);
    tlCtx.lineTo(w, thY);
    tlCtx.stroke();
    tlCtx.restore();

    // Score curve
    if (tl.length >= 2) {
      tlCtx.beginPath();
      tl.forEach((pt, i) => {
        const x = (pt.t / dur) * w;
        const y = padY + (1 - Math.max(0, Math.min(1, pt.s))) * plotH;
        if (i === 0) tlCtx.moveTo(x, y);
        else tlCtx.lineTo(x, y);
      });
      tlCtx.strokeStyle = "#b3382c";
      tlCtx.lineWidth = 2;
      tlCtx.stroke();
    }

    // Peak marker
    const peakX = Math.max(4, Math.min(w - 4, (ev.peak_offset / dur) * w));
    const peakY = padY + (1 - Math.max(0, Math.min(1, ev.peak))) * plotH;
    tlCtx.fillStyle = "#ef4444";
    tlCtx.beginPath();
    tlCtx.arc(peakX, peakY, 4.5, 0, Math.PI * 2);
    tlCtx.fill();
    tlCtx.strokeStyle = "#fff";
    tlCtx.lineWidth = 1.5;
    tlCtx.stroke();

    // Playhead cursor
    const curT = video.currentTime || 0;
    const curX = Math.max(0, Math.min(w, (curT / dur) * w));
    tlCtx.strokeStyle = "#27312d";
    tlCtx.lineWidth = 2;
    tlCtx.beginPath();
    tlCtx.moveTo(curX, 0);
    tlCtx.lineTo(curX, h);
    tlCtx.stroke();

    const samp = sampleTelemetry(ev, curT);
    document.getElementById("timelineReadout").textContent =
      `${curT.toFixed(2)}s / ${dur.toFixed(2)}s · Điểm hiện tại: ${(samp.s * 100).toFixed(1)}% (Ngưỡng ${(ev.threshold * 100).toFixed(1)}%)`;
  }

  function startAnimLoop() {
    if (rafId) cancelAnimationFrame(rafId);
    function tick() {
      const ev = getSelectedEvent();
      if (ev) {
        drawHudOverlay();
        drawTimeline(ev);
      }
      if (!video.paused && !video.ended) {
        rafId = requestAnimationFrame(tick);
      }
    }
    rafId = requestAnimationFrame(tick);
  }

  video.addEventListener("play", startAnimLoop);
  video.addEventListener("timeupdate", () => {
    const ev = getSelectedEvent();
    drawHudOverlay();
    drawTimeline(ev);
  });
  video.addEventListener("loadedmetadata", () => {
    const ev = getSelectedEvent();
    drawHudOverlay();
    drawTimeline(ev);
  });
  window.addEventListener("resize", () => {
    const ev = getSelectedEvent();
    drawHudOverlay();
    drawTimeline(ev);
  });

  ["chkZone", "chkPersons", "chkSpotlight"].forEach(id => {
    document.getElementById(id).addEventListener("change", drawHudOverlay);
  });

  tlCanvas.addEventListener("click", e => {
    const ev = getSelectedEvent();
    if (!ev || !ev.duration) return;
    const r = tlCanvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - r.left) / Math.max(1, r.width)));
    const dur = video.duration || ev.duration;
    video.currentTime = ratio * dur;
  });

  document.getElementById("btnPrevFrame").addEventListener("click", () => {
    const ev = getSelectedEvent();
    if (!ev) return;
    video.pause();
    video.currentTime = Math.max(0, (video.currentTime || 0) - 1 / Math.max(ev.clip_fps || 15, 1));
  });

  document.getElementById("btnNextFrame").addEventListener("click", () => {
    const ev = getSelectedEvent();
    if (!ev) return;
    video.pause();
    video.currentTime = Math.min(video.duration || ev.duration || 0, (video.currentTime || 0) + 1 / Math.max(ev.clip_fps || 15, 1));
  });

  document.getElementById("btnJumpPeak").addEventListener("click", () => {
    const ev = getSelectedEvent();
    if (!ev || !ev.clip_exists) return;
    video.currentTime = ev.peak_offset || 0;
  });

  document.getElementById("selSpeed").addEventListener("change", e => {
    video.playbackRate = parseFloat(e.target.value || "1");
  });

  document.getElementById("btnSnapshot").addEventListener("click", () => {
    const ev = getSelectedEvent();
    if (!ev || !video.videoWidth) return;
    const out = document.createElement("canvas");
    out.width = video.videoWidth;
    out.height = video.videoHeight;
    const octx = out.getContext("2d");
    octx.drawImage(video, 0, 0, out.width, out.height);
    // Project HUD overlay from video content rect onto full resolution snapshot
    const rect = videoContentRect();
    const dpr = window.devicePixelRatio || 1;
    octx.drawImage(
      hudCanvas,
      rect.x * dpr, rect.y * dpr, rect.w * dpr, rect.h * dpr,
      0, 0, out.width, out.height
    );
    const link = document.createElement("a");
    link.download = `evidence_${ev.id.slice(0, 8)}_${(video.currentTime || 0).toFixed(2)}s.png`;
    link.href = out.toDataURL("image/png");
    link.click();
  });

  async function submitReview(label, notesOnly) {
    const ev = getSelectedEvent();
    if (!ev || ev.duration == null) return;
    const payload = {
      id: ev.id,
      review: notesOnly ? ev.review : label,
      notes: document.getElementById("inpNotes").value
    };
    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      state = await res.json();
      renderAll(false);
    }
  }

  document.getElementById("btnMarkTrue").addEventListener("click", () => submitReview("true_alert", false));
  document.getElementById("btnMarkFalse").addEventListener("click", () => submitReview("false_alert", false));
  document.getElementById("btnMarkUnrev").addEventListener("click", () => submitReview("unreviewed", false));
  document.getElementById("btnSaveNotes").addEventListener("click", () => submitReview(null, true));

  document.getElementById("btnVerifyHash").addEventListener("click", async () => {
    const ev = getSelectedEvent();
    if (!ev || ev.duration == null) return;
    const el = document.getElementById("hashStatusText");
    el.style.color = "var(--text-muted)";
    el.textContent = "Đang kiểm tra SHA-256...";
    const res = await fetch(`/api/verify?id=${encodeURIComponent(ev.id)}`);
    if (!res.ok) {
      el.style.color = "var(--red)";
      el.textContent = "Không đọc được file clip!";
      return;
    }
    const data = await res.json();
    if (data.match) {
      el.style.color = "var(--emerald)";
      el.textContent = "✓ Toàn vẹn SHA-256 khớp 100%";
    } else {
      el.style.color = "var(--red)";
      el.textContent = "⚠ Cảnh báo: Hash không khớp hoặc chưa có!";
    }
  });

  document.addEventListener("keydown", e => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (e.key === "1") submitReview("true_alert", false);
    else if (e.key === "2") submitReview("false_alert", false);
    else if (e.key === "3") submitReview("unreviewed", false);
  });

  function renderTables() {
    const logsBody = document.getElementById("logsTableBody");
    const logs = state.logs.filter(l => filterLogLevel === "all" || l.level === filterLogLevel);
    if (!logs.length) {
      logsBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:20px">Chưa có bản ghi nhật ký nào.</td></tr>`;
    } else {
      logsBody.innerHTML = logs.map(l => {
        const scoreTxt = l.score != null ? `${(l.score * 100).toFixed(1)}%` : "--";
        const jumpBtn = l.event_id
          ? `<button class="btn btn-sm jump-ev" data-ev="${esc(l.event_id)}">Xem #${esc(l.event_id.slice(0, 6))}</button>`
          : `<span style="color:var(--text-dim)">--</span>`;
        return `<tr>
          <td class="mono" title="${esc(l.ts)}">${esc(fmtTime(l.ts))}</td>
          <td><span class="mono lvl-${esc(l.level)}">${esc(l.level)}</span></td>
          <td class="mono">${esc(l.code)}</td>
          <td>${esc(l.message)}</td>
          <td class="mono">${scoreTxt}</td>
          <td>${jumpBtn}</td>
        </tr>`;
      }).join("");
      logsBody.querySelectorAll(".jump-ev").forEach(b => {
        b.addEventListener("click", () => {
          selectEvent(b.dataset.ev, true);
          window.scrollTo({ top: 0, behavior: "smooth" });
        });
      });
    }

    const sessBody = document.getElementById("sessionsTableBody");
    if (!state.sessions.length) {
      sessBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:20px">Chưa có phiên giám sát nào.</td></tr>`;
    } else {
      sessBody.innerHTML = state.sessions.map(s => {
        const sm = s.summary || {};
        const cfg = s.config || {};
        const statusBadge = s.status === "completed"
          ? '<span class="badge badge-false">Đã kết thúc</span>'
          : '<span class="badge badge-true">Đang chạy</span>';
        const thTxt = cfg.threshold != null ? Number(cfg.threshold).toFixed(3) : "--";
        const holdTxt = cfg.alert_hold != null ? `${cfg.alert_hold}s` : "--";
        const fps = sm.pipeline_fps != null ? `${Number(sm.pipeline_fps).toFixed(1)} FPS` : "--";
        const inf = sm.inference_mean_ms != null
          ? `${Number(sm.inference_mean_ms).toFixed(1)}ms / ${Number(sm.inference_p95_ms || 0).toFixed(1)}ms`
          : "--";
        const proc = sm.processing_p95_ms != null ? `${Number(sm.processing_p95_ms).toFixed(1)}ms` : "--";
        return `<tr>
          <td class="mono">#${esc(s.id.slice(0, 8))}</td>
          <td><div>${esc(fmtTime(s.started))}</div><div style="margin-top:3px">${statusBadge}</div></td>
          <td><b>${esc(s.source)}</b> · <span class="mono">${esc(String(s.model).toUpperCase())}</span></td>
          <td class="mono">${thTxt} / ${holdTxt}</td>
          <td class="mono"><b>${s.alerts_count || 0}</b></td>
          <td class="mono">${fps}</td>
          <td class="mono">${inf}</td>
          <td class="mono">${proc}</td>
        </tr>`;
      }).join("");
    }
  }

  // Filters & Tabs wiring
  document.querySelectorAll("#reviewFilterTabs .tab-pill").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#reviewFilterTabs .tab-pill").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      filterReview = btn.dataset.val;
      renderAlertList();
    });
  });
  document.getElementById("selSeverity").addEventListener("change", e => {
    filterSeverity = e.target.value;
    renderAlertList();
  });
  document.getElementById("selModel").addEventListener("change", e => {
    filterModel = e.target.value;
    renderAlertList();
  });
  document.getElementById("inpSearch").addEventListener("input", e => {
    filterSearch = e.target.value.trim();
    renderAlertList();
  });
  document.getElementById("selLogLevel").addEventListener("change", e => {
    filterLogLevel = e.target.value;
    renderTables();
  });
  document.querySelectorAll("#bottomTabs .tab-pill").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#bottomTabs .tab-pill").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.getElementById("tabLogs").style.display = tab === "logs" ? "block" : "none";
      document.getElementById("tabSessions").style.display = tab === "sessions" ? "block" : "none";
    });
  });

  async function refreshState() {
    try {
      const res = await fetch("/api/state", { cache: "no-store" });
      if (!res.ok) return;
      state = await res.json();
      renderAll(false);
    } catch (_) {}
  }

  document.getElementById("btnRefresh").addEventListener("click", refreshState);
  document.getElementById("liveToggle").addEventListener("click", () => {
    liveSync = !liveSync;
    document.getElementById("liveToggle").classList.toggle("paused", !liveSync);
    document.getElementById("liveText").textContent = liveSync ? "Đồng bộ · 3 giây" : "Tạm dừng đồng bộ";
    document.getElementById("liveToggle").setAttribute("aria-pressed", String(liveSync));
  });

  setInterval(() => {
    if (liveSync) refreshState();
  }, 3000);

  function renderAll(forceLoadFirst) {
    renderKPIs();
    if (!selectedEventId && state.events.length) {
      const firstReady = state.events.find(e => e.duration != null) || state.events[0];
      selectedEventId = firstReady.id;
      forceLoadFirst = true;
    } else if (selectedEventId && !getSelectedEvent()) {
      selectedEventId = state.events.length ? state.events[0].id : null;
      forceLoadFirst = true;
    }
    renderAlertList();
    selectEvent(selectedEventId, Boolean(forceLoadFirst));
    renderTables();
  }

  renderAll(true);
})();
