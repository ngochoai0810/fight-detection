(() => {
    const stream = document.getElementById('cameraStream');
    const placeholder = document.getElementById('cameraPlaceholder');
    const status = document.getElementById('cameraStatus');
    const start = document.getElementById('cameraStart');
    const stop = document.getElementById('cameraStop');
    const badge = document.getElementById('cameraBadge');
    let timer;
    function renderCamera(state) {
      const busy = ['starting','running','stopping'].includes(state.status);
      start.disabled = busy; stop.disabled = !busy || state.status === 'stopping';
      ['cameraSource','cameraModel','cameraThreshold','cameraWindow'].forEach(id => document.getElementById(id).disabled = busy);
      if (busy) {
        document.getElementById('cameraModel').value = state.model;
        document.getElementById('cameraSource').value = String(state.source);
        if (state.window_seconds != null) document.getElementById('cameraWindow').value = state.window_seconds;
      }
      const labels = {starting:'Đang khởi động',running:'Đang quan sát',stopping:'Đang dừng',stopped:'Đã tắt',failed:'Cần kiểm tra'};
      badge.textContent = state.active ? '⚑ Cảnh báo đánh nhau' : state.pending ? 'Đang xác nhận' : labels[state.status] || state.status;
      badge.className = 'badge ' + (state.active ? 'badge-critical' : state.pending ? 'badge-high' : state.status === 'running' ? 'badge-true' : 'badge-false');
      if (state.status === 'running') {
        if (!stream.getAttribute('src')) stream.src = '/api/camera/stream?t=' + Date.now();
        stream.hidden = false; placeholder.hidden = true;
        status.textContent = !state.ready ? 'Đang lấy cửa sổ M3: ' + (state.window_coverage || 0).toFixed(1) + '/' + (state.window_seconds || 5) + ' giây…' : state.active ? 'Phát hiện đánh nhau. Đang lưu clip bằng chứng.' : state.pending ? 'Điểm vượt ngưỡng. Đang chờ xác nhận cảnh báo…' : 'Đang quan sát · Ngưỡng ' + (state.threshold * 100).toFixed(1) + '%';
      } else {
        stream.removeAttribute('src'); stream.hidden = true; placeholder.hidden = false;
        status.textContent = state.status === 'failed' ? 'Không thể chạy camera: ' + state.error : state.status === 'starting' ? 'Đang tải model và mở camera…' : state.status === 'stopping' ? 'Đang dừng và hoàn tất clip bằng chứng…' : 'Camera đã tắt. Bạn có thể bắt đầu phiên mới.';
      }
      document.getElementById('cameraScore').textContent = state.status === 'running' && state.ready ? (state.score * 100).toFixed(1) + '%' : '—';
      document.getElementById('cameraFps').textContent = state.status === 'running' ? state.fps.toFixed(1) + ' FPS' : '—';
      status.dataset.state = state.status;
    }
    async function pollCamera() {
      clearTimeout(timer);
      try {
        const response = await fetch('/api/camera', {cache:'no-store'});
        if (!response.ok) throw new Error('Không đọc được trạng thái camera');
        renderCamera(await response.json());
      } catch (error) {
        stream.removeAttribute('src'); stream.hidden = true; placeholder.hidden = false;
        status.textContent = 'Mất kết nối máy chủ. Kiểm tra dashboard trước khi thử lại.';
      } finally {timer = setTimeout(pollCamera, 1000);}
    }
    async function control(action, payload) {
      clearTimeout(timer); start.disabled = true; stop.disabled = true;
      try {
        const response = await fetch('/api/camera/' + action, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload || {})});
        const state = await response.json();
        if (!response.ok) throw new Error(state.error || 'Không thực hiện được thao tác');
        renderCamera(state); timer = setTimeout(pollCamera, 500);
      } catch (error) {
        await pollCamera(); status.textContent = error.message;
      }
    }
    document.getElementById('cameraForm').addEventListener('submit', event => {
      event.preventDefault();
      const threshold = document.getElementById('cameraThreshold').value;
      control('start', {source:Number(document.getElementById('cameraSource').value),model:document.getElementById('cameraModel').value,threshold:threshold === '' ? null : Number(threshold),window_seconds:Number(document.getElementById('cameraWindow').value)});
    });
    stop.addEventListener('click', () => control('stop'));
    pollCamera();
  })();
