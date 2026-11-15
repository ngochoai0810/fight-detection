(() => {
    const status = document.getElementById('detectStatus');
    const button = document.getElementById('detectButton');
    const player = document.getElementById('detectResult');
    const fileInput = document.getElementById('detectFile');
    const uploadZone = document.getElementById('uploadZone');
    function updateFileName() {
      document.getElementById('uploadFileName').textContent = fileInput.files[0]?.name || 'Chọn hoặc kéo video vào đây';
    }
    fileInput.addEventListener('change', updateFileName);
    uploadZone.addEventListener('dragover', event => {event.preventDefault(); uploadZone.classList.add('dragging');});
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragging'));
    uploadZone.addEventListener('drop', event => {
      event.preventDefault(); uploadZone.classList.remove('dragging');
      if (event.dataTransfer.files.length) {
        const transfer = new DataTransfer(); transfer.items.add(event.dataTransfer.files[0]);
        fileInput.files = transfer.files; updateFileName();
      }
    });
    function setStage(state) {
      const active = state === 'completed' ? 2 : state === 'running' ? 1 : 0;
      ['stepUpload','stepAnalyze','stepReview'].forEach((id, index) => {
        const step = document.getElementById(id);
        step.classList.toggle('current', index === active); step.classList.toggle('done', index < active);
        if (index === active) step.setAttribute('aria-current','step'); else step.removeAttribute('aria-current');
      });
      status.dataset.state = state;
      button.setAttribute('aria-busy', ['running','uploading'].includes(state) ? 'true' : 'false');
    }
    let timer;
    async function poll() {
      try {
        const response = await fetch('/api/detection', {cache:'no-store'});
        const job = await response.json();
        if (job.status === 'idle') return;
        setStage(job.status);
        button.disabled = ['uploading','running'].includes(job.status);
        status.textContent = ['running','uploading'].includes(job.status) ? 'AI đang quan sát… Các cảnh báo sẽ được cập nhật bên dưới.' :
          job.status === 'completed' ? 'Đã phân tích xong. Xem video kết quả và các clip cảnh báo bên dưới.' :
          'Phân tích thất bại: ' + (job.error || 'Lỗi không xác định');
        if (job.status === 'completed' && player.dataset.job !== job.id) {
          player.src = job.video; player.dataset.job = job.id; player.style.display = 'block';
        }
        if (['running','uploading'].includes(job.status)) timer = setTimeout(poll, 2000);
      } catch (error) {status.textContent = 'Không kết nối được máy chủ: ' + error.message; button.disabled = false; setStage('failed');}
    }
    document.getElementById('detectForm').addEventListener('submit', async event => {
      event.preventDefault(); clearTimeout(timer); button.disabled = true;
      setStage('uploading');
      player.pause(); player.style.display = 'none'; status.textContent = 'Đang tải video lên…';
      try {
        const file = document.getElementById('detectFile').files[0];
        if (!file || file.size > 1024**3) throw new Error('Hãy chọn video tối đa 1 GB.');
        const model = document.getElementById('detectModel').value;
        const response = await fetch('/api/detection?model=' + model, {method:'POST', body:file});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Không thể bắt đầu phân tích');
        await poll();
      } catch (error) {status.textContent = error.message; button.disabled = false; setStage('failed');}
    });
    poll();
  })();
