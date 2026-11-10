/**
 * ==============================================================================
 * PUBLIC SMOKING DETECTION SYSTEM - CLIENT ENGINE
 * Handles Live MJPEG Stream, Telemetry Polling, Config Sliders, and Academic Tabs
 * ==============================================================================
 */

class SmokingDetectionApp {
  constructor() {
    this.audioAlertEnabled = false;
    this.audioContext = null;
    this.lastViolationSoundTime = 0;
    this.statusInterval = null;
    this.violationsInterval = null;

    this.initElements();
    this.initTabs();
    this.initControls();
    this.initPolling();
    this.loadResearchData();
    this.initDropzone();
  }

  initElements() {
    // Tabs
    this.tabBtns = document.querySelectorAll('.tab-btn');
    this.tabContents = document.querySelectorAll('.tab-content');

    // Video Stream
    this.videoStream = document.getElementById('videoStream');
    this.alertBanner = document.getElementById('alertBanner');
    this.sourceSelect = document.getElementById('sourceSelect');
    this.fullscreenBtn = document.getElementById('fullscreenBtn');
    this.snapshotBtn = document.getElementById('snapshotBtn');
    this.audioToggleBtn = document.getElementById('audioToggleBtn');
    this.audioIcon = document.getElementById('audioIcon');

    // Telemetry Badges & KPIs
    this.systemBadge = document.getElementById('systemBadge');
    this.fpsDisplay = document.getElementById('fpsDisplay');
    this.latencyDisplay = document.getElementById('latencyDisplay');
    this.statusKpi = document.getElementById('statusKpi');
    this.violationsTodayKpi = document.getElementById('violationsTodayKpi');
    this.violatorsNowKpi = document.getElementById('violatorsNowKpi');
    this.gpuInfoText = document.getElementById('gpuInfoText');

    // Controls
    this.confSlider = document.getElementById('confSlider');
    this.confVal = document.getElementById('confVal');
    this.iouSlider = document.getElementById('iouSlider');
    this.iouVal = document.getElementById('iouVal');
    this.spatialToggle = document.getElementById('spatialToggle');
    this.temporalToggle = document.getElementById('temporalToggle');

    // Tables & Galleries
    this.violationTableBody = document.getElementById('violationTableBody');
    this.academicGrid = document.getElementById('academicGrid');

    // Modal
    this.modal = document.getElementById('imageModal');
    this.modalImg = document.getElementById('modalImage');
    this.modalTitle = document.getElementById('modalTitle');
    this.modalClose = document.getElementById('modalClose');
  }

  initTabs() {
    this.tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        this.tabBtns.forEach(b => b.classList.remove('active'));
        this.tabContents.forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        const targetContent = document.getElementById(targetTab);
        if (targetContent) targetContent.classList.add('active');

        // Khi quay trở lại Tab Live Monitor, tự động kết nối lại luồng video
        if (targetTab === 'tabLive') {
          this.reconnectStream();
        }
      });
    });

    // Tự động kết nối lại nếu trình duyệt ngắt kết nối stream
    if (this.videoStream) {
      this.videoStream.onerror = () => {
        console.warn("Video stream gián đoạn, đang tự động kết nối lại...");
        setTimeout(() => this.reconnectStream(), 1000);
      };
    }

    // Modal close
    if (this.modalClose) {
      this.modalClose.addEventListener('click', () => {
        this.modal.classList.remove('active');
      });
    }
    if (this.modal) {
      this.modal.addEventListener('click', (e) => {
        if (e.target === this.modal) this.modal.classList.remove('active');
      });
    }
  }

  initControls() {
    // Confidence slider
    if (this.confSlider) {
      this.confSlider.addEventListener('input', (e) => {
        this.confVal.textContent = parseFloat(e.target.value).toFixed(2);
      });
      this.confSlider.addEventListener('change', () => this.syncConfig());
    }

    // IoU slider
    if (this.iouSlider) {
      this.iouSlider.addEventListener('input', (e) => {
        this.iouVal.textContent = parseFloat(e.target.value).toFixed(2);
      });
      this.iouSlider.addEventListener('change', () => this.syncConfig());
    }

    // Toggles
    if (this.spatialToggle) this.spatialToggle.addEventListener('change', () => this.syncConfig());
    if (this.temporalToggle) this.temporalToggle.addEventListener('change', () => this.syncConfig());

    // Source selector
    if (this.sourceSelect) {
      this.sourceSelect.addEventListener('change', async (e) => {
        const val = e.target.value;
        const formData = new FormData();
        formData.append('source_type', val);
        try {
          await fetch('/api/stream/source', { method: 'POST', body: formData });
          this.reconnectStream();
        } catch (err) {
          console.error("Lỗi khi chuyển đổi nguồn phát:", err);
        }
      });
    }

    // Fullscreen
    if (this.fullscreenBtn && this.videoStream) {
      this.fullscreenBtn.addEventListener('click', () => {
        const container = document.querySelector('.video-container');
        if (!document.fullscreenElement) {
          container.requestFullscreen().catch(err => alert(err.message));
        } else {
          document.exitFullscreen();
        }
      });
    }

    // Snapshot button
    if (this.snapshotBtn) {
      this.snapshotBtn.addEventListener('click', () => {
        const a = document.createElement('a');
        a.href = '/api/stream/video?' + new Date().getTime();
        a.download = `CCTV_Snapshot_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      });
    }

    // Reconnect / Reboot Camera button
    const reconnectBtn = document.getElementById('reconnectBtn');
    if (reconnectBtn) {
      reconnectBtn.addEventListener('click', async () => {
        reconnectBtn.classList.add('btn-primary');
        try {
          await fetch('/api/stream/reset', { method: 'POST' });
        } catch (e) {}
        this.reconnectStream();
        setTimeout(() => reconnectBtn.classList.remove('btn-primary'), 600);
      });
    }

    // Audio Alert Toggle
    if (this.audioToggleBtn) {
      this.audioToggleBtn.addEventListener('click', () => {
        this.audioAlertEnabled = !this.audioAlertEnabled;
        if (this.audioAlertEnabled) {
          if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
          }
          this.audioToggleBtn.classList.add('btn-primary');
          this.audioToggleBtn.querySelector('.audio-text').textContent = 'Âm Báo: BẬT';
          this.playBeep(440, 0.1);
        } else {
          this.audioToggleBtn.classList.remove('btn-primary');
          this.audioToggleBtn.querySelector('.audio-text').textContent = 'Âm Báo: TẮT';
        }
      });
    }
  }

  reconnectStream() {
    if (this.videoStream) {
      this.videoStream.src = `/api/stream/video?t=${Date.now()}`;
    }
  }

  playBeep(freq = 660, duration = 0.15) {
    if (!this.audioAlertEnabled || !this.audioContext) return;
    try {
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }
      const osc = this.audioContext.createOscillator();
      const gain = this.audioContext.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, this.audioContext.currentTime);
      gain.gain.setValueAtTime(0.08, this.audioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.audioContext.destination);
      osc.start();
      osc.stop(this.audioContext.currentTime + duration);
    } catch (e) {
      console.warn("Audio Context warn:", e);
    }
  }

  async syncConfig() {
    const payload = {
      conf_threshold: parseFloat(this.confSlider.value),
      iou_threshold: parseFloat(this.iouSlider.value),
      enable_spatial_filter: this.spatialToggle.checked,
      enable_temporal_smoothing: this.temporalToggle.checked
    };
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } catch (e) {
      console.error("Lỗi đồng bộ tham số:", e);
    }
  }

  initPolling() {
    // 1. Telemetry Polling (Mỗi 1000ms)
    this.statusInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/system/status');
        if (!res.ok) return;
        const data = await res.json();
        this.renderTelemetry(data);
      } catch (err) {
        console.warn("Mất kết nối server API:", err);
      }
    }, 1000);

    // 2. Violation Logs Polling (Mỗi 3000ms)
    this.violationsInterval = setInterval(() => {
      this.loadViolations();
    }, 3000);

    // Chạy lần đầu
    this.loadViolations();
  }

  renderTelemetry(data) {
    if (this.fpsDisplay) this.fpsDisplay.textContent = `${data.current_fps} FPS`;
    if (this.latencyDisplay) this.latencyDisplay.textContent = `${data.latency_ms} ms`;
    if (this.violationsTodayKpi) this.violationsTodayKpi.textContent = data.total_violations_today;
    if (this.violatorsNowKpi) this.violatorsNowKpi.textContent = data.violator_count;
    if (this.gpuInfoText && data.hardware !== "N/A") this.gpuInfoText.textContent = data.hardware;

    // Trạng thái cảnh báo vi phạm
    if (data.is_violation_active) {
      if (this.statusKpi) {
        this.statusKpi.textContent = "VI PHẠM!";
        this.statusKpi.className = "kpi-value kpi-status-alert";
      }
      if (this.alertBanner) this.alertBanner.classList.add('active');

      // Kêu còi báo nếu đã qua cooldown
      const now = Date.now();
      if (now - this.lastViolationSoundTime > 2500) {
        this.playBeep(780, 0.2);
        this.lastViolationSoundTime = now;
      }
    } else {
      if (this.statusKpi) {
        this.statusKpi.textContent = "AN TOÀN";
        this.statusKpi.className = "kpi-value kpi-status-normal";
      }
      if (this.alertBanner) this.alertBanner.classList.remove('active');
    }
  }

  async loadViolations() {
    try {
      const res = await fetch('/api/violations?limit=15');
      if (!res.ok) return;
      const data = await res.json();
      this.renderViolationTable(data.violations || []);
    } catch (e) {
      console.warn("Lỗi tải nhật ký vi phạm:", e);
    }
  }

  renderViolationTable(logs) {
    if (!this.violationTableBody) return;
    if (logs.length === 0) {
      this.violationTableBody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align: center; color: var(--text-dim); padding: 24px;">
            Chưa phát hiện sự kiện vi phạm nào. Hệ thống đang tuần tra giám sát.
          </td>
        </tr>`;
      return;
    }

    this.violationTableBody.innerHTML = logs.map(row => {
      const confPercent = (parseFloat(row.Confidence) * 100).toFixed(1) + '%';
      const snapUrl = `/violations/snapshots/${row.Snapshot_File}`;
      return `
        <tr>
          <td style="font-family: var(--font-mono);">${row.Timestamp.split(' ')[1] || row.Timestamp}</td>
          <td style="font-family: var(--font-mono);">${row.Camera_ID}</td>
          <td><span style="color: var(--status-alert); font-weight: 600;">Hút thuốc</span></td>
          <td style="font-family: var(--font-mono); font-weight: 600;">${confPercent}</td>
          <td>
            <img src="${snapUrl}" 
                 class="thumb-preview" 
                 alt="Bằng chứng"
                 onclick="app.openLightbox('${snapUrl}', 'Bằng chứng vi phạm: ${row.Timestamp}')" 
                 onerror="this.style.display='none'">
          </td>
        </tr>`;
    }).join('');
  }

  async loadResearchData() {
    try {
      const res = await fetch('/api/research/metrics');
      if (!res.ok) return;
      const data = await res.json();

      // Cập nhật số liệu nếu có phần tử hiển thị
      const mapEl = document.getElementById('metricMap50');
      const precEl = document.getElementById('metricPrec');
      const recEl = document.getElementById('metricRecall');
      if (mapEl) mapEl.textContent = (data.metrics.mAP50 * 100).toFixed(2) + '%';
      if (precEl) precEl.textContent = (data.metrics.precision * 100).toFixed(2) + '%';
      if (recEl) recEl.textContent = (data.metrics.recall * 100).toFixed(2) + '%';

      // Hiển thị bộ sưu tập biểu đồ khoa học
      if (this.academicGrid && data.plots && data.plots.length > 0) {
        this.academicGrid.innerHTML = data.plots.map(p => `
          <div class="chart-card">
            <img src="${p.url}" alt="${p.title}" onclick="app.openLightbox('${p.url}', '${p.title}')">
            <div class="chart-info">
              <div class="chart-title">${p.title}</div>
              <div class="chart-desc">Biểu đồ đánh giá thực nghiệm từ đợt huấn luyện 100 epochs</div>
            </div>
          </div>
        `).join('');
      }
    } catch (e) {
      console.warn("Lỗi tải thông tin nghiên cứu:", e);
    }
  }

  openLightbox(src, title) {
    if (!this.modal || !this.modalImg) return;
    this.modalImg.src = src;
    if (this.modalTitle) this.modalTitle.textContent = title || "Xem Chi Tiết";
    this.modal.classList.add('active');
  }

  initDropzone() {
    const dropzone = document.getElementById('imageDropzone');
    const fileInput = document.getElementById('imageFileInput');
    const previewContainer = document.getElementById('imageTestResult');
    const previewImg = document.getElementById('resultImage');
    const detectionsList = document.getElementById('detectionsList');

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'var(--accent-blue)';
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
      if (e.dataTransfer.files.length > 0) {
        this.processUploadedImage(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        this.processUploadedImage(e.target.files[0]);
      }
    });
  }

  async processUploadedImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const statusText = document.getElementById('uploadStatusText');
    if (statusText) statusText.textContent = "Đang phân tích hình ảnh qua YOLOv8s...";

    try {
      const res = await fetch('/api/detect/image', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (statusText) statusText.textContent = "Nhấp hoặc kéo thả ảnh vào đây để kiểm tra";

      const previewContainer = document.getElementById('imageTestResult');
      const previewImg = document.getElementById('resultImage');
      const detectionsList = document.getElementById('detectionsList');

      if (previewContainer && previewImg) {
        previewContainer.style.display = 'grid';
        previewImg.src = data.image_base64;
      }

      if (detectionsList && data.meta && data.meta.detections) {
        const dets = data.meta.detections;
        if (dets.length === 0) {
          detectionsList.innerHTML = `<p style="color: var(--text-dim);">Không phát hiện đối tượng liên quan (Không phát hiện vi phạm).</p>`;
        } else {
          detectionsList.innerHTML = dets.map(d => `
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-subtle); font-family: var(--font-mono); font-size: 12px;">
              <span style="color: ${d.class.includes('smoking') || d.class.includes('cig') ? 'var(--status-alert)' : 'var(--text-main)'}; font-weight: 600;">
                ${d.class.toUpperCase()}
              </span>
              <span>${(d.confidence * 100).toFixed(1)}%</span>
            </div>
          `).join('');
        }
      }
    } catch (err) {
      alert("Lỗi khi xử lý ảnh: " + err.message);
      if (statusText) statusText.textContent = "Nhấp hoặc kéo thả ảnh vào đây để kiểm tra";
    }
  }
}

// Khởi chạy ứng dụng
document.addEventListener('DOMContentLoaded', () => {
  window.app = new SmokingDetectionApp();
});
