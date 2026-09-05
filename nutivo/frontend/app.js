// We are serving the frontend directly from FastAPI now!
const API_BASE = '';

// ─── Tab Navigation ───
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');

    if (tab.dataset.tab === 'products-tab') {
      loadProducts();
    }
  });
});

// ─── Toast Notifications ───
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ─── Scan Tab ───
const scanUploadZone = document.getElementById('scan-upload-zone');
const scanFileInput = document.getElementById('scan-file-input');
const scanPreview = document.getElementById('scan-preview');
const scanPreviewImg = document.getElementById('scan-preview-img');
const scanBtn = document.getElementById('scan-btn');
const scanResultsSection = document.getElementById('scan-results-section');
const annotatedImg = document.getElementById('annotated-img');
const resultsGrid = document.getElementById('results-grid');
const resultsCount = document.getElementById('results-count');
const matchedCount = document.getElementById('matched-count');
const unknownCount = document.getElementById('unknown-count');

let selectedScanFile = null;

// Drag and drop
['dragenter', 'dragover'].forEach(evt => {
  scanUploadZone.addEventListener(evt, e => {
    e.preventDefault();
    scanUploadZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(evt => {
  scanUploadZone.addEventListener(evt, e => {
    e.preventDefault();
    scanUploadZone.classList.remove('dragover');
  });
});

scanUploadZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    handleScanFile(file);
  }
});

scanFileInput.addEventListener('change', e => {
  if (e.target.files[0]) {
    handleScanFile(e.target.files[0]);
  }
});

function handleScanFile(file) {
  selectedScanFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    scanPreviewImg.src = e.target.result;
    scanPreview.classList.add('visible');
    scanBtn.disabled = false;
  };
  reader.readAsDataURL(file);
  scanResultsSection.style.display = 'none';
}

scanBtn.addEventListener('click', async () => {
  if (!selectedScanFile) return;

  scanBtn.disabled = true;
  scanBtn.innerHTML = '<span class="spinner" style="width:16px;height:16px;margin:0;border-width:2px;"></span> Scanning...';

  const formData = new FormData();
  formData.append('file', selectedScanFile);

  try {
    // Fire both requests in parallel
    const [jsonRes, imgRes] = await Promise.all([
      fetch(`${API_BASE}/scan`, { method: 'POST', body: formData }),
      fetch(`${API_BASE}/scan/annotated`, { method: 'POST', body: createFormData(selectedScanFile) })
    ]);

    if (!jsonRes.ok) throw new Error(`Scan failed: ${jsonRes.status}`);
    if (!imgRes.ok) throw new Error(`Annotated scan failed: ${imgRes.status}`);

    const results = await jsonRes.json();
    const imgBlob = await imgRes.blob();

    // Show annotated image
    annotatedImg.src = URL.createObjectURL(imgBlob);
    scanResultsSection.style.display = 'block';

    // Populate results grid
    const matched = results.filter(r => r.status === 'matched');
    const unknown = results.filter(r => r.status !== 'matched');

    resultsCount.textContent = `${results.length} products detected`;
    matchedCount.textContent = `${matched.length} matched`;
    unknownCount.textContent = `${unknown.length} unknown`;

    resultsGrid.innerHTML = '';
    results.forEach(r => {
      const isMatched = r.status === 'matched';
      const item = document.createElement('div');
      item.className = 'result-item';
      item.innerHTML = `
        <div class="result-status ${isMatched ? 'matched' : 'unknown'}"></div>
        <div class="result-info">
          <div class="result-name">${isMatched ? r.product_name : 'Unknown Product'}</div>
          <div class="result-confidence">${isMatched ? 'Matched' : 'Below threshold'}</div>
        </div>
        <div class="result-score" style="color: ${isMatched ? 'var(--success)' : 'var(--danger)'}">
          ${(r.confidence * 100).toFixed(1)}%
        </div>
      `;
      resultsGrid.appendChild(item);
    });

    showToast(`Scan complete — ${matched.length} products identified`, 'success');

  } catch (err) {
    console.error(err);
    showToast(err.message, 'error');
  } finally {
    scanBtn.disabled = false;
    scanBtn.innerHTML = '⚡ Scan Shelf';
  }
});

function createFormData(file) {
  const fd = new FormData();
  fd.append('file', file);
  return fd;
}

// ─── Products Tab ───
async function loadProducts() {
  const grid = document.getElementById('products-grid');
  const count = document.getElementById('products-count');
  grid.innerHTML = '<div class="empty-state"><div class="spinner" style="margin:0 auto 12px;"></div><p>Loading products...</p></div>';

  try {
    const res = await fetch(`${API_BASE}/products`);
    if (!res.ok) throw new Error(`Failed to load products: ${res.status}`);
    const products = await res.json();

    count.textContent = `${products.length} products in database`;

    if (products.length === 0) {
      grid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-state-icon">📦</div>
          <div class="empty-state-text">No products yet. Add some in the "Add Product" tab.</div>
        </div>`;
      return;
    }

    grid.innerHTML = '';
    products.forEach(p => {
      const card = document.createElement('div');
      card.className = 'product-card';
      card.innerHTML = `
        <div class="product-icon">📦</div>
        <div class="product-name">${p.product_name || 'Unnamed'}</div>
        <div class="product-filename">${p.filename || 'No file'}</div>
      `;
      grid.appendChild(card);
    });

  } catch (err) {
    console.error(err);
    grid.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <div class="empty-state-icon">⚠️</div>
        <div class="empty-state-text">Failed to load products. Is the API running?</div>
      </div>`;
    showToast('Failed to load products', 'error');
  }
}

// ─── Add Product Tab ───
const addUploadZone = document.getElementById('add-upload-zone');
const addFileInput = document.getElementById('add-file-input');
const addPreview = document.getElementById('add-preview');
const addPreviewImg = document.getElementById('add-preview-img');
const addBtn = document.getElementById('add-btn');
const productNameInput = document.getElementById('product-name-input');

let selectedAddFile = null;

['dragenter', 'dragover'].forEach(evt => {
  addUploadZone.addEventListener(evt, e => {
    e.preventDefault();
    addUploadZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(evt => {
  addUploadZone.addEventListener(evt, e => {
    e.preventDefault();
    addUploadZone.classList.remove('dragover');
  });
});

addUploadZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    handleAddFile(file);
  }
});

addFileInput.addEventListener('change', e => {
  if (e.target.files[0]) {
    handleAddFile(e.target.files[0]);
  }
});

function handleAddFile(file) {
  selectedAddFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    addPreviewImg.src = e.target.result;
    addPreview.classList.add('visible');
  };
  reader.readAsDataURL(file);

  // Auto-fill name from filename (remove extension)
  if (!productNameInput.value) {
    productNameInput.value = file.name.replace(/\.[^/.]+$/, '').replace(/[_-]/g, ' ');
  }
  validateAddForm();
}

productNameInput.addEventListener('input', validateAddForm);

function validateAddForm() {
  addBtn.disabled = !(selectedAddFile && productNameInput.value.trim());
}

addBtn.addEventListener('click', async () => {
  if (!selectedAddFile || !productNameInput.value.trim()) return;

  addBtn.disabled = true;
  addBtn.innerHTML = '<span class="spinner" style="width:16px;height:16px;margin:0;border-width:2px;"></span> Uploading...';

  const formData = new FormData();
  formData.append('file', selectedAddFile);
  formData.append('product_name', productNameInput.value.trim());

  try {
    const res = await fetch(`${API_BASE}/products/add`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);

    showToast(`"${productNameInput.value.trim()}" added successfully!`, 'success');

    // Reset form
    selectedAddFile = null;
    productNameInput.value = '';
    addPreview.classList.remove('visible');
    addFileInput.value = '';
    validateAddForm();

  } catch (err) {
    console.error(err);
    showToast(err.message, 'error');
  } finally {
    addBtn.disabled = false;
    addBtn.innerHTML = '+ Add Product';
    validateAddForm();
  }
});

// ─── Initial Load ───
// Check API health on load
(async () => {
  try {
    const res = await fetch(`${API_BASE}/products`);
    if (res.ok) {
      document.getElementById('api-status-text').textContent = 'API Connected';
      document.querySelector('.status-dot').style.background = 'var(--success)';
    }
  } catch {
    document.getElementById('api-status-text').textContent = 'API Offline';
    document.querySelector('.status-dot').style.background = 'var(--danger)';
    showToast('Cannot connect to API. Make sure the server is running on port 8000.', 'error');
  }
})();
