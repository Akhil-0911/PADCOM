document.addEventListener('DOMContentLoaded', () => {
    const brainButton = document.getElementById('brain-button');
    const chestButton = document.getElementById('chest-button');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const imagePreview = document.getElementById('image-preview');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    const selectedOption = document.getElementById('selected-option');
    const previewContainer = document.querySelector('.preview-container');

    let currentMode = 'brain'; // Default mode

    // Button click handlers
    brainButton.addEventListener('click', () => switchMode('brain'));
    chestButton.addEventListener('click', () => switchMode('chest'));

    function switchMode(mode) {
        currentMode = mode;
        brainButton.classList.toggle('active', mode === 'brain');
        chestButton.classList.toggle('active', mode === 'chest');
        selectedOption.textContent = mode === 'brain' ? 'Brain Tumor Detection' : 'Pneumonia Detection';
        resetUI();
    }

    // File input handling
    uploadBtn.addEventListener('click', () => fileInput.click());

    // Drag and drop handling
    previewContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        previewContainer.style.borderColor = 'var(--primary-color)';
    });

    previewContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        previewContainer.style.borderColor = '#ccc';
    });

    previewContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        previewContainer.style.borderColor = '#ccc';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file.');
            return;
        }

        // Preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
        };
        reader.readAsDataURL(file);

        // Upload and process image
        uploadImage(file);
    }

    function uploadImage(file) {
        resetUI();
        loading.style.display = 'block';

        const formData = new FormData();
        formData.append('image', file);

        const endpoint = currentMode === 'brain' ? '/predict_brain' : '/predict_chest';

        fetch(endpoint, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayResult(data);
        })
        .catch(err => {
            showError(err.message || 'An error occurred during processing.');
        })
        .finally(() => {
            loading.style.display = 'none';
        });
    }

    function displayResult(data) {
        result.style.display = 'block';
        
        // Update result header with prediction
        const resultHeader = result.querySelector('.result-header');
        resultHeader.textContent = data.prediction;
        resultHeader.style.color = data.prediction.includes('Detected') || data.prediction === 'PNEUMONIA' 
            ? 'var(--danger-color)' 
            : 'var(--success-color)';

        // Update confidence bar
        const confidenceFill = result.querySelector('.confidence-fill');
        const confidenceText = result.querySelector('.confidence-text');
        confidenceFill.style.width = data.confidence;
        confidenceText.textContent = data.confidence;

        // Update probability info
        const probabilityInfo = result.querySelector('.probability-info');
        probabilityInfo.textContent = `Raw probability: ${data.probability}`;

        // Add condition-specific information
        const additionalInfo = result.querySelector('.additional-info');
        additionalInfo.innerHTML = getAdditionalInfo(data);
    }

    function getAdditionalInfo(data) {
        const probability = parseFloat(data.probability);
        if (currentMode === 'brain') {
            if (probability > 0.8) {
                return '<p class="warning">⚠️ High confidence tumor detection. Immediate medical attention recommended.</p>';
            } else if (probability > 0.5) {
                return '<p class="warning">⚠️ Possible tumor detected. Medical consultation recommended.</p>';
            } else {
                return '<p class="success">✅ No significant tumor indicators detected.</p>';
            }
        } else {
            if (probability > 0.8) {
                return '<p class="warning">⚠️ Strong indicators of pneumonia. Medical attention recommended.</p>';
            } else if (probability > 0.5) {
                return '<p class="warning">⚠️ Possible pneumonia detected. Medical consultation recommended.</p>';
            } else {
                return '<p class="success">✅ No significant pneumonia indicators detected.</p>';
            }
        }
    }

    function showError(message) {
        error.textContent = message;
        error.style.display = 'block';
        result.style.display = 'none';
    }

    function resetUI() {
        error.style.display = 'none';
        result.style.display = 'none';
        loading.style.display = 'none';
    }
});
