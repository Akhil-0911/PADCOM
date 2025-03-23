document.getElementById('brain-button').addEventListener('click', function() {
    showUploadSection('Brain Tumor Detection');
});

document.getElementById('heart-button').addEventListener('click', function() {
    showUploadSection('Pneumonia Detection');
});

document.getElementById('upload-btn').addEventListener('click', function() {
    document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        uploadImage(file);
    }
});

function showUploadSection(option) {
    document.getElementById('selected-option').textContent = `Upload an image for ${option}`;
    document.getElementById('image-upload').style.display = 'block';
}

function uploadImage(file) {
    const formData = new FormData();
    formData.append('image', file);

    // Show loading indicator
    document.getElementById('loading').style.display = 'block';

    const selectedOption = document.getElementById('selected-option').textContent;

    const url = selectedOption.includes('Brain Tumor') ? '/predict_brain' : '/predict_chest';

    fetch(url, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('result').textContent = data.result || data.error;
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('result').textContent = 'An error occurred, please try again.';
    });
}
