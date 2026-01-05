// Upload form handling
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Remove file
    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.value = '';
        filePreview.style.display = 'none';
        submitBtn.style.display = 'none';
    });

    function handleFileSelect(file) {
        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 
                             'video/webm', 'video/x-matroska', 'video/x-flv'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'];
        
        if (!allowedExtensions.includes(fileExtension)) {
            alert('Invalid file type. Please upload a video file (MP4, AVI, MOV, MKV, WEBM, FLV)');
            return;
        }

        // Validate file size (500MB)
        const maxSize = 500 * 1024 * 1024; // 500MB in bytes
        if (file.size > maxSize) {
            alert('File too large. Maximum size is 500MB');
            return;
        }

        // Show file preview
        fileName.textContent = file.name;
        filePreview.style.display = 'block';
        submitBtn.style.display = 'block';
    }

    // Form submission
    uploadForm.addEventListener('submit', (e) => {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert('Please select a file to upload');
            return;
        }

        // Show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = 'Uploading...';
    });
});

