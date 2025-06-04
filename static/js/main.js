document.getElementById('face').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const preview = document.getElementById('facePreview');

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// AJAX form submit
// AJAX form submit
document.getElementById('wigForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const resultImage = document.getElementById('resultImage');
    const loading = document.getElementById('loading');

    // Show loading spinner
    loading.style.display = 'block';
    resultImage.style.display = 'none';

    fetch('/get_wig', {
        method: 'POST',
        body: formData
    })
    .then(res => res.blob())  // Get binary blob
    .then(blob => {
        const url = URL.createObjectURL(blob);
        resultImage.src = url;
        resultImage.onload = () => {
            loading.style.display = 'none'; // Hide loading after image loads
            resultImage.style.display = 'block';
        };
    })
    .catch(err => {
        loading.style.display = 'none';
        alert("Something went wrong!");
        console.error(err);
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const wigThumbnails = document.querySelectorAll('.wig-thumb');
    const wigInput = document.getElementById('wig');

    // Set default selected value
    const defaultValue = '1';
    wigInput.value = defaultValue;

    wigThumbnails.forEach(thumb => {
        // Highlight the default image
        if (thumb.getAttribute('data-value') === defaultValue) {
            thumb.style.border = '2px solid #007bff';
        }

        // Add click handler
        thumb.addEventListener('click', function () {
            // Update hidden input
            wigInput.value = this.getAttribute('data-value');

            // Reset all borders
            wigThumbnails.forEach(t => t.style.border = '2px solid transparent');

            // Highlight selected image
            this.style.border = '2px solid #007bff';
        });
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const container = document.getElementById('wig-container');
    const selection = document.getElementById('wig-selection');
    const nextBtn = document.getElementById('wig-next');
    const prevBtn = document.getElementById('wig-prev');
    let scrollAmount = 0;
    const scrollStep = 200;

    function updateButtons() {
        const maxScroll = selection.scrollWidth - container.clientWidth;
        prevBtn.style.display = scrollAmount > 0 ? 'block' : 'none';
        nextBtn.style.display = scrollAmount < maxScroll ? 'block' : 'none';
    }

    nextBtn.addEventListener('click', () => {
        const maxScroll = selection.scrollWidth - container.clientWidth;
        scrollAmount = Math.min(scrollAmount + scrollStep, maxScroll);
        selection.style.transform = `translateX(-${scrollAmount}px)`;
        updateButtons();
    });

    prevBtn.addEventListener('click', () => {
        scrollAmount = Math.max(scrollAmount - scrollStep, 0);
        selection.style.transform = `translateX(-${scrollAmount}px)`;
        updateButtons();
    });

    // Initial check
    updateButtons();
});