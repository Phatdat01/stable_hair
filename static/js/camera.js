const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture');
const uploadBtn = document.getElementById('upload');
const image = document.getElementsByClassName('image-btn');
const cap = document.getElementsByClassName('capture-btn');
const resultImage = document.getElementById('resultImage');
const loading = document.getElementById('loading');

let capturedBlob = null;

// Start the camera
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    alert("❌ Could not access the camera. " + err.message);
    console.error(err);
  });

// Capture a photo (with object-fit: cover effect)
captureBtn.addEventListener('click', () => {
  const ctx = canvas.getContext('2d');
  const vWidth = video.videoWidth;
  const vHeight = video.videoHeight;
  const canvasSize = canvas.width;

  let sx, sy, sWidth, sHeight;
  const aspectRatio = vWidth / vHeight;

  if (aspectRatio > 1) {
    sHeight = vHeight;
    sWidth = vHeight;
    sx = (vWidth - sWidth) / 2;
    sy = 0;
  } else {
    sWidth = vWidth;
    sHeight = vWidth;
    sx = 0;
    sy = (vHeight - sHeight) / 2;
  }

  ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, canvasSize, canvasSize);

  canvas.toBlob(blob => {
    capturedBlob = blob;
    alert('✅ Picture captured!');
    image[0].style.display = "block";
    cap[0].style.display = "none";
  }, 'image/png');
});

// Upload photo
uploadBtn.addEventListener('click', () => {
  loading.style.display = 'block';
  resultImage.style.display = 'none';
  if (!capturedBlob) return alert('❌ Capture first!');

  const formData = new FormData();
  formData.append('face', new File([capturedBlob], 'face.png', { type: 'image/png' }));

  const selectedWig = document.querySelector('.wig-thumb.selected');
  if (!selectedWig) {
    alert('❌ Please select a wig.');
    return;
  }

  formData.append('shape', selectedWig.getAttribute('data-value'));

  fetch('/get_wig', {
    method: 'POST',
    body: formData
  })
  .then(res => res.blob())  // Get binary blob
  .then(blob => {
    const url = URL.createObjectURL(blob);
    resultImage.src = url;

    // Optional for debug
    console.log("✅ Image blob URL:", url);

    // Show image when it loads
    resultImage.onload = () => {
      loading.style.display = 'none';
      resultImage.style.display = 'block';
      resultImage.style.visibility = 'visible';
      resultImage.style.opacity = 1;
    };

    // Error fallback
    resultImage.onerror = () => {
      loading.style.display = 'none';
      alert("❌ Failed to load image.");
    };

    // Optional test: force it to show in body
    // document.body.appendChild(resultImage);
  })
  .catch(err => {
    loading.style.display = 'none';
    alert("❌ Something went wrong!");
    console.error(err);
  });
});

// Wig selection
const wigThumbs = document.querySelectorAll('.wig-thumb');
wigThumbs.forEach(thumb => {
  thumb.addEventListener('click', () => {
    wigThumbs.forEach(t => t.classList.remove('selected'));
    thumb.classList.add('selected');
  });
});

// Scroll buttons functionality
document.addEventListener('DOMContentLoaded', function () {
  const container = document.getElementById('wig-container');
  const nextBtn = document.getElementById('wig-next');
  const prevBtn = document.getElementById('wig-prev');
  const scrollStep = 200;

  function updateButtons() {
    const maxScroll = container.scrollWidth - container.clientWidth;
    prevBtn.style.display = container.scrollLeft > 0 ? 'block' : 'none';
    nextBtn.style.display = container.scrollLeft < maxScroll || container.scrollLeft === 0 ? 'block' : 'none';
  }

  nextBtn.addEventListener('click', () => {
    container.scrollBy({ left: scrollStep, behavior: 'smooth' });
  });

  prevBtn.addEventListener('click', () => {
    container.scrollBy({ left: -scrollStep, behavior: 'smooth' });
  });

  container.addEventListener('scroll', updateButtons);
  setTimeout(updateButtons, 10);
});
