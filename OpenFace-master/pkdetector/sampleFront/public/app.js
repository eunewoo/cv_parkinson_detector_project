// app.js
let mediaRecorder;
let recordedBlobs;

const captureButton = document.querySelector('#capture-btn');
const captureButton2 = document.querySelector('#capture-btn-2');
const video = document.querySelector('video');
const canvas = document.createElement('canvas'); // create a canvas element for capturing images

const videoNames = ['smileMe', 'disgustMe', 'surpriseMe'];
const detectButton = document.querySelector('#detect-btn');
const detectButton2 = document.querySelector('#detect-btn-2');

function blobToDataURL(blob) {
  return new Promise((fulfill, reject) => {
      let reader = new FileReader();
      reader.onerror = reject;
      reader.onload = (e) => fulfill(reader.result);
      reader.readAsDataURL(blob);
  });
}

detectButton.addEventListener('click', () => {
  fetch('/run-python', {
    method: 'POST'
  })
  .then(response => response.text())
  .then(data => alert(data)) // Display the result as an alert
  .catch(error => console.error('Error:', error));
});

detectButton2.addEventListener('click', () => {
  fetch('/run-python-2', {
    method: 'POST'
  })
  .then(response => response.text())
  .then(data => alert(data)) // Display the result as an alert
  .catch(error => console.error('Error:', error));
});

let videoIndex = 0;

captureButton.addEventListener('click', async () => {
  for (const name of videoNames) {
    const stream = await navigator.mediaDevices.getUserMedia({video: true});
    video.srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);

    recordedBlobs = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedBlobs.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedBlobs, {type: 'video/webm'});
      const formData = new FormData();
      formData.append('video', blob, `${videoNames[videoIndex]}.webm`);

      fetch('/save-video', {
        method: 'POST',
        body: formData
      });

      videoIndex++;

      if (videoIndex < videoNames.length) {
        await new Promise(resolve => setTimeout(resolve, 3000));
        mediaRecorder.start();
      }
    };

    mediaRecorder.start();
    await new Promise(resolve => setTimeout(resolve, 5000));
    mediaRecorder.stop();
    video.srcObject.getTracks()[0].stop();
  }
});

captureButton2.addEventListener('click', async () => {
  const stream = await navigator.mediaDevices.getUserMedia({video: true});
  video.srcObject = stream;

  await new Promise(resolve => setTimeout(resolve, 5000)); // wait 5 seconds

  const canvas = document.querySelector('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
  
  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('image', blob, 'spiralMe.png');

    fetch('/save-image', {
      method: 'POST',
      body: formData
    });
  }, 'image/png');
  
  video.srcObject.getTracks().forEach(track => track.stop());
});



