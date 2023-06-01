// app.js
let mediaRecorder;
let recordedBlobs;

const captureButton = document.querySelector('#capture-btn');
const video = document.querySelector('video');

const videoNames = ['smileMe', 'disgustMe', 'surpriseMe'];
const detectButton = document.querySelector('#detect-btn');

detectButton.addEventListener('click', () => {
  fetch('/run-python', {
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
