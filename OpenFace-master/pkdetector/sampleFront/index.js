// index.js
const express = require('express');
const fs = require('fs');
const path = require('path');
const multer  = require('multer');
const upload = multer({ dest: 'uploads/' });

const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname, 'public')));

app.post('/save-video', upload.single('video'), (req, res) => {
  fs.rename(req.file.path, path.join(req.file.destination, req.file.originalname), function(err) {
    if (err) {
      console.log(err);
      res.sendStatus(500);
    } else {
      res.sendStatus(200);
    }
  });
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`)
});
