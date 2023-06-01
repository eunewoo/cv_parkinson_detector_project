// index.js
const express = require('express');
const fs = require('fs');
const path = require('path');
const multer  = require('multer');
const upload = multer({ dest: 'uploads/' });
const { exec } = require("child_process");

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

app.post('/save-image', upload.single('image'), (req, res) => {
  fs.rename(req.file.path, path.join(req.file.destination, req.file.originalname), function(err) {
    if (err) {
      console.log(err);
      res.sendStatus(500);
    } else {
      res.sendStatus(200);
    }
  });
});

app.post('/run-python', (req, res) => {
  exec("python ../faceTest.py", (error, stdout, stderr) => {
    if (error) {
      console.log(`error: ${error.message}`);
      res.sendStatus(500);
    } else if (stderr && !stderr.startsWith("[ WARN")) {
      console.log(`stderr: ${stderr}`);
      res.sendStatus(500);
    } else {
      // Read the result from output.json
      fs.readFile('output.json', 'utf8', (err, data) => {
        if (err) {
            console.log(`error: ${err}`);
            res.sendStatus(500);
        } else {
            try {
                const pythonResult = JSON.parse(data);
                if (pythonResult.prediction === 1) {
                  // handle prediction 1
                  console.log('success king')
                } else {
                  // handle prediction 0
                }
                res.send(pythonResult.message); // send only the message
            } catch (e) {
                console.log("Parsing error: ", e);
                res.sendStatus(500);
            }
        }
    });
    
    }
  });
});

app.post('/run-python-2', (req, res) => {
  exec("python ../spiralTest.py", (error, stdout, stderr) => {
    if (error) {
      console.log(`error: ${error.message}`);
      res.sendStatus(500);
    } else if (stderr) {
      console.log(`stderr: ${stderr}`);
      res.sendStatus(500);
    } else {
      // Process the stdout into the response
      res.send(stdout);
    }
  });
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`)
});

