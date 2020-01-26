**Bone** is an ultra simple application for automatic detection of bone morphology openings.   
Bone can detect openings from x-ray bone scan are return as a result two images: morphology openings and opening applied on uploaded x-ray scan.
           
#### How to run applicaiton
Before starting application download pre-trained bk models from personal website of professor Bogdan Kwolek and set BK_MODEL_PATH, BK_MODEL_WEIGHTS_PATH environment variables.  

Then, create virtual env using your favourite tool (pyenv, virtualenv, etc.) or just use base interpreter, then type:

```bash
$> pip install -r requirements.txt
$> python app.py
```
Application starts at **localhost:5000** , next follow the UI :)