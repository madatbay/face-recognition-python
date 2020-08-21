# face-recognition-python

Python face recognition with OpenCV
_In progress_

### Installation

1. Fork or just clone repository to local machine
2. Create virtual environment to keep project files isolated:
   `py -m venv venv` or
   `python3 -m venv /path/to/new/virtual/environment`
   _Read more about - _ [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
3. Activate virtual environment:
   On MacOS & Linux: `source env/bin/activate`
   On Windows: `.\venv\Scripts\activate`
4. Pip install requirements.txt:
   `pip install -r requirements.txt`
5. Run it:
   `python main.py`
   or
   `python3 main.py`

## Face Recognition

You can train program with the help of some of your images. So, program will try to predict who you are each run time.
_How to tratin?_

1. Add the folder container images of you to the 'images' folder:
   `images/madat/1.png`or `images/john/1.jpg` or `images/jack/frontal_face.jpg`
   > Allowed image types - JPG, PNG
2. Run trainer:
   `python trainer.py` or `python3 trainer.py`
3. After training finishes, just run main program:
   `python main.py` or `python3 main.py`
