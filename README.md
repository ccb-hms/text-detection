# text-detection
 Text detection from natural images using EAST algorithm

To use this repo:
- Please create a virtual environment and install the dependecies using the `requirements.txt` file. 
Requires Python 3.12.2 Eg:
```python
python3 -m venv <virtual-environment-name>
source  <virtual-environment-name>/bin/activate
pip install -r requirements.txt
```

To run the detection method with pre-trained weights, please use:

```python
python detect_text.py -c 0.95 -input /path/to/input/images/directory -output /path/to/where/you/want/to/store/results -weights path/to/frozen/weights

```
where `c` represents minimum probability required to inspect a region. Default is set to `0.5`.

If nothing is specified, the code runs example images from directory, `sample_images-text`.




# Some other details:

- The EAST algorithm cpaper can be found at: https://arxiv.org/abs/1704.03155

- Code has been adapted from: https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

- Other implementations: https://github.com/YCICI/EAST-OpenCv/tree/master


 


