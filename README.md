# PyML 
## Resources
1. https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
2. https://www.youtube.com/playlist?list=PLXVfgk9fNX2I7tB6oIINGBmW50rrmFTqf
3. https://www.coursera.org/learn/machine-learning/home/welcome
4. https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC
5. https://classroom.udacity.com/courses/ud730

## Others
1. https://buzzorange.com/techorange/2017/08/16/big-data-for-stock/
2. https://buzzorange.com/techorange/2017/08/16/how-to-act-like-an-ai-expert/


## Google colab: create a ``` app ``` folder
- Ref
    - https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d
    - https://colab.research.google.com/notebooks/intro.ipynb

- GPU selection
Edit > Notebook settings > Hardware accelerator, select GPU
```
# check if CPU or GPU is used
import tensorflow as tf
tf.test.gpu_device_name()
```

```
# check GPU being used, e.g. Tesla K80
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

- mount google drive (copy the auth code)
```
from google.colab import drive
drive.mount('/content/drive/')
```

- Usages
``` 
!cat /proc/cpuinfo                  # CPU usage
!cat /proc/meminfo                  # memory usage
!ls "/content/drive/My Drive/"      # list files
!pip install -q keras               # install keras
!wget https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Titanic.csv -P "/content/drive/My Drive/app"   # download file
!python3 "/content/drive/My Drive/app/mnist_cnn.py"             # run python file
!git clone https://github.com/wxs/keras-mnist-tutorial.git      # clone a github repo
```