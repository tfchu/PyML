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

- CPU/memory usage
``` !cat /proc/cpuinfo ```
``` !cat /proc/meminfo ```

- list files
```!ls "/content/drive/My Drive/"```

- install keras
``` !pip install -q keras ```

- download a file 'Titanic.csv' to 'app'
``` !wget https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Titanic.csv -P "/content/drive/My Drive/app" ```

- run a python file, e.g. upload mnist.py to app
``` !python3 "/content/drive/My Drive/app/mnist_cnn.py" ```

- clone a github repo. files with ext 'ipynb' can be opened/run in lolab
``` !git clone https://github.com/wxs/keras-mnist-tutorial.git ```

