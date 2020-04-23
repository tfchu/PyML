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


## Google colab
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

- change dir
```
import os
os.chdir("drive/app")
```

- usage
    - create an ```app``` folder, and add colab notebook under it
    - install keras
    - mount google drive
    - change dir to ```app``` folder
    - remove ```tb_logs``` folder
    - load tensorboard notebook extension
    - train a model, set tensorboard dir to ```tb_logs```
    - visualize with tensorboard

- tensorboard: https://www.tensorflow.org/tensorboard/get_started
```
%load_ext tensorboard           # Load the TensorBoard notebook extension
%tensorboard --logdir tb_logs/  # visualize
```

- misc commands
``` 
!cat /proc/cpuinfo                  # CPU usage
!cat /proc/meminfo                  # memory usage
!pip install -q keras               # install keras
!ls "/content/drive/My Drive/"      # list files
!rm -rf tb_logs/                    # remove a folder
!wget https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Titanic.csv -P "/content/drive/My Drive/app"   # download file
!python3 "/content/drive/My Drive/app/mnist_cnn.py"             # run python file
!git clone https://github.com/wxs/keras-mnist-tutorial.git      # clone a github repo
!kill -9 -1                         # restart colab
```

- tensor board (TBD)
```
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

# You can change the directory name
LOG_DIR = 'tb_logs'

import os
if not os.path.exists(LOG_DIR):
  os.makedirs(LOG_DIR)

# launch TensorBoard in the background 
get_ipython().system_raw(
    'tensorboard --logdir={} --host=127.0.0.1 --port=6006 &'.format(LOG_DIR)
)

# use ngrok to tunnel TensorBoard port 6006 to the outside world
get_ipython().system_raw('./ngrok http 6006 &')

# get the public URL
!curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```