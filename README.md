# Setup

<pre>
$ conda create -n waymo_ds2 python=3.9
...

$ conda activate waymo_ds2
$ conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
...
$
</pre>

For set up _cuda stuff_ correctly:

<pre>
$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
</pre>

and restart your environment (best in a new shell).

<pre>
$ conda activate waymo_ds2
$ pip install tensorflow==2.6.*
</pre>

Check tensorflow installation (CPU[first command], GPU[second command]):

<pre>
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
2023-03-08 08:09:39.169747: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.231289: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.232368: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.233975: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-08 08:09:39.236132: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.237040: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.237783: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.754388: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.754751: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.755128: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:09:39.755394: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3368 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1
tf.Tensor(10.413879, shape=(), dtype=float32)

$ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
2023-03-08 08:11:16.158024: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:11:16.164009: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2023-03-08 08:11:16.164351: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
</pre>if GPU available

Install waymo_open_dataset package:

<pre>
$ pip3 install waymo-open-dataset-tf-2-6-0
</pre>

<pre>
$ mkdir input
$ mkdir output
$ sutil cp gs://waymo_open_dataset_v_1_2_0_individual_files/training/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord ./input/.
...
</pre>

<pre>
$ mkdir waymo-od
$ git clone git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
</pre>
