
import sys

print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
except ImportError:
    print("TensorFlow is not installed in this Python environment.")
