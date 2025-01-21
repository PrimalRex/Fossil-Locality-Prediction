import tensorflow as tf

# Just check that tensorflow is detecting and using a GPU (Not necessary unless you want faster training)
print(tf.config.list_physical_devices('GPU'))