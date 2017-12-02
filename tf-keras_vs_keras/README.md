# Using tf.keras

Many people might ask why we use tf.keras rather than using Keras? <br>
The reason is that ... <br>
Keras itself is easy to use. However, when you want to "go deeper", e.g. modify the layer, mix losses, etc., you want finally find it is quite wasting time and difficult to manage it.
With tf.keras, it can easisly swithch tensors between TensorFlow itself and Keras functions (such as layer). Also, learning phase contorl in Tensorflow itself is very annoying (as my viewpoint). Hence, Using Keras model.fit is very easy to run your model with well controlling of learning/evaluting phase.

In sum, the advantage of using tf.keras
1) Easy to switch/mix tensors. You can create your model layers with both TF way or Keras-layer way, modify your losses and optimizers.
2) Training phase controller.
3) Combine tf.data.Dataset -- super fast

