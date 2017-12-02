<h2> Using tf.keras </h2>

Many people might ask why we use tf.keras rather than using Keras? The reason is that ... <br>
Keras itself is easy to use. However, when you want to "go deeper", e.g. modify the layer, mix losses, etc., you want finally find it is quite wasting time and difficult to manage it. <br>
With tf.keras, it can easisly swithch tensors between TensorFlow itself and Keras functions (such as layer). Also, learning phase contorl in Tensorflow itself is very annoying (as my viewpoint). Hence, Using Keras model.fit is very easy to run your model with well controlling of learning/evaluting phase. <br>

In sum, the advantage of using tf.keras
1) Easy to switch/mix tensors. You can create your model layers with both TF way or Keras-layer way, modify your losses and optimizers.
2) Training phase controller.
3) Combine tf.data.Dataset -- super fast

-------
<h4> Using the Fizz Buzz example to show the difference between tf.keras and Keras </h4>

In the ipynb, we create 3 models
1) tf.keras model with tf.keras.optimizer
``` {python}

```
[Image of model1](/images_result1.png)

2) tf.keras model with tf.train.optimizer


3) Keras model
