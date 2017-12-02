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

General Inputs and Outputs
``` {python}
x_in = tf.keras.layers.Input(shape=(n_bin_encode,))
x = tf.keras.layers.Dense(1000, activation='relu')(x_in)
y_out = tf.keras.layers.Dense(4, activation='softmax')(x)
```
In the ipynb, we create 3 models
1) tf.keras model with tf.keras.optimizer
``` {python}
model1 = tf.keras.models.Model(inputs=[x_in], outputs=[y_out])  

my_optim = tf.keras.optimizers.Adam(lr = 0.0001) # <----------- ISSUE!!! if use this ... you will fail...
model1.compile(loss = 'categorical_crossentropy', 
               optimizer = my_optim, 
               metrics=['acc'])
```
[Image of model1](/images_result1.png)

2) tf.keras model with tf.train.optimizer
``` {python}
def my_loss(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

model2.compile(loss = my_loss, 
               optimizer = tf.train.AdamOptimizer(learning_rate=0.001), 
               metrics=['acc'])
```
[Image of model2](/images_result2.png)
3) Keras model
``` {python}
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam

k_x_in = Input(shape=(n_bin_encode,))

mx = Dense(1000, activation='relu')(k_x_in)
m_out = Dense(4, activation='softmax')(mx)

model3 = Model(inputs=[k_x_in], outputs=[m_out])

k_optim = Adam(lr=0.001)
model3.compile(loss = 'categorical_crossentropy', optimizer = k_optim, metrics=['acc'])
```
[Image of model3](/images_result3.png)
