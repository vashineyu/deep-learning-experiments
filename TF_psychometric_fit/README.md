### Tutorial of using Tensorflow fit psychometric function

#### Introduction
In this session, we're going to fit the psychometric function with Tensorflow. <br>
> Generally, a psychometric function is an inferential model applied in detection and discrimination tasks. It models the relationship  between a given feature of a physical stimulus, e.g. velocity, duration, brightness, weight etc., and forced-choice responses of a human test subject. ([Wiki](https://en.wikipedia.org/wiki/Psychometric_function)) <br>
  
The shape of the psychometric function is very similar to (or general format) of the sigmoid function. <br>
![](https://www.ncbi.nlm.nih.gov/books/NBK11513/bin/psych1f13.jpg)  
(Image source: The Organization of the Retina and Visual System)

In this tutorial, the task of this data is to evalute how a subject perceive and judge the stimuli is a "Short" stimuli of "Long" one (2AFC task) <br>

Conditions <br>
- Stimuli type (1 / 2) <br>
- Attention type (1 / 2 / 3) <br>
- Session (1 / 2) <br>
Input <br>
- Stimuli time, (400, 600, 800, 1000, 1200, 1400, 1600) <br>
Output <br>
- Probability of subject judge it as 1 (long) <br>


#### Method
Before we start fitting, let's use python to investigate how parameters affect the curve.
```python
y = 1 / (1 + c * math.exp(-(a*t )+b))
```
And here is the result.
![](https://github.com/vashineyu/deep-learning-experiments/blob/master/TF_psychometric_fit/images/Standard_fomula.png)

Here, let's enter the critical part: build the formula above with TF.
```python
t = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

const = tf.constant(1, dtype=tf.float32)
a = tf.Variable(tf.random_normal([1], mean=-0.5, stddev=0.01)) # this is important, if a > 0, the curve will inverse
b = tf.Variable(tf.random_normal([1]))
c = tf.Variable(tf.random_normal([1]))
pred = c + ( (tf.subtract(const, c)) / (const + tf.exp(tf.multiply(a, tf.subtract(t,b))) ) )
loss = tf.reduce_sum(tf.pow(pred - y, 2))
```
variables explain:  
  - t and y: the placeholder for the data point (t) and answer (y) respectively  
  - a, b, and c: three main parameters that we want to find -- so as to optimize the curve  
  - pred: combine these components to as the formula  
  - loss: how we minimize the difference  

#### Conclusion
