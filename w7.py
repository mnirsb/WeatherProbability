#!/usr/bin/env python
# coding: utf-8

# In[1]:


#calculating Weather Using tensorflow and Probability.


# In[2]:


#importing packages
import tensorflow_probability as tfp
import tensorflow as tf


# In[3]:


#as per initials we are taking probability of weather for next seven days
tfd= tfp.distributions

#here for intials we are taking probability that 80% chances are cold and 20% chances for hot
initial_distribution = tfd.Categorical(probs=[0.8,0.2])

#here we are taking 2 values and checking them as condition, whether 70% cold then 30% hot chnaces or 20%cold then 80%hot chances
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],[0.2,0.8]])

#here we are distributing them with mean of 10 then max temp will goes upto 25 and min will goes upto 5 nd mean will be 15, Hence variation will be 10
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5.,10.])


# In[4]:


#lets create and assign the model
#here steps are showing how many days of observation.
model = tfd.HiddenMarkovModel(initial_distribution=initial_distribution,
                             transition_distribution=transition_distribution,
                             observation_distribution=observation_distribution,
                             num_steps=7)


# In[5]:


#creating the session and printing the mean value for 7 days in an array using numpy  
mean = model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())


# In[ ]:




