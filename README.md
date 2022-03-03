# MountainCar_TD_Lambda_Q_value_fun_approx

We use **TD(λ)** coupled with the **elibility traces** to train a value function approximator to estimate the state-action value functions (Q-values), and then derive the optimal policy from it. The value function approximator is a linear function. We also use the RBF kernels to increase the feature space from 2 to 2000 to increase the performance of the model.

<br />

## Task:

The goal is to design a control policy for a car in a valley to get on the top of the valley as soon as possible.



## Solution:

Here we use the **TD(λ)** for estimating the state-action values for different pairs of (state, action). The **Eligibility Traces** allow us to use a **Backward-View implementation**, which means that we do not need to wait until the end of episodes to calculate the updates. Using such estimated values as training samples, we train a (linear) function approximator (with some kernels to shift the feature space to a higher dimension space) to estimate the state-action values (Q-values).

---



When updating the Q-values (i.e. state-action value functions), the updates coming from the **TD(λ)** algorithm are given as some mixture of the multi-step return predictions, ranging from TD(0) (which is the original temporal-difference, TD(0), algorithm) to TD(∞) (which is practically speaking the Monte-Carlo approach, and ∞ means "till the end of the episode").

Now, one can easily see that if we implement TD(λ) without any *smart idea*, then the computational time would be significant. That smart idea is called **eligibility traces** which allow the algorithm to be incremental. This idea is also referred to as the **Backward-View**.



In fact, the eligibility traces can be deﬁned in multiple ways and hence TD(λ) exists in correspondingly many multiple forms. For implementing TD(λ) with value funcition approximators, we use the following update formulations [1]:

<p float="left">
  <img src="/figs/Formulation_TD_lambda_Q_value_fun_approx.png" width="450" />
</p>

Where $\delta$ is the error of the function approximator, $V_{\theta}$ is the value function approximator with $\theta$ as its parameters. $\lambda$ is the discount rate, and $\alpha$ is the learning rate. $X$ is the feature representaion of (a) state. $$z$$ \[z\] is the elibility traces.



Note how the idea of the elibility traces resembles the idea of the *momentum* in the optimization methods.



### A review of the environment

A car is on a one-dimensional track, positioned between two “mountains”. The goal is to drive up the mountain on the right; however, the car’s engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

The car’s state, at any point in time, is given by a **two-dimensional vector** containing its **horizonal position** and **velocity**. The car commences each episode stationary, at the bottom of the valley between the hills (at position approximately -0.5), and the episode ends when either the car reaches the flag (position > 0.5).

At each move, the car has three actions available to it: push **left**, push **right** or **do nothing**, and a penalty of 1 unit is applied for each move taken (including doing nothing). This actually allows for using the exploration technique called **optimistic initial values**.



### Results

After training the model, the results are shown as the follows:

The performance of the optimal policy:

<p float="left">
  <img src="/figs/MountainCar_TD_Lambda_Q_value_fun_approx.gif" width="450" />
</p>

We also record (a video for) the performance of the algorithm for the optimal policy.



Average total reward over different episodes:

<p float="left">
  <img src="/figs/Mountain_Car_Average_Total_Reward_TD_lambda_Q_value_fun_approx.png" width="450" />
</p>

As you can see it is very volatile. It is better to take a look at the running average of this figure, which is practically at each point, it is the average of the 10 recent steps. It reduces the effect of the randomness of the initial point, and helps us with interpreting the results.

<p float="left">
  <img src="/figs/Mountain_Car_Running_Average_Total_Reward_TD_lambda_Q_value_fun_approx.png" width="450" />
</p>

And the average # steps to reach to the top of the mountain at different state-action pairs (which is practically $(-Q^*(s,a))$) is shown in the following figure:

<p float="left">
  <img src="/figs/Mountain_Car_Num_steps_to_Reach_Mountain_TD_lambda_Q_value_fun_approx.png" width="450" />
</p>



