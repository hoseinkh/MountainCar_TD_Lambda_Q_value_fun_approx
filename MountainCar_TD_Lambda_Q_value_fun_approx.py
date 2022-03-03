###############################################################################
# For more info, see https://hoseinkh.github.io/
###############################################################################
import gym
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
# code we already wrote
# import q_learning
###############################################################################
class SGDRegressor_modified:
  # we implement and modify the SGD regressor!
  # we assume the loss is l2. If the loss is something else, we need ...
  # ... to change the gradient!
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)
    self.learning_rate = 0.01
  ######################################
  def partial_fit(self, X, Y, eligibility_trace, input_learning_rate=None):
    if input_learning_rate != None:
      self.learning_rate = input_learning_rate
    #
    # The following is the implementation of the RL with eligibility traces as discussed in ...
    # ... the Book "Algorithms for Reinforcement Learning; Szepesvari", (see section 2.2.1, equation 2.5)
    delta_error_of_value_fun_approximator = (Y - X.dot(self.w))
    self.w += self.learning_rate*delta_error_of_value_fun_approximator*eligibility_trace
  ######################################
  def predict(self, X):
    pred_target = X.dot(self.w)
    if type(X.dot(self.w)) == np.ndarray:
      pred_target = pred_target[0]
    return pred_target
###############################################################################
class FeatureTransformer:
  def __init__(self, env, n_components=500):
    ## generate states (observations)
    """
    NOTE!! state samples are poor, b/c you get velocities --> infinity, which result in ...
    ... poor performance! Hence, instead of sampling from the environment directly, we ...
    ... sample from it manually, and hence define the range for which we are interested in!
    """
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    # observation_examples = np.random.random((20000, 2)) * 2 - 1
    # define scaler and scale the states (observations) --> mean 0 and variance 1
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    #
    # Now we basically use RBF to for feature generation
    # Each RBFSampler takes each (original) (feature representation) of ...
    # ... a state and converts it to "n_components" new featuers.
    # Hence, after concatenating the new features, we convert each state to ...
    # ... {(# RBF samplers) * n_components} new features.
    #
    # We use RBF kernels with different variances to cover different parts ...
    # ... of the space.
    #
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    # For all the generated samples, transform original state representaions ...
    # ... to a new state representation using "featurizer"
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    #
    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
  ######################################
  def transform(self, observations):
    #
    scaled_original_state_representation = self.scaler.transform(observations)
    #
    scaled_higher_dimensions_state_representation = self.featurizer.transform(scaled_original_state_representation)
    return scaled_higher_dimensions_state_representation
###############################################################################
# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    #
    D = feature_transformer.dimensions
    # we initialize the eligibility traces to 0. For more info, ...
    # ... see the book "Algorithms for Reinforcement Learning; Szepesvari", (see section 2.2.1, equation 2.5)
    self.eligibilities = np.zeros((env.action_space.n, D))
    #
    for i in range(env.action_space.n):
      model = SGDRegressor_modified(D)
      self.models.append(model)
  ######################################
  # the following method returns the estimated returns for each action for the current state
  def predict(self, s):
    X = self.feature_transformer.transform([s])
    result = np.stack([m.predict(X) for m in self.models]).T
    return result
  ######################################
  # this method takes the state, action, reward, and ...
  # ... (1) updates the eligibility traces,
  # ... (2) updates the weights of the value function approximator for the given action.
  def update(self, s, a, G, discount_rate, lambda_):
    X = self.feature_transformer.transform([s])
    #
    # The following is the implementation of the RL with the eligibility traces as discussed in ...
    # ... the Book "Algorithms for Reinforcement Learning; Szepesvari", (see section 2.2.1, equation 2.5)
    self.eligibilities *= discount_rate*lambda_
    self.eligibilities[a] += X[0] # This is the gradient of value fun approximator w.r.t. the weights w
    #
    self.models[a].partial_fit(X[0], G, self.eligibilities[a])
  ######################################
  def epsilon_greedy_action_selection(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))
###############################################################################
# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, discount_rate, lambda_):
  observation = env.reset()
  done = False
  totalreward = 0
  states_actions_rewards = []
  iters = 0
  # model.reset()
  while not done and iters < 10000:
    action = model.epsilon_greedy_action_selection(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    #
    # update the model
    next = model.predict(observation)
    G = reward + discount_rate * np.max(next[0])
    model.update(prev_observation, action, G, discount_rate, lambda_)
    #
    totalreward += reward
    iters += 1
    #
  return states_actions_rewards, totalreward
###############################################################################
def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.ylabel("Running Average Total Reward")
  plt.xlabel("Episode")
  # plt.show()
  plt.savefig("./figs/Running_Average_Total_Reward.png")
  plt.close()
###############################################################################
# here we plot the negative of the optimal state value functions (i,e, -V*(s))!
# Note that the optimal action values are equal to the negative of the average optimal time ...
# ... that it takes to reach the mountain.
# Hence this plot shows the average optimal time to reach the top of the mountain at each state.
def plot_avg_num_remaining_steps(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -1*np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)
  #
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Num steps to reach mountain == -V(s)')
  ax.set_title("Num steps to Reach Mountain Function")
  fig.colorbar(surf)
  fig.savefig("./figs/Num_steps_to_Reach_Mountain.png")
  # plt.show()
  plt.close()
###############################################################################
if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft)
  discount_rate = 0.9999
  lambda_ = 0.7
  #
  if True:
    monitor_dir = os.getcwd() + "/videos/" + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
  #
  num_of_episodes = 500
  totalrewards = np.empty(num_of_episodes)
  costs = np.empty(num_of_episodes)
  for i in tqdm(range(num_of_episodes)):
    # curr_eps = 1.0/(0.1*i+1)
    curr_eps = 0.1*(0.97**i)
    # curr_eps = 1.0/np.sqrt(i+1)
    # curr_eps = 0.1
    states_actions_rewards, totalreward = play_one(model, env, curr_eps, discount_rate, lambda_)
    totalrewards[i] = totalreward
    if i % 100 == 0:
      print("episode:", i, "total reward:", totalreward, "eps:", curr_eps, "avg reward (last 100):", totalrewards[max(0, i-100):(i+1)].mean())
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())
  #
  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.savefig("./figs/Average_Total_Reward.png")
  plt.show()
  plt.close()
  #
  plot_running_avg(totalrewards)
  # plot the optimal state-value function
  plot_avg_num_remaining_steps(env, model)


