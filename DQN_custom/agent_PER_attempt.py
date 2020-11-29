############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import copy
import collections
import matplotlib.pyplot as plt

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 200
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # create DQN
        self.dqn = DQN()
        # create replay buffer
        self.buffer = ReplayBuffer()
        # set minibatch size
        self.minibatch_size = 500
        # create list to store loss across episode
        self.loss_list = []
        # variable to record whether goal state reached
        self.finished_training = False
        
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        action = self.get_greedy_action(state) # get greedy action
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        if np.all(next_state == self.state): # punish if at boundary
            reward = -0.3
        else:
            reward = 1 - distance_to_goal**2 
        # Create a transition
        transition = (self.state, self._continuous_action_to_discrete(self.action), reward, next_state) # get transition tuple, converting continuous action to discrete 
        
        episode_num = np.floor(self.num_steps_taken/self.episode_length)
        
        # if reached the goal
        if distance_to_goal <= 0.03 and episode_num % 10 == 0:
            self.finished_training == True
            print('greedy policy leads to goal state - training stopped')
        
        self.buffer.append_transition_tuple(transition)        
        
        # print('Transition: ', transition)
        
        # create tensor to store loss
        loss = torch.tensor(0)
        
        if self.num_steps_taken >= self.minibatch_size:
            loss, td_error = self.dqn.train_q_network(self.buffer, self.minibatch_size, 0.95) # train network 
        else:
            td_error = 1
            
        self.buffer.append_td_error(td_error)        
        self.buffer.update_probabilities() 
            
        number_of_episodes = np.floor(self.num_steps_taken/self.episode_length)
        if number_of_episodes%5 == 0:
            self.dqn.update_target_network() # update target network every 5 episodes
        
        return loss.item()
    
    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        
        epsilon_decay = 0.99
        
        episode_num = np.floor(self.num_steps_taken/self.episode_length)
        num_steps_left_in_episode = self.episode_length - self.num_steps_taken%self.episode_length
        epsilon = epsilon_decay ** (episode_num) + episode_num * num_steps_left_in_episode/10e5
                
        state = torch.tensor(state)
        
        predictions = self.dqn.q_network.forward(torch.unsqueeze(state, 0))[0]
        greedy_action = int(torch.argmax(predictions,0))
        predictions = torch.cat([predictions[0:greedy_action], predictions[greedy_action + 1: ]])
        next_greedy_action = int(torch.argmax(predictions,0))
        
        num = np.random.random()        
        
        # if num > epsilon:
        #     action = greedy_action
        # else:
        #     action = np.random.choice([0,1,2,3])
        
        # every 10 episodes, set to greedy actions
        if episode_num % 10 == 0 and episode_num!=0:
            epsilon = 0
            print('greedy action taken')
        
        # if finished training (because greedy policy leads to goal state) set epsilon to zero so agent continuous to take this for the rest of the training period
        if self.finished_training == True:
            epsilon = 0
            print('finished training')
                            
        # graded epsilon-greedy
        if num > epsilon:
            action = greedy_action
        elif num > epsilon + (1-epsilon)/3:
            action = next_greedy_action
        else:
            action = np.random.choice([0,1,2])
                
        action = self._discrete_action_to_continuous(action) # convert action back to continous terms since this is how it will be treated in train_and_test.py
        
        return action
    
    # function to convert continuous action to discrete 
    # 0 = up, 1 = right, 2 = down, 3 = left
    def _continuous_action_to_discrete(self,continuous_action):
        if np.all(continuous_action == np.array([0,0.02],dtype=np.float32)): # Move up
            discrete_action = 0
        elif np.all(continuous_action == np.array([0.02,0],dtype=np.float32)): # Move right
            discrete_action = 1
        elif np.all(continuous_action == np.array([0,-0.02],dtype=np.float32)): # Move down
            discrete_action = 2
        # elif np.all(continuous_action == np.array([-0.02,0],dtype=np.float32)): # Move left
        #     discrete_action = 3
        return discrete_action
    
    # function to convert discrete action to continous
    # 0 = up, 1 = right, 2 = down, 3 = left
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0: # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 1: # Move right
            continuous_action = np.array([0.02,0], dtype=np.float32)
        elif discrete_action == 2: # Move down
            continuous_action = np.array([0,-0.02], dtype=np.float32)
        # elif discrete_action == 3: # Move left
        #     continuous_action = np.array([-0.02,0], dtype=np.float32)
        return continuous_action
    
    # # function to plot loss for analysis
    # def plot_loss(self):
    #     if self.num_steps_taken >= self.minibatch_size:
    #         loss = self.dqn.train_q_network(self.buffer, self.minibatch_size, 0.95) # train network and return loss
    #         self.loss_list.append(loss)
    #         if self.num_steps_taken % 50 == 0:
    #             plt.plot(range(self.num_steps_taken),self.loss_list)
    #             plt.x_label('number of steps')
    #             plt.y_label('average loss')
    #             plt.show()

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output
    
# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # create target network object
        self.target_network = copy.deepcopy(self.q_network)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, buffer, minibatch_length, discount):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # sample minibatch
        minibatch = buffer.sample_prioritised_minibatch(buffer.get_length(), minibatch_length)
        # Calculate the loss for this transition.
        loss, td_error = self._calculate_minibatch_loss(minibatch, discount)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss, td_error

    
    # function to calculate loss of minibatch of transitions to train Q network
    def _calculate_minibatch_loss(self, minibatch, discount):
        
        # create lists to hold transition variables for transitions in minibatch
        rewards = []
        minibatch_states = []
        minibatch_actions = []
        minibatch_next_states = []
        
        # append values for each transition to their respective lists
        for i in range(len(minibatch)):
            rewards.append(minibatch[i][2])
            minibatch_states.append(minibatch[i][0])
            minibatch_actions.append(minibatch[i][1])
            minibatch_next_states.append(minibatch[i][3])
            
        # convert lists to PyTorch tensors
        rewards_tensor = torch.tensor(rewards,dtype=torch.float32)
        minibatch_states_tensor = torch.tensor(minibatch_states)
        minibatch_actions_tensor = torch.tensor(minibatch_actions)
        minibatch_next_states_tensor = torch.tensor(minibatch_next_states)
        
        # next state Q-values for use in Bellman equation - Double Deep Q Network
        idx = torch.argmax(self.target_network.forward(minibatch_next_states_tensor),1) # use target network to find action with highest Q value
        max_next_state_Q_values = self.q_network.forward(minibatch_next_states_tensor).gather(dim=1, index=idx.long().unsqueeze(-1)).squeeze(-1)# calculate Q-value for action with highest Q-value using Q-network (Double Deep Q Network)
        
        # predict Q-values
        Q_predictions = self.q_network.forward(minibatch_states_tensor).gather(dim=1, index=minibatch_actions_tensor.long().unsqueeze(-1)).squeeze(-1)
        
        # measure change in Q-value compared to previous
        
        # use Bellman equation to calculate actual Q values
        Q_actual = rewards_tensor + discount*max_next_state_Q_values
        
        # calculate TD error for use in Prioritised Experience Replay Buffer
        td_error = Q_predictions - Q_actual
        
        loss = torch.nn.MSELoss()(Q_predictions, Q_actual)
        
        return loss, td_error
    
    # function to update target network by copying weights from main network when called
    def update_target_network(self):
        self.target_network = copy.deepcopy(self.q_network)
        
    
# REPLAY BUFFER CLASS FOR MINIBATCH TRAINING
class ReplayBuffer():
    
    def __init__(self):  
        self.buffer = collections.deque(maxlen=5000)
        self.td_errors = []
        self.probabilities = []
        
    def append_transition_tuple(self, transition_tuple):
        self.buffer.append(transition_tuple)
    
    def append_td_error(self,td_error):
        self.td_errors.append(td_error)
        
    def sample_minibatch(self, total_transitions, minibatch_size):
        minibatch_indices = np.random.choice(range(total_transitions), minibatch_size, replace=False) # replace = False so no repeated elements, otherwise minibatch size may be <50 due to way we're defining it
        # enumerate deque and create list of tuples according to randomly generated minibatch indices         
        minibatch = [transition_tuple for i, transition_tuple in enumerate(self.buffer) if i in minibatch_indices]
        return minibatch
    
    def sample_prioritised_minibatch(self, total_transitions, minibatch_size):
        print('Probabilities length: ', len(self.probabilities))
        print('Buffer length: ', len(self.buffer))
        minibatch_indices = np.random.choice(range(total_transitions), minibatch_size, p=self.probabilities, replace=False)
        minibatch = [transition_tuple for i, transition_tuple in enumerate(self.buffer) if i in minibatch_indices]
        return minibatch
    
    def update_probabilities(self):
        alpha = 0.5
        err = self.td_errors[-1]
        err_sum = 0
        for err in self.td_errors:
            err_sum += err
        prob = err**alpha/err_sum
        self.probabilities.append(prob)
        
    def get_length(self):
        return len(self.buffer)
    
    
    
    
    