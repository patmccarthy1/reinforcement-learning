# Import some modules from other libraries
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import collections
import random
import cv2
import copy


# Import the environment module
from environment import Environment


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self,episode):
        # Choose the next action.
        # discrete_action = self._choose_next_action_random() 
        discrete_action = self._choose_epsilon_greedy_action(self.state,episode) # greedy action
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action_random(self):
        # Choose random action and return
        action = np.random.randint(0,4)  
        return action
    
    # function to choose action in epsilon greedy manner
    def _choose_epsilon_greedy_action(self, current_state, k):
        
        epsilon_decay = 0.993
        epsilon = epsilon_decay**k
        
        current_state = torch.tensor(current_state)
        
        predictions = dqn.q_network.forward(torch.unsqueeze(current_state, 0))
        greedy_action = int(torch.argmax(predictions,1)[0])
        
        num = np.random.random()        
        
        if num < epsilon:
            action = greedy_action
        else:
            action = np.random.choice([0,1,2,3])
            
        return action
        

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    # 0 = right, 1 = left, 2 = up, 3 = down
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0: # Move right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1: # Move up
            continuous_action = np.array([0,0.1], dtype=np.float32)
        elif discrete_action == 2: # Move left
            continuous_action = np.array([-0.1,0], dtype=np.float32)
        elif discrete_action == 3: # Move down
            continuous_action = np.array([0,-0.1], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


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
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # create target network object
        self.target_network = copy.deepcopy(self.q_network)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition, train_type, discount):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        if train_type == 'online':
            loss = self._calculate_loss(transition)
        elif train_type == 'minibatch':
            loss = self._calculate_minibatch_loss(transition, discount)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # FUNCTION TO CALCULATE LOSS FOR A PARTICULAR TRANSITION
    def _calculate_loss(self, transition):
        
        # 'transition' is tuple containing [state,action,reward,next_state] for agent's exploration so can work out actual state-action value (Q_actual) from this
        current_state = torch.tensor(transition[0])
        action = transition[1]
        instantaneous_reward = torch.tensor(transition[2],dtype=torch.float32)
        predicted_rewards = self.q_network.forward(torch.unsqueeze(current_state, 0))
        predicted_reward = predicted_rewards[0, action]
        
        # when using instantaneous reward
        Q_predicted = predicted_reward
        Q_actual = instantaneous_reward
        
        loss = torch.nn.MSELoss()(Q_predicted, Q_actual)
        
        return loss
    
    # FUNCTION TO CALCULATE LOSS FOR MINIBATCH OF TRANSITIONS
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
        
        # next state Q values for use in Bellman equation
        # max_next_state_Q_values, idx = torch.max(self.q_network.forward(minibatch_next_states_tensor),1)
        max_next_state_Q_values, idx = torch.max(self.target_network.forward(minibatch_next_states_tensor),1) # target network
        max_next_state_Q_values = max_next_state_Q_values.detach()
        
        # predict Q values
        Q_predictions = self.q_network.forward(minibatch_states_tensor).gather(dim=1, index=minibatch_actions_tensor.long().unsqueeze(-1)).squeeze(-1)
        
        # use Bellman equation to calculate actual Q values
        Q_actual = rewards_tensor + discount*max_next_state_Q_values
        
        loss = torch.nn.MSELoss()(Q_predictions, Q_actual)
        
        return loss
    
    # function to update target network by copying weights from main network when called
    def update_target_network(self):
        self.target_network = copy.deepcopy(self.q_network)
        
    # function to get Q values
    def get_Q_vals(self):
        q_values = np.zeros([10,10,4]) # create 3D array to hold q values
        for row in range(10):
            for col in range(10):
                state = torch.tensor([row/10 + 0.05,col/10 + 0.05])
                for action in range(4):
                    predicted_rewards = self.q_network.forward(torch.unsqueeze(state,0))
                    q_values[row,col,action] = predicted_rewards[0, action]
        return q_values
    
# REPLAY BUFFER CLASS FOR MINIBATCH TRAINING
class ReplayBuffer:
    def __init__(self):  
        self.buffer = collections.deque(maxlen=5000)
        
    def append_transition_tuple(self, transition_tuple):
        self.buffer.append(transition_tuple)
    
    def sample_minibatch(self, total_transitions, minibatch_size):
        minibatch_indices = np.random.choice(range(total_transitions), minibatch_size, replace=False) # replace = False so no repeated elements, otherwise minibatch size may be <50 due to way we're defining it
        # enumerate deque and create list of tuples according to randomly generated minibatch indices         
        minibatch = [transition_tuple for i, transition_tuple in enumerate(self.buffer) if i in minibatch_indices]
        return minibatch
    
    def get_length(self):
        return len(self.buffer)

class QValueVisualiser:

    def __init__(self, environment, magnification=500):
        self.environment = environment
        self.magnification = magnification
        self.half_cell_length = 0.05 * self.magnification
        # Create the initial q values image
        self.q_values_image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)

    def draw_q_values(self, q_values):
        # Create an empty image
        self.q_values_image.fill(0)
        # Loop over the grid cells and actions, and draw each q value
        for col in range(10):
            for row in range(10):
                # Find the q value ranges for this state
                max_q_value = np.max(q_values[col, row])
                min_q_value = np.min(q_values[col, row])
                q_value_range = max_q_value - min_q_value
                # Draw the q values for this state
                for action in range(4):
                    # Normalise the q value with respect to the minimum and maximum q values
                    q_value_norm = (q_values[col, row, action] - min_q_value) / q_value_range
                    # Draw this q value
                    x = (col / 10.0) + 0.05
                    y = (row / 10.0) + 0.05
                    self._draw_q_value(x, y, action, float(q_value_norm))
        # Draw the grid cells
        self._draw_grid_cells()
        # Show the image
        # cv2.imwrite('q_values_epsilon_greedy.png', self.q_values_image)
        cv2.imshow("Q Values", self.q_values_image)
        cv2.waitKey(1)
    
    def save_Q_image(self,name):
        cv2.imwrite('{}.png'.format(name), self.q_values_image)

    def _draw_q_value(self, x, y, action, q_value_norm):
        # First, convert state space to image space for the "up-down" axis, because the world space origin is the bottom left, whereas the image space origin is the top left
        y = 1 - y
        # Compute the image coordinates of the centre of the triangle for this action
        centre_x = x * self.magnification
        centre_y = y * self.magnification
        # Compute the colour for this q value
        colour_r = int((1 - q_value_norm) * 255)
        colour_g = int(q_value_norm * 255)
        colour_b = 0
        colour = (colour_b, colour_g, colour_r)
        # Depending on the particular action, the triangle representing the action will be drawn in a different position on the image
        if action == 0:  # Move right
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y - self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 1:  # Move up
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = centre_x - self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 2:  # Move left
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y + self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 3:  # Move down
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = centre_x + self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    def _draw_grid_cells(self):
        # Draw the state cell borders
        for col in range(11):
            point_1 = (int((col / 10.0) * self.magnification), 0)
            point_2 = (int((col / 10.0) * self.magnification), int(self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        for row in range(11):
            point_1 = (0, int((row / 10.0) * self.magnification))
            point_2 = (int(self.magnification), int((row / 10.0) * self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
    
    # FUNCTION TO DRAW GREEDY POLICY
    # def draw_greedy_policy(self):
        
            
# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    
    # Create an agent
    agent = Agent(environment)
    
    # Create a DQN (Deep Q-Network)
    dqn = DQN()
    
    num_episodes = 400 # number of episodes
    episode_length = 50 # episode length

    # TRAINING
    
    # # Online training 
    # # Loop over episodes
    # ave_loss_list = []
    # for ep in range(num_episodes):
    #     # Reset the environment for the start of the episode.
    #     agent.reset()
    #     loss_list = []
    #     gamma = 0
    #     # Loop over steps within this episode. The episode length here is 20.
    #     for step_num in range(episode_length):
    #         # Step the agent once, and get the transition tuple for this step
    #         transition = agent.step()
    #         print('Episode: ', ep, '\nTransition ',step_num, ':', transition,'\n')
    #         loss = dqn.train_q_network(transition, 'online', gamma) # calculate loss for single step
    #         # Sleep, so that you can observe the agent moving. Note: this line should be removed when you submit the code in your coursework, to speed up training
    #         loss_list.append(loss) # append loss to list
    #     ave_loss = np.mean(loss_list)
    #     ave_loss_list.append(ave_loss)
    # plt.plot(range(len(ave_loss_list)),ave_loss_list)
    # plt.yscale('log')
    # plt.grid()
    # plt.xlabel('number of episodes')
    # plt.ylabel('average loss')
    # plt.title('Loss Curve for Online Learning')
    # plt.savefig('loss_online.png', dpi=500)

        
    # Minibatch training
    # ave_loss_list = []
    # buffer = ReplayBuffer() # create a new replay buffer to store transitions
    # for episode in range(num_episodes):
    #     # Reset the environment for the start of the episode.
    #     agent.reset()
    #     loss_list = []
    #     minibatch_length = 100
    #     gamma = 0.9 # discount factor
    #     print('Episode: ', episode)
    #     # Loop over steps within this episode. The episode length here is 20.
    #     for step_num in range(episode_length):
    #         # Step the agent once, and get the transition tuple for this step
    #         transition = agent.step()
    #         print('Episode: ', episode, '\nTransition ', step_num, ':', transition,'\n')
    #         buffer.append_transition_tuple(transition)
    #         if buffer.get_length() >= minibatch_length:
    #             minibatch = buffer.sample_minibatch(buffer.get_length(), minibatch_length) # sample minibatch
    #             loss = dqn.train_q_network(minibatch,'minibatch',gamma) # calculate loss for minibatch
    #             loss_list.append(loss) # append minibatch loss to list
    #     ave_loss = np.mean(loss_list) # calculate average loss for episode
    #     ave_loss_list.append(ave_loss) # append average loss to list 
    #             # train network on minibatches
    #         # Sleep, so that you can observe the agent moving. Note: this line should be removed when you submit the code in your coursework, to speed up training
    #         # time.sleep(0.2)
    # plt.plot(range(len(ave_loss_list)),ave_loss_list)
    # plt.yscale('log')
    # plt.grid()
    # plt.xlabel('number of episodes')
    # plt.ylabel('average loss')
    # plt.title('Loss Curve for Buffer, Bellman equation ($\gamma$ = 0.9) and no Target Network')
    # plt.savefig('loss_replay_bellman_no_target.png', dpi=500)
    
    # Bellman equation with target network
    # Minibatch training
    ave_loss_list = []
    buffer = ReplayBuffer() # create a new replay buffer to store transitions
    visualiser = QValueVisualiser(environment)
    for episode in range(num_episodes):
        # Reset the environment for the start of the episode.
        agent.reset()
        loss_list = []
        minibatch_length = 100
        gamma = 0.9 # discount factor
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(episode_length):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step(episode)
            # print('Episode: ', episode, '\nTransition ', step_num, ':', transition,'\n')
            buffer.append_transition_tuple(transition)
            if buffer.get_length() >= minibatch_length:
                minibatch = buffer.sample_minibatch(buffer.get_length(), minibatch_length) # sample minibatch
                loss = dqn.train_q_network(minibatch,'minibatch',gamma) # calculate loss for minibatch
                loss_list.append(loss) # append minibatch loss to list
        # if step_num%10 == 0:
        #     dqn.update_target_network()
        visualiser.draw_q_values(dqn.get_Q_vals())
        ave_loss = np.mean(loss_list) # calculate average loss for episode
        print('Episode: ', episode, '\nLoss: ', ave_loss)
        ave_loss_list.append(ave_loss) # append average loss to list 
        if episode%10 == 0:
            dqn.update_target_network()
        # train network on minibatches
            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you submit the code in your coursework, to speed up training
            # time.sleep(0.2)
    plt.plot(range(len(ave_loss_list)),ave_loss_list)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('number of episodes')
    plt.ylabel('average loss')
    plt.title('Loss Curve for Buffer, Bellman equation ($\gamma$ = 0.9) and Target Network')
    plt.savefig('loss_replay_bellman_target_epsilon_greedy_5.png', dpi=500)
    
    
    visualiser.save_Q_image('epsilon_greedy_Q_vals_5')

    
    #%% CALCULATE Q VALUES USING TRAINED NETWORK AND VISUALISE
    q_values = np.zeros([10,10,4]) # create 3D array to hold q values
    for row in range(10):
        for col in range(10):
            state = torch.tensor([row/10 + 0.05,col/10 + 0.05])
            for action in range(4):
                predicted_rewards = dqn.q_network.forward(torch.unsqueeze(state,0))
                q_values[row,col,action] = predicted_rewards[0, action]
    print(q_values)
    
    # CALCULATE OPTIMAL POLICY AND DRAW GREEDY POLICY
    policy = []
    current_state = torch.tensor(agent.environment.init_state) 
    for step_num in range(20):
        policy.append(current_state.detach().numpy())
        predictions = dqn.q_network.forward(torch.unsqueeze(current_state,0))
        discrete_action = torch.argmax(predictions, 1)
        continuous_action = agent._discrete_action_to_continuous(discrete_action)
        next_state, distance_to_goal = agent.environment.step(current_state, continuous_action)
        current_state = next_state
        
    agent.environment.draw_policy(policy)
    
    
    
    
    
    