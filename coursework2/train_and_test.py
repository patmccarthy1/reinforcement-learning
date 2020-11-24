import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import numpy as np

from random_environment import Environment
from agent import Agent

import matplotlib.pyplot as plt

def main(EPISODE_LENGTH, MINIBATCH_SIZE, DISCOUNT, EPSILON_DECAY, TARGET_UPDATE):
    
    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    agent = Agent(EPISODE_LENGTH, MINIBATCH_SIZE, DISCOUNT, EPSILON_DECAY, TARGET_UPDATE)

    # keep track of episode
    episode = 0
    
    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    loss_list = []
    ave_loss_list = []
    
    # fig = plt.gcf()
    # fig.show()
    # fig.canvas.draw()
        
    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            time_passed = time.time() - start_time
            episode += 1
            # mins = int(np.floor(time_passed/60))
            # secs = int(time_passed % 60)
            # print('Episode: ', episode)
            # print('Time passed: {} mins {} secs'.format(mins,secs))
            ave_loss = np.mean(loss_list)
            ave_loss_list.append(ave_loss)
            # plt.plot(range(episode),ave_loss_list)
            # plt.xlabel('episode number')
            # plt.ylabel('average loss')
            # plt.yscale('log')
            # plt.title('average loss vs episode number (log scale)')
            state = environment.init_state
            # fig.canvas.draw()
            # plt.pause(0.0001)
            loss_list = []
        
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        loss = agent.set_next_state_and_distance(next_state, distance_to_goal)
        loss_list.append(loss)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on:
            environment.show(state)
    
    plt.plot(range(episode),ave_loss_list)
    plt.xlabel('episode number')
    plt.ylabel('average loss')
    plt.yscale('log')
    plt.title('average loss vs episode number (log scale)')   
    plt.savefig('eplen{}_mbsize{}_disc{}_epdec{}_targup{}.png'.format(EPISODE_LENGTH, MINIBATCH_SIZE, DISCOUNT, EPSILON_DECAY, TARGET_UPDATE),dpi=1000)
    
    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))

if __name__ == "__main__":
    import sys
    main(EPISODE_LENGTH, BATCH_SIZE, DISCOUNT, EPSILON_DECAY, TARGET_UPDATE)
    