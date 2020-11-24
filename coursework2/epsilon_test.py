import matplotlib.pyplot as plt

epsilon = []
epsilon_decay = 0.99
episode_length= 300
steps_in_episode = 160
N_eps = 0

for i in range(episode_length*steps_in_episode):
    if i%episode_length == 0:
        N_eps +=1
    # epsilon.append(epsilon_decay**(N_eps*(1 + 0.004*(episode_length - i%episode_length))))
    epsilon.append(epsilon_decay**(N_eps) + N_eps * (episode_length - i%episode_length)/105000)

plt.plot(range(len(epsilon)),epsilon)
plt.ylabel('$\epsilon$')
plt.xlabel('number of steps')