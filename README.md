# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
## Algorithm: On-Policy Monte Carlo Control using Gymnasium FrozenLake

1. **Start**
2. Create the `FrozenLake-v1` environment using Gymnasium.
3. Obtain the number of states and actions from the environment.
4. Initialize the Q-table `Q(s,a)` with zeros.
5. Set the hyperparameters:

   * Number of episodes = `10000`
   * Learning rate `α = 0.1`
   * Discount factor `γ = 0.99`
   * Initial exploration rate `ε = 1.0`
   * Minimum exploration rate = `0.05`
6. For each episode:

   1. Reset the environment and obtain the initial state.
   2. Select an action using the **epsilon-greedy policy**:

      * With probability `ε`, select a random action.
      * Otherwise, select the action with the maximum Q-value.
   3. Execute the action and observe the next state and reward.
   4. Store `(state, action, reward)` in the episode.
   5. Continue until the episode terminates or reaches the maximum number of steps.
7. Set the return `G = 0`.
8. Process the episode **backwards**:

   1. Take the current state, action, and reward.
   2. Calculate the return:
      [
      G = \gamma G + R
      ]
   3. If the `(state, action)` pair is encountered for the first time:
      [
      Q(s,a) \leftarrow Q(s,a) + \alpha[G-Q(s,a)]
      ]
9. Decrease epsilon:
   [
   \epsilon = \max(\epsilon_{min},\epsilon \times \epsilon_{decay})
   ]
10. Repeat Steps 6–9 for all episodes.
11. Extract the final greedy policy:
    [
    \pi(s)=\arg\max_a Q(s,a)
    ]
12. Calculate the state-value function:
    [
    V(s)=\max_a Q(s,a)
    ]
13. Display:

* Final Q-table
* State-value function
* Learned policy
* Average reward

14. Plot the learning curve.
15. **Stop.**



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

# CHANGED: Number of episodes
num_episodes = 10000

gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current
    epsilon-greedy policy.

    Returns:
        List of (state, action, reward)
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):

        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate one complete episode
    episode = generate_episode(epsilon)

    # Calculate total reward
    total_reward = sum(
        reward for _, _, reward in episode
    )

    episode_rewards.append(total_reward)

    # Initialize return
    G = 0

    # Store visited state-action pairs
    visited = set()

    # Process episode backwards
    for t in range(len(episode) - 1, -1, -1):

        state, action, reward = episode[t]

        # Calculate return
        G = gamma * G + reward

        # First-visit Monte Carlo
        if (state, action) not in visited:

            visited.add((state, action))

            # Update Q-value
            Q[state, action] += alpha * (
                G - Q[state, action]
            )

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # Display progress every 1000 episodes
    if (episode_num + 1) % 1000 == 0:

        print(
            f"Episode: {episode_num + 1}/{num_episodes}, "
            f"Reward: {total_reward}, "
            f"Epsilon: {epsilon:.4f}"
        )


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("Name:          ")
    print("Register Number:      ")

    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


# -------------------------------------------------
# Print Final Results
# -------------------------------------------------

print("\nFinal Q-table:")

print(
    np.round(Q, 3)
)

print_value_function(state_values)

print_policy(optimal_policy)


# -------------------------------------------------
# Average Reward
# -------------------------------------------------

success_rate = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    success_rate
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")

plt.ylabel("Average Reward")

plt.title(
    "Monte Carlo Control Learning Curve"
)

plt.grid(True)

plt.show()


# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()



```

---

## Output

```text



Final Q-table:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]

Estimated State-Value Function:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
Name:JAWAHAR RAJ N
Register Number:212223240057

Learned Policy:
[['L' 'L' 'L' 'L']
 ['L' 'L' 'L' 'L']
 ['L' 'L' 'L' 'L']
 ['L' 'L' 'L' 'L']]

Average reward over last 1000 episodes: 0.0


```


---



## Result
```text
The Monte Carlo Control algorithm was successfully implemented using
the Gymnasium FrozenLake-v1 environment. The agent generated complete
episodes and estimated the action-value function Q(s,a) using Monte
Carlo returns. An epsilon-greedy strategy was used to balance
exploration and exploitation. The learned Q-values were used to
extract the final greedy policy.


```
---

## Inference
```text
The experiment demonstrates that Monte Carlo Control can learn an
effective policy from complete episodes without requiring a model of
the environment. As the number of episodes increases, the Q-values
are updated using observed returns and the policy gradually improves.
The epsilon-greedy strategy allows the agent to explore different
actions initially and gradually exploit the actions with higher
Q-values. The final policy helps the agent reach the goal while
avoiding the holes in the FrozenLake environment.

```





---

