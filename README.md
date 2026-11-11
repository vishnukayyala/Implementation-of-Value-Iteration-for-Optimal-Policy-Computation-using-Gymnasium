
# Implementation of Value Iteration for Optimal Policy Computation Using Gymnasium

## Aim

To implement the Value Iteration algorithm using the Gymnasium FrozenLake-v1 environment to calculate the optimal state-value function and determine the optimal policy for a finite Markov Decision Process (MDP).

## Problem Statement

The objective of this experiment is to solve the FrozenLake-v1 environment using the Value Iteration algorithm.

The agent starts from an initial position and must reach the goal while avoiding holes. The environment is represented as a finite Markov Decision Process.

Value Iteration calculates the optimal value of every state by repeatedly applying the Bellman optimality equation. Using these values, the agent determines the best action to perform in each state.

The main objectives are:

1. Initialize the state-value function.
2. Calculate the expected returns for all possible actions.
3. Update state values using the maximum expected return.
4. Repeat the updates until convergence.
5. Extract the optimal policy from the final value function.

## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

**Requirements:**
- Python 3.x
- Gymnasium
- NumPy
- Jupyter Notebook or any Python IDE

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-based environment where an agent moves across frozen tiles to reach a goal while avoiding holes.

The default environment contains a 4 × 4 grid.

| Component | Description |
|---|---|
| Environment | FrozenLake-v1 |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Action 0 | Left |
| Action 1 | Down |
| Action 2 | Right |
| Action 3 | Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and holes |

**Environment layout:**

```text
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

Where:

- S = Starting position
- F = Frozen surface
- H = Hole
- G = Goal

## Theory

Value Iteration is a Dynamic Programming algorithm used in reinforcement learning to find the optimal policy of a Markov Decision Process.

It repeatedly updates the value of each state by selecting the action that provides the maximum expected future reward.

Unlike Policy Iteration, Value Iteration combines policy evaluation and policy improvement into a single update process.

### Bellman Optimality Equation

The Bellman optimality equation calculates the maximum expected return for each state.

\[
V^*(s)=\max_a\sum_{s'}P(s'|s,a)
[R(s,a,s')+\gamma V^*(s')]
\]

Where:

| Symbol | Meaning |
|---|---|
| \(s\) | Current state |
| \(a\) | Available action |
| \(s'\) | Next state |
| \(P(s'|s,a)\) | Transition probability |
| \(R(s,a,s')\) | Immediate reward |
| \(\gamma\) | Discount factor |
| \(V^*(s)\) | Optimal value of the state |

### Value Iteration Update

The value function is initialized with zeros and updated repeatedly using:

\[
V_{k+1}(s)=
\max_a\sum_{s'}P(s'|s,a)
[R(s,a,s')+\gamma V_k(s')]
\]

Here, \(k\) represents the current iteration.

The algorithm continues updating values until the maximum difference between consecutive iterations becomes smaller than a predefined threshold.

\[
\Delta=\max_s|V_{k+1}(s)-V_k(s)|
\]

The convergence condition is:

\[
\Delta<\theta
\]

Where \(\theta\) is the convergence threshold.

### Optimal Policy Extraction

After the value function converges, the optimal policy is obtained by selecting the action with the highest expected return.

\[
\pi^*(s)=\arg\max_a
\sum_{s'}P(s'|s,a)
[R(s,a,s')+\gamma V^*(s')]
\]

The resulting policy specifies the best action for every state.

## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the environment's transition probabilities.
3. Initialize the state-value function with zeros.
4. Set the discount factor and convergence threshold.
5. For every state:
   - Calculate the expected return for each available action.
   - Consider all possible next states and their transition probabilities.
   - Select the maximum expected return.
6. Update the state-value function using the Bellman optimality equation.
7. Calculate the maximum change between consecutive value estimates.
8. Repeat the process until the convergence condition is satisfied.
9. Extract the optimal policy by selecting the action with the highest expected return for each state.
10. Display the optimal state-value function and policy.

## Python Program

```python
import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)
P = env.unwrapped.P

# Parameters
gamma = 0.99
theta = 1e-8

n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize value function
V = np.zeros(n_states)

# Identify terminal states
desc = env.unwrapped.desc.flatten()
terminal = [
    s for s in range(n_states)
    if desc[s] in (b'H', b'G')
]

# Value Iteration
iterations = 0

while True:
    delta = 0
    new_V = V.copy()

    for s in range(n_states):

        if s in terminal:
            new_V[s] = 0
            continue

        action_values = []

        for a in range(n_actions):
            value = 0

            for prob, next_s, reward, done in P[s][a]:
                next_value = 0 if done else V[next_s]

                value += prob * (
                    reward + gamma * next_value
                )

            action_values.append(value)

        new_V[s] = max(action_values)

        delta = max(
            delta,
            abs(new_V[s] - V[s])
        )

    V = new_V
    iterations += 1

    if delta < theta:
        break

# Extract optimal policy
policy = np.zeros(n_states, dtype=int)

for s in range(n_states):

    if s in terminal:
        continue

    action_values = []

    for a in range(n_actions):
        value = 0

        for prob, next_s, reward, done in P[s][a]:
            next_value = 0 if done else V[next_s]

            value += prob * (
                reward + gamma * next_value
            )

        action_values.append(value)

    policy[s] = np.argmax(action_values)

# Display results
actions = ['L', 'D', 'R', 'U']

print("Total Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

print("\nOptimal Policy:")
print(np.array(actions)[policy].reshape(4, 4))

env.close()
```

## Output

The program displays the number of iterations required for convergence, the optimal state-value function, and the optimal policy.

**Expected output format:**

```text
Total Iterations: [Calculated by program]

Optimal State-Value Function:

[[0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]]

Optimal Policy:

[['L' 'D' 'R' 'U']
 ['L' 'D' 'R' 'U']
 ['L' 'D' 'R' 'U']
 ['L' 'D' 'R' 'U']]
```

The values and actions above are placeholders showing the output format. The actual results are calculated when the program is executed.

**Action symbols:**

- L = Left
- D = Down
- R = Right
- U = Up

## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

The optimal state-value function was calculated using repeated applications of the Bellman optimality equation. After convergence, the optimal policy was extracted by selecting the action with the highest expected return in each state.

## Inference

- Value Iteration is used to calculate the optimal state-value function of an MDP.
- It uses the Bellman optimality equation to update state values.
- Every available action is evaluated to determine the maximum expected return.
- The discount factor controls the importance of future rewards.
- The convergence threshold determines when the value updates should stop.
- The optimal policy is extracted after the value function converges.
- Value Iteration combines policy evaluation and policy improvement into a single update process.

## About

Implementation of the Value Iteration algorithm for optimal policy computation using Python, NumPy, and Gymnasium.
