# Implementation of Value Iteration for Optimal Policy Computation using Gymnasium

## Aim

To implement the Value Iteration algorithm for solving a finite Markov Decision Process (MDP) using the Gymnasium **FrozenLake-v1** environment and compute the optimal state-value function and optimal policy using the Bellman Optimality Equation.

---

## Problem Statement

In this experiment, the **FrozenLake-v1** environment is solved using the **Value Iteration** algorithm.

The agent starts from the initial state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process (MDP). Value Iteration repeatedly updates the value of each state using the Bellman Optimality Equation until the values converge.

The objectives are to compute:

- The optimal state-value function, $V^*(s)$
- The optimal policy, $\pi^*(s)$

---

## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the **Gymnasium FrozenLake-v1** environment.

FrozenLake is a grid-world reinforcement learning environment where the agent moves over frozen tiles and attempts to reach the goal without falling into holes.

For the default **4 × 4 FrozenLake** map:

| Component | Description |
|-----------|-------------|
| Environment | FrozenLake-v1 |
| Map Size | 4 × 4 |
| Observation Space | 16 discrete states |
| Action Space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal States | Goal state and Hole states |

---

## MDP Representation

A Markov Decision Process is represented as:

$$
MDP = (S, A, P, R, \gamma)
$$

Where:

| Symbol | Meaning |
|--------|---------|
| $S$ | Set of states |
| $A$ | Set of actions |
| $P$ | Transition probability function |
| $R$ | Reward function |
| $\gamma$ | Discount factor |

The FrozenLake environment contains **16 states**, **4 actions**, probabilistic state transitions, rewards, and a discount factor used to compute future rewards.

---

## Theory

Value Iteration is a **Dynamic Programming** algorithm used to compute the optimal state-value function and the optimal policy of a Markov Decision Process.

Instead of evaluating a fixed policy, Value Iteration directly computes the optimal value of each state by repeatedly applying the **Bellman Optimality Equation** until the value function converges.

The Bellman Optimality Equation is:


```math
V(s) = \max_{a} \sum_{s'} P(s' \mid s, a)
\left[ R(s, a, s') + \gamma V(s') \right]
```

Where:

| Symbol | Meaning |
|--------|---------|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^*(s)$ | Optimal state-value function |

After the value function converges, the optimal policy is obtained by selecting the action that produces the highest expected value for every state.

---

## Algorithm

1. Create the **FrozenLake-v1** environment using Gymnasium.
2. Initialize the value function $V(s)=0$ for all states.
3. Repeat until convergence:
   - Compute the value of each action.
   - Update every state's value using the Bellman Optimality Equation.
   - Store the maximum action value.
4. Stop when the change in state values becomes smaller than the convergence threshold.
5. Derive the optimal policy by selecting the action with the highest value for every state.
6. Display the optimal state-value function and the optimal policy.

---

## Python Program

```python
# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration and returns the optimal
    value function and optimal policy.
    """

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)

    iteration = 0

    while True:

        delta = 0

        for state in range(n_states):

            old_value = V[state]

            action_values = np.zeros(n_actions)

            for action in range(n_actions):

                for prob, next_state, reward, done in env.P[state][action]:

                    action_values[action] += prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            V[state] = np.max(action_values)

            delta = max(delta, abs(old_value - V[state]))

        iteration += 1

        if delta < theta:
            break

    policy = np.zeros(n_states, dtype=int)

    for state in range(n_states):

        action_values = np.zeros(n_actions)

        for action in range(n_actions):

            for prob, next_state, reward, done in env.P[state][action]:

                action_values[action] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )

        policy[state] = np.argmax(action_values)

    return V, policy, iteration

```
---

## Output

<img width="561" height="372" alt="image" src="https://github.com/user-attachments/assets/854f52ab-0ce1-4496-bfc4-d4765341e368" />



---

## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed using the Bellman Optimality Equation.

---

## Inference

The Value Iteration algorithm successfully computed the optimal state-value function and the optimal policy for the FrozenLake environment.
The state values converged after repeated updates using the Bellman Optimality Equation.
This demonstrates that Value Iteration efficiently solves a finite Markov Decision Process by directly computing the optimal policy.
