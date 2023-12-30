# Q-Learning

***Q-Learning algorithm for solving the Frozen Lake Problem.***

Given the environment that can be interacted with via a specific set of actions,
Q-Learning lets the agent *learn* how to effectively function in it by choosing
actions (randomly at first) and observing their outcomes.

Little by little, the agent understands which actions are beneficial to take in
a given situation, considering not only the immediate reward but also potential
future rewards and the ability to eventually reach the final goal.

## Vanilla Q-Learning

The most straightforward implementation of Q-Learning is to use a table that
maps all possible states to available actions and their respective Q-values.
Agent's farsightedness is realized by considering the maximum Q-value of the
next state when updating the current state's Q-value using the Bellman equation.

### Training

![Vanilla Training](/vanilla/training.gif)

### Testing

![Vanilla Testing](/vanilla/result.gif)

## Deep Q-Learning

Vanilla, table-based Q-Learning is hardly applicable to environments with large
state spaces. Also, it's harder for it to adapt to environments that are not
fully determined and where random action outcomes are sometimes possible.

To overcome these limitations, Deep Q-Learning was introduced, in which the
Q-Table is replaced with a neural network, able to generalize the Q-values
for unseen states and discover the patterns of much higher complexity.

Two neural networks are used to train the agent: the *actual*, which directly
determines the agent's behavior, and the *ideal*, which is needed to compute
loss and perform optimization. *Experience Replay* is used to decorrelate
the training samples and stabilize the overall training process.

### Result

![DQN Testing](/deep/result.gif)
