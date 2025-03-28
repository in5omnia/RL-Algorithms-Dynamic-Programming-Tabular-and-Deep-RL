
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the Every-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and Every-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) Every-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / Every-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Every-Visit Monte Carlo evaluates complete trajectories and updates value estimates using the total return, the sum of rewards discounted over time, making it heavily influenced by gamma, the discount factor. Since in FrozenLake the reward is sparse and the episodes are long, reaching distant goals heavily depends on gamma. In contrast, Q-learning is less affected by gamma, because it incrementally updates Q-values based on one-step lookahead, bootstrapping from the next state’s estimated Q-value. Since it applies gamma once at each timestep, rather than to the full return, its effect is more gradual and further diluted by alpha."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer

def question2_5() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) on the differences between the non-slippery and the slippery varian of the problem.
    by [Q-learning / Every-Visit Monte Carlo].
    return: answer (str): your answer as a string (100 words max)
    """
    answer = ""  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In the DiscreteRL algorithm, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    # Mean over 11 runs:
    # for lr = 2e-2: -181.825272727
    # for lr = 2e-3: -179.813181818
    # for lr = 2e-4: -180.677181818
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The exploration fraction defines the percentage of training progress at which exploration should reach its minimum, making it relative to the environment’s episode count. This ensures a predictable and consistent reduction in exploration, that adapts to different environments. Meanwhile, exponential decay, which reduces epsilon at a fixed rate, can lead to issues: it can drop exploration too quickly, limiting learning, or too slowly, making training inefficient. Since a decay strategy based on the exploration fraction ties exploration directly to training progress, offering greater adaptability, it may be more generally applicable across different environments."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "In supervised learning, models are trained on static, labelled datasets, facilitating convergence and a steady decrease in loss. In contrast, DQN faces a dynamic data distribution. As the agent learns, it generates new experiences that alter the data sampled for training. Target values also shift due to updates in the target network. Furthermore, the replay buffer, from which experiences are sampled for each network update, introduces variability as different subsets are used. This combination of newly generated experiences, non-stationary targets, and random sampling contributes to training instability, reflected in a highly fluctuating loss. Moreover, as training progresses, the agent encounters novel high-reward state-action pairs, initially associated with low Q-values, and suboptimal pairs, previously deemed as optimal. This leads to substantial variations in the Q-values associated with these pairs, contributing to an overall increase in loss during training."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The target network's update frequency is 2000, which is the timestep intervals at which the spikes occur. In each update, the target network's parameters are synchronised with the primary Q-network, causing an abrupt shift in the target Q-value distribution. The primary network, trained to predict values based on the previous target network's outputs, cannot immediately adapt to these changes. This causes the MSE loss to temporarily increase, generating the spikes, when the predicted Q-values deviate from the newly updated targets. Over time, the primary network is trained to match the new target distribution, reducing the loss until the next update."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = ""  # TYPE YOUR ANSWER HERE (200 words max)
    return answer
