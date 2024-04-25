import random
import time
from puzzle import State, move, num_solved_sides, num_pieces_correct_side, shuffle, n_move_state, one_move_state
import numpy as np
import pprint as pp


# q-values key = (state, action) => value = (q-value, update_count)
class Agent:

    # initialize agent, can be passed a dictionary of Q-Values
    # if it already exists, and a cube, otherwise, initializes new
    # cube if not provided one
    def __init__(self, QValues={}, scramble_depth: int = 3):
        # maps a state action pair to a Q-Value, and an update count for that Q-Value
        self.QV = QValues
        # create or store initial cube state, and store list of actions
        self.start_state = n_move_state(n=scramble_depth)
        self.depth = scramble_depth
        # print(self.start_state)
        self.curr_state = self.start_state.copy()
        self.actions = self.start_state.actions
        self.move = {"front": 0, "back": 0, "left": 0, "right": 0, "top": 0, "bottom": 0, "afront": 0, "aback": 0,
                     "aleft": 0, "aright": 0, "atop": 0, "abottom": 0}

    def adi(self, to_depth: int = 1, reward_coefficient: float = 1.0) -> None:
        goal_state = State()
        states_with_rewards = []
        states_with_rewards.append([goal_state])
        qvTable = {goal_state.__hash__(): np.zeros(12)}

        for d in range(1, to_depth + 1):
            reward = (to_depth + 1 - d)

            states_at_depth = []
            for s in states_with_rewards[d - 1]:
                for action in self.actions:
                    s_ = move(s, action)

                    if s_.__hash__() not in qvTable.keys():
                        states_at_depth.append(s_)
                        goodAction = self.reverse_action(action)
                        qvTable.update(
                            {s_.__hash__(): np.full(12, -1 * reward)})
                        qvTable[s_.__hash__()][self.actions.index(
                            goodAction)] = reward
            states_with_rewards.append(states_at_depth)

        self.QV = qvTable

    # def register_patterns(self, to_depth: int = 1, with_reward_coefficient: float = 1.0) -> None:
    #     # list of dictionaries, each dictionary a depth distance from the goal state
    #     states_with_rewards = []
    #     goal_state = State()
    #     states_with_rewards.append(
    #         {goal_state: {(goal_state.__hash__(), None): to_depth * with_reward_coefficient * 10}})
    #     for d in range(1, to_depth + 1):
    #         states_to_rewards_at_this_depth = {}
    #         reward = (to_depth + 1 - d) * with_reward_coefficient

    #         for s in states_with_rewards[d - 1]:
    #             for good_action in self.actions:
    #                 s_ = move(s, good_action)

    #                 good_action = self.reverse_action(good_action)

    #                 states_to_rewards_at_this_depth[s_] = {
    #                     (s_.__hash__(), good_action): reward}
    #                 for bad_action in self.actions:
    #                     if bad_action != good_action and (s_.__hash__(), bad_action) not in states_to_rewards_at_this_depth[s_]:
    #                         states_to_rewards_at_this_depth[s_][(
    #                             s_.__hash__(), bad_action)] = -1*reward
    #             states_with_rewards.append(states_to_rewards_at_this_depth)

    #     for state_with_reward in reversed(states_with_rewards):
    #         for state, state_action_rewards in state_with_reward.items():
    #             self.QV.update(state_action_rewards)

    def reverse_action(self, action):
        if action[0] == 'a':
            return action[1:]
        else:
            return f'a{action}'

    # explore
    def QLearn(self, epochs=100, gamma=0.99, steps=20, epsilon=0.9, eta=0.6, depth_from_baseline=1):
        # execute q learning for specified number of episodes
        steps_required = np.zeros(epochs)
        Epsilons = [i / epochs for i in range(epochs)]
        Epsilons.reverse()
        for j in range(epochs):
            self.curr_state = n_move_state(
                n=self.depth + depth_from_baseline)  # six_move_state()
            for i in range(steps):
                if not (self.curr_state.__hash__()) in self.QV.keys():
                    self.QV.update({self.curr_state.__hash__(): np.zeros(12)})
                # Observe current state
                state = self.curr_state.copy()
                # Choose an action using epsilon greedy
                action = self.chooseAction(Epsilons[j])
                # Perform the action and receive reward
                reward = self.reward(state, action)
                self.curr_state.move(self.actions[action])
                if not (self.curr_state.__hash__()) in self.QV.keys():
                    self.QV.update({self.curr_state.__hash__(): np.zeros(12)})
                # Update Q Table
                self.QV[state.__hash__()][action] = self.QV[state.__hash__()][action] + eta * (
                    reward + gamma*np.max(self.QV[self.curr_state.__hash__()]) - self.QV[state.__hash__()][action])

                # Check for end state
                if self.curr_state.isGoalState():
                    steps_required[j] = i
                    break
            if steps_required[j] == 0:
                steps_required[j] = steps
        return steps_required

    def chooseAction(self, epsilon=0):
        if np.random.rand() < (1 - epsilon):
            action = np.argmax(self.QV[self.curr_state.__hash__()])
        else:
            action = np.random.randint(0, 11)
        return action

    def Play(self, max_steps=20, depth_from_baseline=1, tests=1):
        test_steps = np.zeros(tests)
        for j in range(tests):
            self.curr_state = n_move_state(n=self.depth + depth_from_baseline)

            for i in range(max_steps):
                # If the current state is not in the QV table
                if not (self.curr_state.__hash__()) in self.QV.keys():
                    self.QV.update({self.curr_state.__hash__(): np.zeros(12)})

                action = self.chooseAction()
                self.curr_state.move(self.actions[action])

                if self.curr_state.isGoalState():
                    test_steps[j] = i
                    break
            if test_steps[j] == 0:
                test_steps[j] = max_steps
        return test_steps

    def reward(self, state, action):
        next_state = move(state, self.actions[action])
        if next_state.isGoalState():
            return 100
        else:
            return 0


def saveData(trainingData, testData, title):
    np.savetxt(f'{title}_training.txt', trainingData, fmt='%d')
    np.savetxt(f'{title}_test.txt', testData, fmt='%d')
    print(f'{title} saved.')


def score(testData, max_steps=20):
    return np.count_nonzero(testData < max_steps)


def experiment():
    # Due to compute time, train_depth = 4 was chosen
    train_depth = 4
    trainingEpochs = 5000
    testCases = 1000

    # etas = [i / 10 for i in range(10)]
    # gammas = [i / 10 for i in range(10)]
    test_depths = [i for i in range(10)]
    eta = 0.3
    gamma = 0.9

    # eta_score = np.zeros(10)
    # for i, eta in enumerate(etas):
    #     agent = Agent(scramble_depth=train_depth)
    #     agent.adi(train_depth)
    #     training_steps = agent.QLearn(
    #         epochs=trainingEpochs, steps=60, eta=eta, depth_from_baseline=2)
    #     test_steps = agent.Play(depth_from_baseline=2, tests=testCases)
    #     saveData(training_steps, test_steps, f'eta_{eta}')
    #     eta_score[i] = score(test_steps)
    # eta = etas[np.argmax(eta_score)]
    # print(f'Best eta = {eta}')

    # gamma_score = np.zeros(10)
    # for i, gamma in enumerate(gammas):
    #     agent = Agent(scramble_depth=train_depth)
    #     agent.adi(train_depth)
    #     training_steps = agent.QLearn(
    #         epochs=trainingEpochs, steps=60, eta=eta, gamma=gamma, depth_from_baseline=2)
    #     test_steps = agent.Play(depth_from_baseline=2, tests=testCases)
    #     saveData(training_steps, test_steps, f'eta_{eta}_gamma_{gamma}')
    #     gamma_score[i] = score(test_steps)
    # gamma = gammas[np.argmax(gamma_score)]
    # print(f'Best gamma = {gamma}')

    depth_score = np.zeros(10)
    for i, depth in enumerate(test_depths):
        agent = Agent(scramble_depth=train_depth)
        # agent.adi(train_depth)
        training_steps = agent.QLearn(
            epochs=trainingEpochs, steps=80, eta=eta, gamma=gamma, depth_from_baseline=depth)
        test_steps = agent.Play(
            depth_from_baseline=depth, tests=testCases)
        saveData(training_steps, test_steps,
                 f'eta_{eta}_gamma_{gamma}_testdepth_{depth}_withoutADI')
        depth_score[i] = score(test_steps)
    depth = test_depths[np.argmax(depth_score)]
    print(f'Best depth = {depth}')


if __name__ == '__main__':
    experiment()
    # train_depth = 3
    # agent = Agent(scramble_depth=train_depth)
    # print("REGISTERING PATTERN DATABASE, THIS WILL TAKE A LITTLE WHILE")

    # agent.adi(train_depth, 1.0)
    # # training_steps = np.zeros(training_episodes)
    # # test_steps = np.zeros(test_episodes)
    # # Epsilons = [i / training_episodes for i in range(training_episodes)]
    # # Epsilons.reverse()

    # training_steps = agent.QLearn(epochs=100, steps=60)
    # test_steps = agent.Play(max_steps=20, depth_from_baseline=1, tests=20)

    # # print(training_steps)
    # print(test_steps)

    # TODO: Randomize the start training and start test cubes to figure out how many can actually be trained
