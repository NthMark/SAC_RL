import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# from .common.config import GRAPH_AVERAGE_REWARD,PLOT_PATH
from matplotlib.ticker import MaxNLocator
GRAPH_AVERAGE_REWARD=10
PLOT_PATH="/home/mark/limo_ws/src/rl_sac/rl_sac/plot/"
matplotlib.use('Agg')
class Graph():
    def __init__(self):
        plt.show()
        self.legend_labels = ['Success', 'Collision Wall','Timeout']
        self.legend_colors = ['b', 'g', 'r']

        self.outcome_histories = [[],[],[]]
        self.global_steps = 0
        self.data_outcome_history = []
        self.data_rewards = []
        self.data_loss_critic = []
        self.data_loss_actor = []
        self.graphdata = [self.global_steps, self.data_outcome_history, self.data_rewards, self.data_loss_critic, self.data_loss_actor]

        self.fig, self.ax = plt.subplots(2, 2)
        self.fig.set_size_inches(18.5, 10.5)

        titles = ['outcomes', 'avg critic loss over episode', 'avg actor loss over episode', 'avg reward over 10 episodes']
        for i in range(4):
            ax = self.ax[int(i/2)][int(i%2!=0)]
            ax.set_title(titles[i])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.legend_set = False


    def update_data(self, step, outcome, reward_sum, loss_critic_sum, loss_actor_sum):
        self.step = step
        self.data_outcome_history.append(outcome)
        self.data_rewards.append(reward_sum)
        self.data_loss_critic.append(loss_critic_sum / step)
        self.data_loss_actor.append(loss_actor_sum / step)

    def draw_plots(self, episode):
        for ax_row in self.ax:
            for ax in ax_row:
                ax.clear()
        xaxis = np.array(range(episode + 1))
        print(episode)
        print(len(xaxis))
        # Plot outcome history
        if len(self.data_outcome_history) ==1:
            self.outcome_histories[0].append(1 if self.data_outcome_history[0][0]else 0)
            self.outcome_histories[1].append(1 if self.data_outcome_history[0][1]else 0)
            self.outcome_histories[2].append(1 if self.data_outcome_history[0][2]else 0)
        elif len(self.data_outcome_history)==2:
            self.outcome_histories[0].append(self.outcome_histories[0][-1]+self.data_outcome_history[-1][0])
            self.outcome_histories[1].append(self.outcome_histories[1][-1]+self.data_outcome_history[-1][1])
            self.outcome_histories[2].append(self.outcome_histories[2][-1]+self.data_outcome_history[-1][2])
        elif len(self.data_outcome_history)<=11:
            for idx in range(2,len(self.data_outcome_history)):
                self.outcome_histories[0].append(self.outcome_histories[0][-1]+self.data_outcome_history[idx][0])
                self.outcome_histories[1].append(self.outcome_histories[1][-1]+self.data_outcome_history[idx][1])
                self.outcome_histories[2].append(self.outcome_histories[2][-1]+self.data_outcome_history[idx][2])
        else:
            for idx in range(len(self.data_outcome_history)-10,len(self.data_outcome_history)):
                self.outcome_histories[0].append(self.outcome_histories[0][-1]+self.data_outcome_history[idx][0])
                self.outcome_histories[1].append(self.outcome_histories[1][-1]+self.data_outcome_history[idx][1])
                self.outcome_histories[2].append(self.outcome_histories[2][-1]+self.data_outcome_history[idx][2])
        if len(self.data_outcome_history) > 0:
            i = 0
            for outcome_history in np.array(self.outcome_histories):
                self.ax[0][0].plot(xaxis, outcome_history, color=self.legend_colors[i], label=self.legend_labels[i])
                i += 1
            self.ax[0][0].legend()

        # Plot critic loss
        y = np.array(self.data_loss_critic)
        self.ax[0][1].plot(xaxis, y)

        # Plot actor loss
        y = np.array(self.data_loss_actor)
        self.ax[1][0].plot(xaxis, y)

        # Plot average reward
        count = int(episode / GRAPH_AVERAGE_REWARD)
        if count > 0:
            xaxis = np.array(range(GRAPH_AVERAGE_REWARD, episode+1, GRAPH_AVERAGE_REWARD))
            averages = list()
            for i in range(count):
                avg_sum = 0
                for j in range(GRAPH_AVERAGE_REWARD):
                    avg_sum += self.data_rewards[i * GRAPH_AVERAGE_REWARD + j]
                averages.append(avg_sum / GRAPH_AVERAGE_REWARD)
            y = np.array(averages)
            self.ax[1][1].plot(xaxis, y)

        plt.draw()
        plt.pause(0.2)
        plt.savefig(os.path.join(PLOT_PATH, f"{episode}_figure.png"))
    # def get_success_count(self):
    #     suc = self.data_outcome_history[-GRAPH_DRAW_INTERVAL:]
    #     return suc.count(SUCCESS)

    # def get_reward_average(self):
    #     rew = self.data_rewards[-GRAPH_DRAW_INTERVAL:]
    #     return sum(rew) / len(rew)
if __name__=='__main__':
    graph=Graph()
    graph.update_data(1,[0,0,0],5,2,2)
    graph.draw_plots(0)
    graph.update_data(1,[1,1,1],5,2,2)
    graph.draw_plots(1)
