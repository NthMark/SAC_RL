from .config import ENABLE_VISUAL
if ENABLE_VISUAL:
    from PyQt5 import QtWidgets
    import pyqtgraph as pg
    import numpy as np
    import sys

    pg.setConfigOptions(antialias=False)

    class DrlVisual(pg.GraphicsLayoutWidget):  # Updated from GraphicsView to GraphicsLayoutWidget
        def __init__(self):
            super().__init__()
            self.show()
            self.resize(1980, 1200)

            # Create layout for organizing different plots
            self.setWindowTitle('Deep Reinforcement Learning Visualization')

            # Action Linear Plot
            self.plot_item_action_linear = self.addPlot(title="Action Linear")
            self.plot_item_action_linear.setXRange(-20, 20, padding=0)
            self.plot_item_action_linear.setYRange(-0.22, 0.22, padding=0)
            self.bar_graph_action_linear = pg.BarGraphItem(x=[0], height=[0], width=0.5)  # Specify height with initial value
            self.plot_item_action_linear.addItem(self.bar_graph_action_linear)

            # Action Angular Plot
            self.nextRow()  # Move to the next row for the next plot
            self.plot_item_action_angular = self.addPlot(title="Action Angular")
            self.plot_item_action_angular.setXRange(-1, 1, padding=0)
            self.plot_item_action_angular.setYRange(-2, 2, padding=0)
            self.bar_graph_action_angular = pg.BarGraphItem(x=[0], height=[0], width=0.5)  # Specify height with initial value
            self.plot_item_action_angular.addItem(self.bar_graph_action_angular)

            # Accumulated Reward Plot
            self.nextRow()  # Move to the next row for the next plot
            self.plot_item_reward = self.addPlot(title="Accumulated Reward")
            self.plot_item_reward.setXRange(-1, 1, padding=0)
            self.plot_item_reward.setYRange(-3000, 5000, padding=0)
            self.bar_graph_reward = pg.BarGraphItem(x=[0], height=[0], width=0.5)  # Specify height with initial value
            self.plot_item_reward.addItem(self.bar_graph_reward)

            self.iteration = 0

        def update_action(self, actions):
            actions = actions.tolist()
            self.bar_graph_action_linear.setOpts(height=[actions[1]])
            self.bar_graph_action_angular.setOpts(height=[actions[0]])

        def update_reward(self, acc_reward):
            self.bar_graph_reward.setOpts(height=[acc_reward])
            if acc_reward > 0:
                self.bar_graph_reward.setOpts(brush='g')
            else:
                self.bar_graph_reward.setOpts(brush='r')


if __name__ == '__main__':
    if ENABLE_VISUAL:
        # Create a QApplication instance
        app = QtWidgets.QApplication(sys.argv)

        # Create an instance of the DrlVisual class
        visual = DrlVisual()

        # Run the application loop
        sys.exit(app.exec_())
