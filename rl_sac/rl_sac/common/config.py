action_dim = 2
data_map_length=331776
state_dim  = 16+data_map_length
hidden_dim = 512
ACTION_V_MIN = 0.0 # m/s
ACTION_W_MIN = -2. # rad/s
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2. # rad/s
EPISODE, MAX_STEP_SIZE = 7000, 3000
THRESHOLD_TRAINING=10000
BATCH_SIZE=200

GRAPH_DRAW_INTERVAL      = 10       # Draw the graph every N episodes (drawing too often will slow down training)
GRAPH_AVERAGE_REWARD     = 10       # Average the reward graph over every N episodes

PLOT_PATH="/home/mark/limo_ws/src/rl_sac/rl_sac/plot"

ENABLE_VISUAL = False