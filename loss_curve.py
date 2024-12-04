import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # for easy rolling average

# Load the JSON file
with open('./fine_tuned_llama_13b_acad/trainer_state.json', 'r') as f:
    data = json.load(f)

# Extract steps and losses
steps = []
train_losses = []
eval_losses = []
current_train_loss = None

for log in data['log_history']:
    if 'step' in log:
        step = log['step']
        
        # Handle training loss
        if 'loss' in log:
            current_train_loss = log['loss']
            steps.append(step)
            train_losses.append(current_train_loss)
        
        # Handle eval loss
        if 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])

# Calculate rolling average (window of 5)
train_losses_smooth = pd.Series(train_losses).rolling(window=5, center=True).mean()
# Remove NaN values from the edges
train_losses_smooth = train_losses_smooth.dropna()
steps_smooth = steps[2:-2]  # Adjust steps to match the smoothed data

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(steps_smooth, train_losses_smooth, label='Training Loss', alpha=0.8, color='blue')

# Plot eval losses at their corresponding steps
eval_steps = [step for step in steps if step % 50 == 0]  # Eval was done every 50 steps
plt.plot(eval_steps, eval_losses, label='Evaluation Loss', alpha=0.8, color='orange')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Set fixed y-axis limits
plt.ylim(1.0, 1.8)

plt.savefig('loss_curves.png')
plt.show()