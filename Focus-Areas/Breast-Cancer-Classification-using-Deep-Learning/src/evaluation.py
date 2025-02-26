import logging
import matplotlib.pyplot as plt
import pandas as pd
import json

# Model evaluation and plotting
def plot_history(history):
    try:
        history_df = pd.DataFrame(history.history)
        history_df[['accuracy', 'val_accuracy']].plot()
        history_df[['loss', 'val_loss']].plot()
        plt.show()
    except Exception as e:
        logging.error("Error during plotting history: %s", e)
        raise

# Save training history
def save_history(history):
    try:
        with open('history.json', 'w') as f:
            json.dump(str(history.history), f)
    except Exception as e:
        logging.error("Error saving training history: %s", e)
        raise

