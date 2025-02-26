import logging
from config import BASE_DIR
from model import build_proposed_model, get_callbacks
from train import data_gen, train_model
from evaluation import plot_history, save_history

# Main execution function
def main():
    try:
        # Build the model
        model = build_proposed_model()
        
        # Data generation
        train_gen = data_gen()
        
        # Get callbacks
        callbacks = get_callbacks()

        # Train the model
        history = train_model(model, train_gen)
        
        # Save and plot the history
        save_history(history)
        plot_history(history)
        
        logging.info("Training complete!")
    except Exception as e:
        logging.error("Error in main execution: %s", e)

if __name__ == "__main__":
    main()

