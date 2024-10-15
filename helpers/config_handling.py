import json
from threading import Lock

config_lock = Lock()

# Load the configuration
def load_config():
    try:
        with open('config.json', 'r') as config_file:
            config = json.load(config_file)
        return config
    except Exception as e:
        print(f"Error reading config.json: {e}")
        raise

# Update the configuration
def update_config(new_config):
    try:
        with config_lock:
            # Load the existing configuration
            current_config = load_config()

            # Update only the keys provided in the new config
            current_config.update(new_config)

            # Save the updated configuration
            with open('config.json', 'w') as config_file:
                json.dump(current_config, config_file, indent=4)
    except Exception as e:
        print(f"Error updating config: {e}")
        raise

# Load the initial models and configurations
def load_initial_data(initialize):
    try:
        with config_lock:
            init_data = initialize()  # Call to your initialize function
        return init_data
    except Exception as e:
        print(f"Error loading initial data: {e}")
        raise

# Reinitialize models
def reinitialize_models(initialize):
    try:
        load_initial_data(initialize)
        print("Models reinitialized successfully.")
    except Exception as e:
        print(f"Error during model reinitialization: {e}")
