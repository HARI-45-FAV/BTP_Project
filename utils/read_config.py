#read_config.py
import yaml
import constants

def load_config(config_file):
    """
    Load the configuration from a YAML file.

    Args:
        config_file (str): The path to the YAML configuration file.

    Returns:
        tuple: A tuple containing the following elements:
            - encoding (str): The encoding used in the configuration file.
            - config_quanv (dict): The configuration for the 'quanv' section.
            - optimization_config (dict): The configuration for the 'optimization' section.
            - model_config (dict): The configuration for the 'model' section.
    """
    if ".yaml" not in config_file:
        config_file = config_file + ".yaml"
        
    with open(config_file, 'r') as file:

        config = yaml.safe_load(file)
        # if the encoding is not specified, default to None
        encoding = None
        config_quanv = None
        if 'encoding' in config:
            encoding = config['encoding']
            #I want to convert encoding from a string to an item in constants.CircuitEncoding
            encoding = getattr(constants.CircuitEncoding, encoding)

            config_quanv = config['quanv']
        optimization_config = None
        if 'optimization' in config:
            optimization_config = config['optimization']
        model_config = config['model']
    return encoding, config_quanv, optimization_config, model_config

