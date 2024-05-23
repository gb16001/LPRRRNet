

def load_yml_box():
    import yaml
    from box import Box

    # Define the path to your YAML file
    yaml_file_path = 'args.yaml'

    # Open the YAML file and load its contents
    with open(yaml_file_path, 'r') as file:
        args = yaml.safe_load(file)
    config = Box(args)
    lr_s=config.lr_schedule
    weight_d=config.weight_decay
    # Print the loaded configuration
    print(lr_s,weight_d)
    return

def load_yml_dynaconf():
    from dynaconf import Dynaconf

    # Load the configuration
    settings = Dynaconf(settings_files=['args.yaml'])

    # Access configuration values using dot notation
    database_host = settings.lr_schedule
    server_port = settings.weight_decay

    # Use the configuration values as needed
    print(f"Database Host: {database_host}")
    print(f"Server Port: {server_port}")
    return
if __name__=="__main__":
    load_yml_box()
    load_yml_dynaconf()