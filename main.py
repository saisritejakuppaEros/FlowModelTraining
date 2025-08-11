# read the config file
import yaml
import pprint
class Operation:
    def __init__(self, config):
        self.config = config
        
    def run(self):
        if self.config["operation"]["name"] == "train":
            self.train()

    def train(self):
        pass

def read_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = read_config("config/train.yaml")
    operation = Operation(config)
    print(yaml.dump(operation.config, default_flow_style=False))
    operation.run()