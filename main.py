# read the config file
import yaml
import pprint
from data_ops.imagecaption_database import ImageCaptionModule




class Operation:
    def __init__(self, config):
        self.config = config
        
    def run(self):
        if self.config["operation"]["name"] == "train":
            self.train()

    def train(self):
        # pass
        
        # load the dataset
        train_dataloader, val_dataloader, test_dataloader = ImageCaptionModule(self.config["dataset"]).get_dataloader()
        
        
        # print the no of samples in the train, val and test dataloader
        print(f"No of samples in train dataloader: {len(train_dataloader)}")
        print(f"No of samples in val dataloader: {len(val_dataloader)}")
        print(f"No of samples in test dataloader: {len(test_dataloader)}")
        

        # # get me a batch from the train dataloader
        # for batch in train_dataloader:
        #     imgs, captions = batch["raw_images"], batch["raw_texts"]
        #     print(imgs.shape)
        #     print(captions)
        #     break


def read_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = read_config("config/train.yaml")
    operation = Operation(config)
    print(yaml.dump(operation.config, default_flow_style=False))
    operation.run()