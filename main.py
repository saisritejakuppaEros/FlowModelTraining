# read the config file
import yaml
import pprint
from data_ops.imagecaption_database import ImageCaptionModule
from model_ops.model_trainer import ModelTrainer



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

        # # get me a batch from the train dataloader
        # for batch in train_dataloader:
        #     imgs, captions = batch["raw_images"], batch["raw_texts"]
        #     print(imgs.shape)
        #     print(captions)
        #     break


        model_trainer = ModelTrainer()
        model_trainer.load_vae(self.config["model_arch"]["vae"])
        model_trainer.load_text_encoder(self.config["model_arch"]["text_encoder"])
        model_trainer.load_clip_encoder(self.config["model_arch"]["clip_encoder"])
        model_trainer.load_patchifier(self.config["model_arch"]["patchifier"])
        model_trainer.load_denoiser(self.config["model_arch"]["denoiser"])


        # load the sampler
        # model_trainer.load_timesampler(self.config["model_arch"]["time_sampler"])
        # model_trainer.load_timewarper(self.config["model_arch"]["time_warper"])





def read_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = read_config("config/train.yaml")
    operation = Operation(config)
    print(yaml.dump(operation.config, default_flow_style=False))
    operation.run()