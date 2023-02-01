import torch
import torchvision
import argparse
import subprocess
from folders import pil_loader
from models import Net

from cog import BasePredictor, Path, Input

torch.manual_seed(0)
device = torch.device("cuda:0")


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "-p", "/root/.cache/torch/hub/checkpoints"])
        subprocess.run(["cp", "-r", "resnet50-19c8e357.pth", "/root/.cache/torch/hub/checkpoints/"])

        ckpt = 'pretrained_models/live_1_2021/sv/bestmodel_1_2021.zip'
        config = argparse.Namespace()
        config.network = 'resnet50'
        config.nheadt = 16
        config.num_encoder_layerst = 2
        config.dim_feedforwardt = 64

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        self.net = Net(config, device).to(device)
        self.net.load_state_dict(torch.load(ckpt))
        self.net.eval()

    def predict(
            self,
            input_image: Path = Input(description="Image to run on."),
    ) -> float:
        img = self.transforms(pil_loader(input_image)).to(device).unsqueeze(0)

        img = torch.as_tensor(img)
        pred, _ = self.net(img)

        return pred.item()

