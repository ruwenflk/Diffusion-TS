import os
import torch
import numpy as np

from engine.solver import Trainer
from Utils.metric_utils import visualization
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one


if __name__ == "__main__":

    class Args_Example:
        def __init__(self) -> None:
            self.config_path = "./Config/metropt3_100.yaml"
            self.save_dir = "./metropt3_100"
            self.gpu = 0
            os.makedirs(self.save_dir, exist_ok=True)

    args = Args_Example()
    configs = load_yaml_config(args.config_path)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dl_info = build_dataloader(configs, args)
    model = instantiate_from_config(configs["model"]).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)
    trainer.train()

    seq_length, feature_dim = dl_info["dataset"].window, dl_info["dataset"].var_num
    fake_data = trainer.sample(
        num=10000, size_every=128, shape=[seq_length, feature_dim]
    )
    np.save(os.path.join(args.save_dir, f"ddpm_fake_sines.npy"), fake_data)
