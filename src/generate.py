import argparse
from pathlib import Path

import numpy as np

from pl_modules.model import PixelCNN

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=Path, help="Path to the checkpoint")
parser.add_argument("--num_sample", type=int, help="Number of sample to generate")
args = parser.parse_args()

# load the lighninig model
model = PixelCNN.load_from_checkpoint(args.ckpt_path)

model.eval()
out = model(args.num_sample)

save_path = "results/"
size = out["sample"].shape[-1] ** 2
save_name = (
    "size-"
    + str(size)
    + "_sample-"
    + str(args.num_sample)
    + "_"
    + args.ckpt_path.parts[-3]
)

print("Saving...", save_name)
np.savez(save_path + save_name, **out)
