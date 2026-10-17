"""Evaluate a saved TA-DOTCNet experiment checkpoint."""

import argparse
import json

# Initialize NumPy before PyTorch when inheriting an Intel MKL environment.
import numpy as np
import torch
from open_clip import tokenize

import datasets
from ta_dotcnet import model_from_checkpoint
from train_ta_dotcnet import (
    CLASSES, evaluate, evaluation_prompts, load_remoteclip, load_task_state,
    restore_prototypes, PROTOCOL,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default="dataset/train")
    parser.add_argument("--target-domain", default=None)
    parser.add_argument("--remoteclip-checkpoint", default="weights/RemoteCLIP-ViT-B-32.pt")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    if config.get("protocol") != PROTOCOL:
        parser.error(
            f"Checkpoint protocol {config.get('protocol')!r} does not match {PROTOCOL!r}. "
            "Evaluate this checkpoint with the code version that trained it."
        )
    target = args.target_domain or config.get("target_domain")
    if target is None:
        parser.error("--target-domain is required when checkpoint config has no target domain")
    batch_size = args.batch_size or config.get("batch_size", 16)
    clip_model, _, transform = load_remoteclip(args.remoteclip_checkpoint, device)
    model = model_from_checkpoint(clip_model, tokenize, checkpoint).to(device)
    model.ot_epsilon = config.get("epsilon", 0.1)
    model.ot_max_iter = config.get("sinkhorn_iterations", 200)
    load_task_state(model, checkpoint.get("model_state_dict", checkpoint))
    if "evaluation_start_buffers" in checkpoint:
        restore_prototypes(model, checkpoint["evaluation_start_buffers"])
    dataset, loader = datasets.load_testing(
        args.data_root, target, batch_size,
        {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"},
        transform=transform,
    )
    if tuple(dataset.class_names) != CLASSES:
        raise ValueError(f"Expected class folders {CLASSES}, got {dataset.class_names}")
    accuracy = evaluate(model, loader, evaluation_prompts(CLASSES), device)
    print(json.dumps(dict(target=target, data_root=args.data_root, samples=len(dataset),
                          accuracy=accuracy, batch_size=batch_size)), flush=True)


if __name__ == "__main__":
    main()
