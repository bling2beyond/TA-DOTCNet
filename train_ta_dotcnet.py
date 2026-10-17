"""Train TA-DOTCNet for remote sensing cross-scene classification."""

import argparse
import json
import os
import random
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, tokenize

import datasets
from prompts import LLM_CANDIDATE_PROMPTS
from ta_dotcnet import PROTOCOL, TADOTCNet


CLASSES = ("farmland", "forest", "industrial", "parking", "residential", "river")
PROTOTYPE_BUFFERS = ("global_src_proto", "global_tgt_proto", "num_updates")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/train")
    parser.add_argument("--source-domain", default="WHU")
    parser.add_argument("--target-domain", default="UCM")
    parser.add_argument("--remoteclip-checkpoint", default="weights/RemoteCLIP-ViT-B-32.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lambda-ot", type=float, default=1.25)
    parser.add_argument("--gamma-target", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def load_remoteclip(checkpoint_path, device):
    model, train_transform, eval_transform = create_model_and_transforms(
        "ViT-B-32", pretrained=None
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, train_transform, eval_transform


@torch.no_grad()
def encode_text_anchors(clip_model, class_names, device):
    """Average the five normalized prompt embeddings for each category."""
    anchors = []
    for name in class_names:
        tokens = tokenize(LLM_CANDIDATE_PROMPTS[name]).to(device)
        features = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
        anchors.append(F.normalize(features.mean(dim=0), dim=-1))
    return torch.stack(anchors)


def evaluation_prompts(class_names):
    """The recorded evaluation uses the first prompt, without projection."""
    return [LLM_CANDIDATE_PROMPTS[name][0] for name in class_names]


def task_state_dict(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("clip_model.")
    }


def load_task_state(model, state):
    """Load the task state, allowing only the separately loaded backbone to be absent."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_task = [name for name in missing if not name.startswith("clip_model.")]
    if missing_task or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing_task}, unexpected={unexpected}")


def prototype_state(model):
    return {name: getattr(model, name).detach().clone() for name in PROTOTYPE_BUFFERS}


@torch.no_grad()
def restore_prototypes(model, state):
    for name in PROTOTYPE_BUFFERS:
        getattr(model, name).copy_(state[name])


def build_loaders(args, train_transform, eval_transform):
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        worker_init_fn=partial(seed_worker, seed=args.seed),
        generator=generator,
    )
    source, source_loader = datasets.load_training(
        args.data_root, args.source_domain, args.batch_size, options, transform=train_transform
    )
    target, target_loader = datasets.load_training(
        args.data_root, args.target_domain, args.batch_size, options, transform=train_transform
    )
    _, eval_loader = datasets.load_testing(
        args.data_root, args.target_domain, args.batch_size, options, transform=eval_transform
    )
    if tuple(source.class_names) != CLASSES or tuple(target.class_names) != CLASSES:
        raise ValueError(f"Both domains must contain the ordered class folders {CLASSES}")
    if not len(source_loader) or not len(target_loader):
        raise ValueError("Each training domain must contain at least one complete batch")
    return source_loader, target_loader, eval_loader


@torch.no_grad()
def calibrate_target_statistics(model, target_loader, text_anchors, device, seed):
    """Calibrate target-domain BN and save the fixed text semantic pool."""
    dropout_states = [(module, module.training) for module in model.projector.modules()
                      if isinstance(module, torch.nn.Dropout)]
    try:
        for module, _ in dropout_states:
            module.eval()
        for images, _ in target_loader:
            model.encode_projected_features(images.to(device)).cpu()
    finally:
        for module, training in dropout_states:
            module.train(training)
    model.eval()
    model.set_text_anchors(text_anchors)


def train_epoch(model, source_loader, target_loader, text_anchors, args, epoch):
    model.train()
    # The saved recipe starts a new SGD optimizer (and momentum history) each epoch.
    learning_rate = args.learning_rate / (1 + 10 * (epoch - 1) / args.epochs) ** 0.75
    optimizer = torch.optim.SGD(
        [
            {"params": model.projector.parameters()},
            {"params": model.classifier.parameters()},
        ],
        lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    target_iterator = iter(target_loader)
    loss_sum = 0.0
    for source_images, source_labels in source_loader:
        try:
            target_images, _ = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_images, _ = next(target_iterator)
        # Frozen text embeddings provide the same semantic pool throughout training.
        anchors = text_anchors
        optimizer.zero_grad()
        outputs = model(
            source_images.to(args.device), target_images.to(args.device),
            source_labels.to(args.device), text_src=anchors, text_tgt=anchors,
        )
        loss = (outputs["loss_src_ce"] + args.gamma_target * outputs["loss_tgt_ce"]
                + args.lambda_ot * outputs["loss_dec"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(source_loader), learning_rate


@torch.no_grad()
def evaluate(model, data_loader, prompts, device, persist_prototypes=False):
    """Replay the recorded source-branch evaluation with batchwise prototype updates.

    Labels are used only to count correct predictions. Standalone evaluation
    restores the starting buffers, so repeated calls do not accumulate updates.
    """
    model.eval()
    start = prototype_state(model)
    correct = total = 0
    try:
        for images, labels in data_loader:
            images = images.to(device)
            # The full forward computes unused losses; dummy labels keep ground
            # truth outside the prediction path.
            dummy_labels = torch.zeros(len(images), dtype=torch.long, device=device)
            outputs = model(images, images, dummy_labels, text_src=prompts, text_tgt=None)
            predictions = outputs["src_logits"].argmax(dim=1).cpu()
            correct += int((predictions == labels.cpu()).sum())
            total += len(labels)
    finally:
        if not persist_prototypes:
            restore_prototypes(model, start)
    return 100.0 * correct / total


def run(args):
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("epochs must be positive and batch-size must be at least two")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    output = Path(args.output_dir) / f"{args.source_domain}_to_{args.target_domain}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(f"Use a new output directory; metrics already exist in {output}")
    clip_model, train_transform, eval_transform = load_remoteclip(args.remoteclip_checkpoint, device)
    model = TADOTCNet(clip_model=clip_model, tokenizer=tokenize).to(device)
    model.ot_epsilon = args.epsilon
    model.ot_max_iter = args.sinkhorn_iterations
    source_loader, target_loader, eval_loader = build_loaders(args, train_transform, eval_transform)
    anchors = encode_text_anchors(clip_model, CLASSES, device)
    calibrate_target_statistics(model, target_loader, anchors, device, args.seed)
    config = dict(vars(args), protocol=PROTOCOL, class_names=list(CLASSES),
                  source_samples=len(source_loader.dataset), target_samples=len(target_loader.dataset))
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_accuracy, best_epoch = -1.0, 0
    checkpoint_path = output / "best_ta_dotcnet.pth"
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        loss, learning_rate = train_epoch(model, source_loader, target_loader, anchors, args, epoch)
        evaluation_start = prototype_state(model)
        accuracy = evaluate(model, eval_loader, evaluation_prompts(CLASSES), device, persist_prototypes=True)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(dict(
                model_state_dict=task_state_dict(model), config=config,
                evaluation_start_buffers={key: value.cpu() for key, value in evaluation_start.items()},
                epoch=epoch, accuracy=accuracy,
            ), checkpoint_path)
        row = dict(epoch=epoch, loss=loss, learning_rate=learning_rate, accuracy=accuracy,
                   best_accuracy=best_accuracy, seconds=time.monotonic() - started,
                   sinkhorn=dict(model.sinkhorn_stats))
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    result = dict(best_accuracy=best_accuracy, best_epoch=best_epoch, epochs=args.epochs,
                  final_accuracy=accuracy, checkpoint=str(checkpoint_path), protocol=PROTOCOL)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(build_parser().parse_args())
