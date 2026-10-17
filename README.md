# TA-DOTCNet

Code accompanying **Text-Anchored Deep Optimal Transport Clustering for Remote Sensing Cross-Scene Classification**.

**Authors:** Yue Zhao, Haoyang Fang, Maoguo Gong, Zhao Wang, Xin Mei, Hua Yu, Lianzhong Zhang, and Yew-Soon Ong.

**Publication:** IEEE Transactions on Geoscience and Remote Sensing, Early Access, 2026.

**Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11721406) | **DOI:** [10.1109/TGRS.2026.3740705](https://doi.org/10.1109/TGRS.2026.3740705)

TA-DOTCNet combines text-guided cross-scene adaptation with deep optimal transport clustering. The implementation uses a frozen RemoteCLIP ViT-B/32 backbone, a shared residual adapter, prototype-driven text fusion, and a parameter-free DOTC head.

## Repository structure

```text
TA-DOTCNet/
  train_ta_dotcnet.py       # Training entry point
  evaluate_ta_dotcnet.py    # Checkpoint evaluation
  ta_dotcnet.py             # Model, fusion classifier, and DOTC head
  cross_scene_adapter.py    # Shared residual adapter
  optimal_transport.py     # Log-domain Sinkhorn solver
  datasets.py              # Image-folder dataset and data loaders
  prompts.py               # Scene descriptions
  requirements.txt
  CITATION.cff
  LICENSE
  README.md
```

## Installation

Use Python 3.10 and install the dependencies:

```bash
conda create -n ta-dotcnet python=3.10 -y
conda activate ta-dotcnet
python -m pip install -r requirements.txt
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch build before the requirements. For example:

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Other supported CUDA builds are listed in the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v231). Use `--device cpu` for CPU execution.

## Data preparation

Prepare AID, UCM, WHU, and RSSCN7 using the same six class folders: `farmland`, `forest`, `industrial`, `parking`, `residential`, and `river`.

```text
dataset/
  train/
    AID/
      farmland/
      forest/
      industrial/
      parking/
      residential/
      river/
    UCM/
    WHU/
    RSSCN7/
  test/
    AID/
    UCM/
    WHU/
    RSSCN7/
```

Every domain folder follows the AID layout above. Place images directly inside the class folders. Supported formats are JPG, JPEG, PNG, TIF, and TIFF. Prepare the class mappings and train/test image lists before running the code. Datasets are supplied separately.

## Pretrained backbone

Download `RemoteCLIP-ViT-B-32.pt` from the [RemoteCLIP model repository](https://huggingface.co/chendelong/RemoteCLIP/tree/main) and place it at:

```text
weights/RemoteCLIP-ViT-B-32.pt
```

The scene descriptions are included in `prompts.py`; no LLM API setup is required.

## Training

Run commands from the repository root. For WHU to UCM:

```bash
python train_ta_dotcnet.py --data-root dataset/train --source-domain WHU --target-domain UCM --output-dir outputs/seed1
```

Default settings:

| Setting | Value |
| --- | --- |
| Backbone | RemoteCLIP ViT-B/32, frozen |
| Epochs | 200 |
| Batch size | 16 per domain |
| Optimizer | SGD |
| Initial learning rate | 0.02 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| OT loss weight | 1.25 |
| Target pseudo-label CE weight | 0.01 |
| OT entropy coefficient | 0.1 |
| Seed | 1 |

Use a fresh output directory for each run. The task folder `outputs/seed1/WHU_to_UCM/` contains:

- `config.json`: configuration and sample counts.
- `metrics.jsonl`: epoch losses, accuracy, learning rate, timing, and OT solver statistics.
- `result.json`: run summary.
- `best_ta_dotcnet.pth`: the checkpoint selected by adaptation-pool accuracy.

Training duration can be overridden with `--epochs`. For example, `--epochs 50` performs a shorter validation run.

To train all twelve directed source-target tasks in Bash:

```bash
for source in UCM WHU AID RSSCN7; do
  for target in UCM WHU AID RSSCN7; do
    if [ "$source" != "$target" ]; then
      python train_ta_dotcnet.py --source-domain "$source" --target-domain "$target" --output-dir outputs/seed1 || exit 1
    fi
  done
done
```

## Evaluation

Reload the checkpoint and evaluate the target test split:

```bash
python evaluate_ta_dotcnet.py outputs/seed1/WHU_to_UCM/best_ta_dotcnet.pth --data-root dataset/test
```

The evaluator prints accuracy and sample count. The target domain and batch size default to the saved configuration. Use `--data-root dataset/train` to replay adaptation-pool evaluation.

Task checkpoints store the adapter, classifier, prototypes, fixed text anchors, and evaluation buffers. The RemoteCLIP backbone is loaded separately. Use the same pretrained backbone, class folders, evaluation order, and batch size as the original run.

For all available options:

```bash
python train_ta_dotcnet.py --help
python evaluate_ta_dotcnet.py --help
```

## Citation

If you use this code, please cite the accompanying paper. Citation metadata is also provided in `CITATION.cff`.

```bibtex
@article{zhao2026tadotcnet,
  title = {Text-Anchored Deep Optimal Transport Clustering for Remote Sensing Cross-Scene Classification},
  author = {Zhao, Yue and Fang, Haoyang and Gong, Maoguo and Wang, Zhao and Mei, Xin and Yu, Hua and Zhang, Lianzhong and Ong, Yew-Soon},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year = {2026},
  doi = {10.1109/TGRS.2026.3740705},
  url = {https://ieeexplore.ieee.org/document/11721406},
  note = {Early Access}
}
```

## Acknowledgements

This implementation uses [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip).

## License

The source code is released under the [MIT License](LICENSE). Datasets and pretrained models retain their respective licenses.
