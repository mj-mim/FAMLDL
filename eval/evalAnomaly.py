
# evalAnomaly.py — pixel-wise anomaly scoring with ERFNet
# Methods: MSP, MaxLogit, MaxEntropy
import os, sys, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erfnet import ERFNet

NUM_CLASSES   = 20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(path, device):
    """Load ERFNet weights. encoder.output_conv is training-only; skip it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Weights not found: {path}')
    model = ERFNet(NUM_CLASSES)
    state = torch.load(path, map_location=device)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected_missing = {'encoder.output_conv.weight', 'encoder.output_conv.bias'}
    real_missing = set(missing) - expected_missing
    if real_missing:
        raise RuntimeError(f'Unexpected missing keys: {real_missing}')
    if unexpected:
        print(f'[WARNING] Unexpected keys in checkpoint: {unexpected}')
    return model.to(device).eval()


def preprocess(path):
    """Load RGB image → normalised (1,3,H,W) tensor."""
    img = Image.open(path).convert('RGB')
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return t(img).unsqueeze(0)


@torch.no_grad()
def get_logits(model, tensor, device):
    """Forward pass → raw logits (C,H,W)."""
    return model(tensor.to(device)).squeeze(0)


# ── Scoring methods (higher value = more anomalous) ────────────────────────

def score_msp(logits):
    """
    Maximum Softmax Probability (MSP) — Hendrycks & Gimpel, 2017.
    score = 1 - max_c softmax(logits)_c
    Low max-probability pixels are uncertain → likely anomalous.
    """
    return (1.0 - F.softmax(logits, dim=0).max(dim=0).values).cpu().numpy()


def score_maxlogit(logits):
    """
    MaxLogit — Hendrycks et al., 2022.
    score = -max_c logits_c
    Low max-logit pixels have no strong class signal → likely anomalous.
    """
    return (-logits.max(dim=0).values).cpu().numpy()


def score_maxentropy(logits):
    """
    Softmax Entropy — Chan et al. (Meta-OOD), 2021.
    score = -sum_c p_c * log(p_c)
    High entropy → uniform distribution → network is confused → anomaly.
    Reference implementation:
      https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/master/methods/baselines.py#L85
    """
    p = F.softmax(logits, dim=0)
    return (-(p * torch.log(p.clamp(min=1e-10))).sum(dim=0)).cpu().numpy()


METHODS = {
    'msp':        score_msp,
    'maxlogit':   score_maxlogit,
    'maxentropy': score_maxentropy,
}


def find_images(pattern):
    """Glob for images; try .png then .jpg if nothing found."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        alt = pattern.replace('*.png', '*.jpg')
        paths = sorted(glob.glob(alt))
    if not paths:
        alt = pattern.replace('*.png', '*.*')
        paths = sorted(glob.glob(alt))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',   required=True, help='Glob pattern for images')
    ap.add_argument('--method',  default='msp', choices=METHODS)
    ap.add_argument('--weights', default='../trained_models/erfnet_pretrained.pth')
    ap.add_argument('--output',  default='./scores')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device={device}  Method={args.method}')

    model = load_model(args.weights, device)
    paths = find_images(args.input)
    if not paths:
        print(f'[ERROR] No images found: {args.input}'); return
    print(f'[INFO] {len(paths)} image(s) found.')

    os.makedirs(args.output, exist_ok=True)
    fn = METHODS[args.method]

    for i, p in enumerate(paths):
        stem  = os.path.splitext(os.path.basename(p))[0]
        score = fn(get_logits(model, preprocess(p), device))
        np.save(os.path.join(args.output, f'{stem}_{args.method}.npy'), score)
        if (i + 1) % 10 == 0 or (i + 1) == len(paths):
            print(f'  [{i+1}/{len(paths)}]')

    print(f'[DONE] Saved to {args.output}')


if __name__ == '__main__':
    main()
