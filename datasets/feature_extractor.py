import functools

import numpy as np
import PIL.Image
import scipy.spatial
import torch
import tqdm
from torchvision import models
from torchvision.models import vision_transformer

import argparse

synsetid_to_cate = {
    "airplane": "02691156",
    "bag": "02773838",
    "basket": "02801938",
    "bathtub": "02808440",
    "bed": "02818832",
    "bench": "02828884",
    "bottle": "02876657",
    "bowl": "02880940",
    "bus": "02924116",
    "cabinet": "02933112",
    "can": "02747177",
    "camera": "02942699",
    "cap": "02954340",
    "car": "02958343",
    "chair": "03001627",
    "clock": "03046257",
    "dishwasher": "03207941",
    "monitor": "03211117",
    "table": "04379243",
    "telephone": "04401088",
    "tin_can": "02946921",
    "tower": "04460130",
    "train": "04468005",
    "keyboard": "03085013",
    "earphone": "03261776",
    "faucet": "03325088",
    "file": "03337140",
    "guitar": "03467517",
    "helmet": "03513137",
    "jar": "03593526",
    "knife": "03624134",
    "lamp": "03636649",
    "laptop": "03642806",
    "speaker": "03691459",
    "mailbox": "03710193",
    "microphone": "03759954",
    "microwave": "03761084",
    "motorcycle": "03790512",
    "mug": "03797390",
    "piano": "03928116",
    "pillow": "03938244",
    "pistol": "03948459",
    "pot": "03991062",
    "printer": "04004475",
    "remote_control": "04074963",
    "rifle": "04090263",
    "rocket": "04099429",
    "skateboard": "04225987",
    "sofa": "04256520",
    "stove": "04330267",
    "vessel": "04530566",
    "washer": "04554684",
    "cellphone": "02992529",
    "birdhouse": "02843684",
    "bookshelf": "02871439",
    # 'boat': '02858304', no boat in our dataset, merged into vessels
    # 'bicycle': '02834778', not in our taxonomy
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", required=True, help="Data Root with npy point clouds"
    )
    parser.add_argument("--r2n2dir", required=True, help="R2N2 Directory")
    parser.add_argument("--category", default="chair", type=str)
    return parser


def points_on_sphere(N):
    points = np.zeros((N, 3))

    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        y = 1 - (i / float(N - 1)) * 2
        radius = np.sqrt(1 - y**2)
        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = np.asarray([x, y, z])

    return points


class Identity(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class ImageFeatureExtractor:
    def __init__(self, precompute=None, device=None):

        if torch.cuda.is_available():
            self._device = device or f"cuda"
        else:
            self._device = "cpu"

        weights = vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vision_transformer.vit_b_16(weights=weights).to(self._device)

        self._shape = [3, 224, 224]
        self.transforms = weights.transforms()
        self._frames_data = dict()
        self._ncomponents = 768

        if precompute is not None:

            print("Precomputing Image Features")

            batch_size = len(precompute)
            precompute = np.asarray(precompute)
            length = len(precompute)
            precompute_idxs = list(range(length))

            idx_batches = [
                range(i, min(i + batch_size, length))
                for i in precompute_idxs[::batch_size]
            ]

            for i in tqdm.tqdm(idx_batches):
                _ = self.features(precompute[i])

            result = self.features(precompute[precompute_idxs])
            self.tree = scipy.spatial.KDTree(result)

    def features(self, img_paths=None, pil_imgs=None):

        assert img_paths is None or pil_imgs is None  # We gotta do XOR

        cached = bool(img_paths)

        nimgs = len(img_paths) if img_paths is not None else len(pil_imgs)

        result = torch.zeros(nimgs, self._ncomponents, dtype=torch.float32)

        computed_data = []
        computed_indices = []
        cached_indices = []
        cached_data = []

        for i in range(nimgs):
            if cached:
                if img_paths[i].name not in self._frames_data:
                    img = PIL.Image.open(img_paths[i]).convert("RGB")
                else:
                    cached_indices.append(i)
                    cached_data.append(self._frames_data[img_path.name])
                    continue
            else:
                img = pil_imgs[i]

            inputs = self.transforms(img)
            computed_indices.append(i)
            computed_data.append(inputs[None, ...])

        if len(computed_indices) > 0:
            computed_data = torch.concat(computed_data).to(self._device)
            with torch.no_grad():
                computed_result = self.model._process_input(computed_data)
                batch_class_token = self.model.class_token.expand(
                    computed_result.shape[0], -1, -1
                )
                computed_result = torch.cat([batch_class_token, computed_result], dim=1)
                computed_result = self.model.encoder(computed_result)
                computed_result = computed_result[:, 1:].mean(dim=1)
                computed_result = computed_result.squeeze().detach().cpu()

            result[computed_indices] = computed_result.to(result.device)

            if cached:
                for idx in computed_indices:
                    self._frames_data[img_paths[idx].name] = result[idx]

        if len(cached_indices) > 0:
            result[cached_indices] = torch.vstack(cached_data).to(result.device)

        return result


if __name__ == "__main__":

    import numpy as np
    import pathlib
    import tqdm

    parser = build_parser()

    args, _ = parser.parse_known_args()

    root = pathlib.Path(args.dataroot)
    rendered_dir = pathlib.Path(args.r2n2dir)

    extractor = ImageFeatureExtractor()

    synset_id = synsetid_to_cate[args.category]

    for f in tqdm.tqdm(list(root.glob(f"{synset_id}/*/*.npy"))):
        feat_dir = f.parent / "feats"

        feat_dir.mkdir(exist_ok=True)

        feat_path = feat_dir / f"{f.stem}_feat.npy"

        try:
            np.load(f, allow_pickle=True)
        except:
            continue

        imgs = []
        refs = []

        images = rendered_dir / synset_id / f.stem / "rendering"
        for i, image_path in enumerate(images.glob("*.png")):
            img = PIL.Image.open(str(image_path)).convert("RGB")
            imgs.append(img)
        if len(imgs) == 0:
          raise OSError("Files not found.")
        feat = extractor.features(pil_imgs=imgs).squeeze()
        np.save(str(feat_path), feat.squeeze())
