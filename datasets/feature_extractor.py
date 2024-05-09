import functools

import numpy as np
import PIL.Image
import scipy.spatial
import torch
import tqdm
from sklearn import manifold
from torchvision import models

import argparse
import ray

if not ray.is_initialized():
    ray.init(num_gpus=torch.cuda.device_count())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", required=True, help="Data Root with npy point clouds"
    )
    parser.add_argument("--nviews", type=int, default=5, help="Number of Views")
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


@ray.remote(num_gpus=torch.cuda.is_available() * 0.05)
class ImageFeatureExtractor:
    def __init__(self, precompute=None, device=None):

        if torch.cuda.is_available():
            self._device = device or f"cuda"
        else:
            self._device = "cpu"

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        self.model.fc = Identity()
        self.model = self.model.to(self._device)

        self._shape = [3, 224, 224]
        self.transforms = weights.transforms()
        self._frames_data = dict()
        self._ncomponents = 2048  # 32

        self._reduce = manifold.LocallyLinearEmbedding(
            n_neighbors=10, n_components=self._ncomponents
        )

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

    def mark(self, dp=False, ddp=False):

        assert dp or ddp, "DataParallel or DistributedDataParallel has to be set"

        dp_wrapper_fn = torch.nn.parallel.DataParallel
        ddp_wrapper_fn = functools.partial(
            torch.nn.parallel.DistributedDataParallel,
            device_ids=[self._device],
            output_device=[self._device],
        )

        if dp:
            return dp_wrapper_fn(self.model)

        if ddp:
            return ddp_wrapper_fn(self.model)

        return self.model

    def features_img(self, img, latent_dim=2048):
        num_imgs = len(img)
        batch_size = min(num_imgs, 64)
        computed_result = torch.zeros(
            (num_imgs, self._ncomponents), dtype=torch.float32
        )
        with torch.no_grad():
            for i in range(0, num_imgs // batch_size, batch_size):
                inputs = torch.from_numpy(img[i : i + batch_size]).to(self._device)
                computed_result = self.model(self.transforms(inputs))
        return computed_result

    def features(self, img_paths=None, pil_imgs=None):

        assert img_paths is None or pil_imgs is None  # We gotta do XOR

        cached = bool(img_paths)

        nimgs = len(img_paths) if img_paths is not None else len(pil_imgs)

        result = torch.empty(nimgs, self._ncomponents, dtype=torch.float32)

        computed_data = []
        computed_indices = []
        cached_indices = []
        cached_data = []

        for i in range(nimgs):
            if cached and img_paths[i].name not in self._frames_data:
                img = PIL.Image.open(img_paths[i]).convert("RGB")
            elif pil_imgs is not None:
                inputs = self.transforms(pil_imgs[i])
                computed_indices.append(i)
                computed_data.append(inputs[None, ...])
            else:
                cached_indices.append(i)
                cached_data.append(self._frames_data[img_path.name])

        if len(computed_indices) > 0:
            computed_data = torch.concat(computed_data).to(self._device)
            with torch.no_grad():
                computed_result = self.model(computed_data)
            #                computed_result = torch.tensor(
            #                    self._reduce.fit_transform(computed_result.detach().cpu().numpy())
            #                ).to(torch.float32)
            result[computed_indices] = computed_result.to(result.device)

            if cached:
                for idx in computed_indices:
                    self._frames_data[img_paths[idx].name] = result[idx]

        if len(cached_indices) > 0:
            result[cached_indices] = torch.vstack(cached_data).to(result.device)

        return result


def ray_get_with_progress(refs):
    pbar = tqdm.tqdm(total=len(refs))
    results = []
    while refs:
        done, refs = ray.wait(refs)
        results.append(ray.get(done[0]))
        pbar.update(1)
    return results


if __name__ == "__main__":

    from utils import render
    import numpy as np
    import pathlib
    import tqdm

    parser = build_parser()

    args, _ = parser.parse_known_args()

    renderer = render.Renderer(center=[0, 0, 0], world_up=[0, 0, 1], res=(224, 224))

    eyes = 1.5 * points_on_sphere(args.nviews)

    root = pathlib.Path(args.dataroot)

    dump_interval = 100

    refs = []
    dump_names = []

    for file in tqdm.tqdm(list(root.glob("*/*/*.npy"))):
        img_dir = file.parent / "feats"
        if not img_dir.exists():
            img_dir.mkdir()

        feat_path = img_dir / f"{file.stem}_feat.npy"

        try:
            cloud = np.load(file, allow_pickle=True)
        except:
            continue

        imgs = []
        for i in range(len(eyes)):
            pixels = renderer.render_cloud(cloud, eye=eyes[i]).squeeze()
            imgs.append(PIL.Image.fromarray(pixels.astype(np.uint8)))

        extractor = ImageFeatureExtractor.remote()
        refs.append(extractor.features.remote(pil_imgs=imgs))
        dump_names.append(str(feat_path))

        if len(refs) >= dump_interval:
            print("Waiting ...")
            feats = ray_get_with_progress(refs)
            print("Dumping ...")
            for i in range(len(feats)):
                np.save(dump_names[i], feats[i].squeeze())

            dump_names = []
            refs = []
