import functools

import numpy as np
import PIL.Image
import scipy.spatial
import torch
import tqdm
from sklearn import manifold
from torchvision import models

import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", required=True, help="Data Root with npy point clouds"
    )
    parser.add_argument("--nviews", type=int, default=5, help="Number of Views")
    return parser


def points_on_sphere(N):
    points = np.zeros((N, 3))

    phi = np.pi * (3. - np.sqrt(5.))
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
            self._device = device or f"cuda:{torch.cuda.device_count() - 1}"
        else:
            self._device = "cpu"

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        self.model.fc = Identity()
        self.model = self.model.to(self._device)

        self._shape = [3, 224, 224]
        self.transforms = weights.transforms()
        self._frames_data = dict()
        self._ncomponents = 32
        self._ldim = 2048

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
        computed_result = torch.zeros((num_imgs, self._ldim), dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, num_imgs // batch_size, batch_size):
                inputs = torch.from_numpy(img[i : i + batch_size]).to(self._device)
                computed_result = self.model(self.transforms(inputs))
        return computed_result

    def features(self, img_paths):
        result = torch.empty(len(img_paths), self._ncomponents, dtype=torch.float32)

        computed_data = []
        computed_indices = []
        cached_indices = []
        cached_data = []

        for i, img_path in enumerate(img_paths):
            if img_path.name not in self._frames_data:
                img = PIL.Image.open(img_path).convert("RGB")
                inputs = self.transforms(img)
                computed_indices.append(i)
                computed_data.append(inputs[None, ...])
            else:
                cached_indices.append(i)
                cached_data.append(self._frames_data[img_path.name])

        if len(computed_indices) > 0:
            computed_data = torch.concat(computed_data)
            with torch.no_grad():
                computed_result = self.model(computed_data)
            #                computed_result = torch.tensor(
            #                    self._reduce.fit_transform(computed_result.detach().cpu().numpy())
            #                ).to(torch.float32)
            result[computed_indices] = computed_result
            for idx in computed_indices:
                self._frames_data[img_paths[idx].name] = result[idx]

        if len(cached_indices) > 0:
            result[cached_indices] = torch.vstack(cached_data).to(result.device)

        return result


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

    extractor = ImageFeatureExtractor()

    for file in tqdm.tqdm(root.glob("*/*/*.npy")):
        img_dir = file.parent / "feats"
        if not img_dir.exists():
            img_dir.mkdir()

        feat_path = img_dir / f"{file.stem}_feat.npy"

        try:
          cloud = np.load(file, allow_pickle=True)
        except:
          continue

        img_tensor = torch.zeros((len(eyes), 224, 224, 3), dtype=torch.float32)
        for i in range(len(eyes)):
            eye = eyes[i]
            img_tensor[i] = torch.from_numpy(
                renderer.render_cloud(cloud, eye=eye)
            ).squeeze()

        feats = extractor.features_img(img_tensor).numpy()
        np.save(str(feat_path), feats)
