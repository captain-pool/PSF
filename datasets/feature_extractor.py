import numpy as np
import PIL.Image
import torch
import tqdm
from torchvision import models
from sklearn import manifold
import scipy.spatial


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
        self.model = models.resnet50(weights=weights).to(self._device)
        self._shape = [3, 224, 224]
        self.transforms = weights.transforms()
        self._frames_data = dict()
        self._ncomponents =  32
        self._ldim = 2048

        self._reduce = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=self._ncomponents)
        self.model.fc = Identity()
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

    def features_img(self, img, latent_dim=2048):
      old_device = img.device
      num_imgs = len(img)
      batch_size = min(num_imgs, 64)
      computed_result = torch.zeros((num_imgs, self._ldim), dtype=torch.float32)
      with torch.no_grad():
        for i in range(0, num_imgs // batch_size, batch_size):
          computed_result = self.model(self.transforms(img[i:i + batch_size].to(self._device)))
      return computed_result.to(old_device)

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
