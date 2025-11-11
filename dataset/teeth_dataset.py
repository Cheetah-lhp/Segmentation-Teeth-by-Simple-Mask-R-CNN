import os
import json
import numpy as np
import skimage.io
import skimage.color
import skimage.draw


class TeethDataset:
    def __init__(self):
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]  # Background class
        self._image_ids = []

    def add_class(self, source, class_id, class_name):
        # Avoid duplicates
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name
        })

    def add_image(self, source, image_id, path, **kwargs):
        info = {
            "id": image_id,
            "source": source,
            "path": path
        }
        info.update(kwargs)
        self.image_info.append(info)

    def load_teeth(self, dataset_dir, subset, annotation_json):
        assert subset in ["train", "val"]
        subset_dir = os.path.join(dataset_dir, subset)

        with open(annotation_json) as f:
            annotations = json.load(f)

        # Register all unique classes
        class_titles = set()
        for item in annotations:
            for obj in item["Label"]["objects"]:
                class_titles.add(obj["title"])

        for title in class_titles:
            class_id = int(title)
            self.add_class("teeth", class_id, f"tooth_{title}")

        # Add image entries
        for item in annotations:
            filename = item["External ID"]
            image_path = os.path.join(subset_dir, filename)
            if not os.path.exists(image_path):
                continue

            objects = []
            for obj in item["Label"]["objects"]:
                objects.append({
                    "class_id": int(obj["title"]),
                    "polygons": obj["polygons"]
                })

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source="teeth",
                image_id=filename,
                path=image_path,
                width=width,
                height=height,
                objects=objects
            )

    def prepare(self):
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {
            f"{info['source']}.{info['id']}": i
            for i, info in enumerate(self.class_info)
        }

        self.image_from_source_map = {
            f"{info['source']}.{info['id']}": i
            for i, info in enumerate(self.image_info)
        }

        self.sources = list(set(info["source"] for info in self.class_info))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = [
                i for i, info in enumerate(self.class_info)
                if i == 0 or info["source"] == source
            ]

    @property
    def image_ids(self):
        return self._image_ids

    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "teeth":
            return np.empty([0, 0, 0]), np.empty([0], np.int32)

        objects = info["objects"]
        height, width = info["height"], info["width"]
        masks = []
        class_ids = []

        for obj in objects:
            for poly in obj["polygons"]:
                if len(poly) < 3:
                    continue
                poly = np.array(poly)
                rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0], shape=(height, width))
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[rr, cc] = 1
                masks.append(mask)
                class_ids.append(obj["class_id"])

        if masks:
            mask = np.stack(masks, axis=-1)
            return mask.astype(bool), np.array(class_ids, dtype=np.int32)
        else:
            return np.empty([0, 0, 0]), np.array([], dtype=np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]["path"]
