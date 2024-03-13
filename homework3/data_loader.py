# coding=utf-8

from collections import defaultdict

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class DatasetPreparation(Dataset):
    def __init__(self, args, df, iaa_transform=None, transform=None):
        self.args = args
        self.dir_path = args.data_dir / args.img_dir
        self.annotations = df
        self.labels_list = args.concepts
        self.iaa_transform = iaa_transform
        self.transform = transform
        self.mean = args.mean
        self.std = args.std
        self.image_dict = self._generate_image_dict()

    def _generate_image_dict(self):
        image_dict = defaultdict(lambda: {"boxes": [], "labels": []})

        for idx, row in self.annotations.iterrows():
            study_id = row["patient_id"]
            image_id = row["image_id"]
            boxes = row[["resized_xmin", "resized_ymin", "resized_xmax", "resized_ymax"]].values.tolist()
            labels = [label.strip() for label in row["finding_categories"].strip("[]").split(",")]
            for label in labels:
                label = label.strip("''")

                if label in self.labels_list:
                    index = self.labels_list.index(label)
                    image_dict[(study_id, image_id)]["boxes"].append(boxes + [index])
                    image_dict[(study_id, image_id)]["labels"].append(index)

        return image_dict

    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, idx):
        return self.get_items(idx)

    def get_items(self, idx):
        study_id, image_id = list(self.image_dict.keys())[idx]
        boxes = self.image_dict[(study_id, image_id)]["boxes"]
        labels = self.image_dict[(study_id, image_id)]["labels"]
        path = f"{self.dir_path}/{study_id}/{image_id}"

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image).convert('RGB')
        image = np.array(image)

        if self.iaa_transform:
            bb_box = []
            for bb in boxes:
                bb_box.append(BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]))

            bbs_on_image = BoundingBoxesOnImage(bb_box, shape=image.shape)
            image, boxes = self.iaa_transform(image=image, bounding_boxes=[bbs_on_image])

        if self.transform:
            image = self.transform(image)

        image = image.to(torch.float32)
        image -= image.min()
        image /= image.max()
        image = torch.tensor((image - self.mean) / self.std, dtype=torch.float32)

        bb_final = []
        for idx, bb in enumerate(boxes[0]):
            bb_final.append([bb.x1, bb.y1, bb.x2, bb.y2, labels[idx]])

        target = {"boxes": torch.tensor(bb_final), "labels": labels, }

        return {
            "image": image,
            "target": target,
            "study_id": study_id,
            "image_id": image_id,
            "img_path": path
        }


def dataset_loader(data):
    image = [s["image"] for s in data]
    res_bbox_tensor = [s["target"]["boxes"] for s in data]
    image_path = [s['img_path'] for s in data]

    max_num_annots = max(annot.shape[0] for annot in res_bbox_tensor)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(res_bbox_tensor), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(res_bbox_tensor):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(res_bbox_tensor), 1, 5)) * -1

    return {
        "image": torch.stack(image),
        "res_bbox_tensor": annot_padded,
        "image_path": image_path
    }


def get_transforms(args):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    train_affine_trans = iaa.Sequential([
        iaa.Resize({'height': args.resize, 'width': args.resize}),
        iaa.Fliplr(0.5),  # HorizontalFlip
        iaa.Flipud(0.5),  # VerticalFlip
        iaa.Affine(
            rotate=(-20, 20),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.8, 1.2),
            shear=(-20, 20)
        ),
        iaa.ElasticTransformation(alpha=args.alpha, sigma=args.sigma)
    ])

    test_affine_trans = iaa.Sequential([
        iaa.Resize({'height': args.resize, 'width': args.resize}),
        iaa.CropToFixedSize(width=args.resize, height=args.resize)  # Adjust width and height as needed
    ])

    return transform, train_affine_trans, test_affine_trans


def get_dataloader(args, train=True):
    transform, train_affine_trans, test_affine_trans = get_transforms(args)
    valid_dataset = DatasetPreparation(
        args=args,
        df=args.valid_folds,
        iaa_transform=test_affine_trans,
        transform=transform,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_loader,
    )

    if train:
        train_dataset = DatasetPreparation(
            args=args,
            df=args.train_folds,
            iaa_transform=train_affine_trans,
            transform=transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=dataset_loader
        )
        return train_loader, valid_loader, valid_dataset
    else:
        return valid_loader, valid_dataset


def get_dataset(args, train=True):
    return get_dataloader(args, train=train)
