import os
import json
from typing import Any, cast, override
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from tqdm import tqdm


type DatasetReturnT = tuple[torch.Tensor, dict[str, torch.Tensor]]
type DatasetT = Dataset[DatasetReturnT]


class BDD100KDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    image_dir: str
    label_dir: str
    transforms: any  # pyright: ignore[reportGeneralTypeIssues]
    images: list[str]
    labels: list[str]

    def __init__(self, image_dir: str, label_dir: str, transforms: Any = None):  # pyright: ignore[reportExplicitAny]
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms

        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

        assert len(self.images) == len(self.labels), (
            "Number of images and labels must be the same."
        )

    def __len__(self) -> int:
        return len(self.images)

    @override
    def __getitem__(self, idx: int) -> DatasetReturnT:
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        label_path = os.path.join(self.label_dir, self.labels[idx])
        with open(label_path, "r") as f:
            label = json.load(f)

        car_objects = [
            obj for obj in label["frames"][0]["objects"] if obj["category"] == "car"
        ]
        boxes = [
            [
                obj["box2d"]["x1"],
                obj["box2d"]["y1"],
                obj["box2d"]["x2"],
                obj["box2d"]["y2"],
            ]
            for obj in car_objects
        ]

        # Handle empty boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # 1 = car

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            image = self.transforms(image)

        return cast(torch.Tensor, image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int) -> FasterRCNN:
    # Load pre-trained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_one_epoch(
    model: FasterRCNN,
    dataloader: DataLoader[DatasetReturnT],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):
    _ = model.train()

    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass - model returns losses during training
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: FasterRCNN,
    dataloader: DataLoader[DatasetReturnT],
    device: torch.device,
    iou_threshold: float = 0.5,
):
    _ = model.eval()

    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images = [img.to(device) for img in images]

        # Get predictions
        predictions = model(images)

        for pred, target in zip(predictions, targets):
            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()

            gt_boxes = target["boxes"]

            # Filter predictions by score threshold
            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]

            # Calculate IoU between predictions and ground truth
            matched_gt = set()
            tp = 0
            fp = 0

            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_box in enumerate(gt_boxes):
                    iou = box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"\n--- Evaluation Metrics ---")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"-------------------------\n")

    return f1


def box_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def main():
    num_epochs = 5
    learning_rate = 0.005
    batch_size = 2

    max_train_samples = 200
    max_test_samples = 50

    # Simple transform - just convert to tensor
    transform = transforms.ToTensor()

    train_dataset_full = BDD100KDataset(
        "bdd100k/images/train",
        "bdd100k/labels/train",
        transforms=transform,
    )
    test_dataset_full = BDD100KDataset(
        "bdd100k/images/test",
        "bdd100k/labels/test",
        transforms=transform,
    )

    train_dataset = Subset(
        train_dataset_full, range(min(max_train_samples, len(train_dataset_full)))
    )
    test_dataset = Subset(
        test_dataset_full, range(min(max_test_samples, len(test_dataset_full)))
    )

    print(f"Training on {len(train_dataset)} images")
    print(f"Testing on {len(test_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # num_classes = 2 (background + car)
    model = get_model(num_classes=2).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Training Loss: {train_loss:.4f}")

        import gc

        _ = gc.collect()

        f1_score = evaluate(model, test_loader, device)

        _ = gc.collect()

        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), "best_faster_rcnn_model.pth")
            print("✅ Saved Best Model")

        lr_scheduler.step()

    print(f"\n🎉 Training Complete! Best F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
