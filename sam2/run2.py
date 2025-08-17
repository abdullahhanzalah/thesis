import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import faiss
from sam2.build_sam import build_sam2_video_predictor
import h5py
from dataset_sam2 import create_dataset_paths, create_validation_paths
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Color scheme for medical segmentation (common classes)
MEDICAL_COLORS = [
    [0, 0, 0],  # Background (black)
    [255, 0, 0],  # Class 1: e.g., Left Ventricle (red)
    [0, 255, 0],  # Class 2: e.g., Myocardium (green)
    [0, 0, 255],  # Class 3: e.g., Right Ventricle (blue)
]


def load_medical_imgs(path_list):
    imgs, labels = [], []
    for path in path_list:
        h5f = h5py.File(path, "r")
        img = h5f["image"][:]
        label = h5f["label"][:]
        imgs.append(img)
        labels.append(label)
    return imgs, labels


def load_dino_model():
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )
    model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    return processor, model


def preprocess_imgs_for_sam2(images):

    def process(image):
        """Preprocess image specifically for SAM2 - returns numpy array"""
        # Handle different input shapes
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)  # Remove batch dimension
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(-1)  # Remove channel dimension if single channel

        # Ensure 2D
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1 if image.shape[-1] <= 3 else 0)

        # Normalize to [0, 1] for SAM2
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image_normalized = (image - image_min) / (image_max - image_min)
        else:
            image_normalized = np.zeros_like(image, dtype=np.float32)

        image_normalized = image_normalized.astype(np.float32)
        return image_normalized

    proc_imgs = []
    if not isinstance(images, list):
        return process(images)
    else:
        for image in images:
            image_normalized = process(image)
            proc_imgs.append(image_normalized)

        return proc_imgs


def extract_class_masks_from_label(label, n_classes):

    class_masks = []
    for class_id in range(1, n_classes + 1):  # Start from 1, skip background (0)
        class_mask = label == class_id
        class_masks.append(class_mask)
        print(f"Class {class_id} mask: {class_mask.sum()} pixels")

    return class_masks


# def create_multiclass_overlay(image, class_predictions, class_colors=None, alpha=0.6):
#     """
#     Create overlay with different colors for each class

#     Args:
#         image: original image (H, W) or (H, W, 3)
#         class_predictions: list of boolean masks for each class (including background)
#         class_colors: list of RGB colors for each class
#         alpha: transparency of overlay

#     Returns:
#         overlayed_image: image with colored class overlays
#         legend_info: list of (class_name, color) for legend
#     """
#     # Default colors for classes
#     if class_colors is None:
#         class_colors = [
#             [0, 0, 0],        # Background (black/transparent)
#             [255, 0, 0],      # Class 1 (red)
#             [0, 255, 0],      # Class 2 (green)
#             [0, 0, 255],      # Class 3 (blue)
#             [255, 255, 0],    # Class 4 (yellow)
#             [255, 0, 255],    # Class 5 (magenta)
#             [0, 255, 255],    # Class 6 (cyan)
#             [255, 128, 0],    # Class 7 (orange)
#             [128, 0, 255],    # Class 8 (purple)
#         ]

#     labels = ["Background", "LV", "Myocardium", "RV"]
#     # Ensure image is in RGB format
#     if len(image.shape) == 2:
#         image = np.stack([image] * 3, axis=-1)

#     # Normalize image to [0, 255] if needed
#     if image.max() <= 1.0:
#         image = (image * 255).astype(np.uint8)
#     else:
#         image = image.astype(np.uint8)

#     overlayed = image.copy().astype(np.float32)
#     legend_info = []

#     # Apply each class overlay (skip background class 0)
#     for i, class_mask in enumerate(class_predictions):
#         print(f"Processing class {i} with mask shape: {class_mask.shape}")
#         if i == 0:  # Skip background
#             continue

#         if class_mask.sum() > 0:  # Only overlay if class has pixels
#             color = class_colors[i % len(class_colors)]
#             class_overlay = np.zeros_like(overlayed)

#             # Create colored overlay for this class
#             mask_3d = class_mask[..., np.newaxis]
#             class_overlay[class_mask] = color

#             # Blend with existing overlay
#             overlayed = np.where(mask_3d,
#                                (1 - alpha) * overlayed + alpha * class_overlay,
#                                overlayed)

#             legend_info.append((f"{labels[i]}", color))

#     return overlayed.astype(np.uint8), legend_info


def create_multiclass_overlay(image, class_predictions, class_colors=None, alpha=0.6):
    """
    Create overlay with different colors for each class.

    Args:
        image: original image (H, W) or (H, W, 3)
        class_predictions:
            - list of boolean masks (model output), OR
            - label map (H, W) with integer IDs
        class_colors: list of RGB colors for each class
        alpha: transparency of overlay

    Returns:
        overlayed_image: image with colored class overlays
        legend_info: list of (class_name, color) for legend
    """
    if class_colors is None:
        class_colors = [
            [0, 0, 0],  # Background
            [255, 0, 0],  # Class 1
            [0, 255, 0],  # Class 2
            [0, 0, 255],  # Class 3
            [255, 255, 0],  # etc...
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
        ]

    labels = ["Background", "LV", "Myocardium", "RV"]

    # Ensure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    overlayed = image.copy().astype(np.float32)
    legend_info = []

    # Case 1: Already a list of boolean masks
    if isinstance(class_predictions, list):
        for i, class_mask in enumerate(class_predictions):
            if i == 0:  # skip background
                continue
            if class_mask.sum() > 0:
                color = np.array(class_colors[i % len(class_colors)], dtype=np.float32)
                mask_3d = class_mask[..., np.newaxis]
                overlayed = np.where(
                    mask_3d, (1 - alpha) * overlayed + alpha * color, overlayed
                )
                if i < len(labels):
                    legend_info.append((labels[i], color))
                else:
                    legend_info.append((f"Class {i}", color))

    # Case 2: Label map (H, W) of integers
    else:
        unique_classes = np.unique(class_predictions)
        for class_id in unique_classes:
            if class_id == 0:
                continue  # skip background
            class_mask = class_predictions == class_id
            if class_mask.sum() > 0:
                color = np.array(
                    class_colors[class_id % len(class_colors)], dtype=np.float32
                )
                mask_3d = class_mask[..., np.newaxis]
                overlayed = np.where(
                    mask_3d, (1 - alpha) * overlayed + alpha * color, overlayed
                )
                if class_id < len(labels):
                    legend_info.append((labels[class_id], color))
                else:
                    legend_info.append((f"Class {class_id}", color))

    return overlayed.astype(np.uint8), legend_info


def visualize_multiclass_results(
    image_test_sam,
    class_predictions,
    lbl_refs,
    image_refs_sam,
    lbl_query,
    class_colors=None,
    save_prefix="multiclass_sam2",
):
    """
    Comprehensive visualization for multi-class segmentation results
    """
    n_classes = (
        len(class_predictions) - 1 if class_predictions else 0
    )  # Exclude background

    if class_predictions is None or len(class_predictions) == 0:
        print("No predictions to visualize")
        return None, None

    # Create combined prediction label
    class_preds_array = np.stack(class_predictions, axis=0)
    pred_label = np.argmax(class_preds_array, axis=0)

    # Create multiclass overlay
    overlayed_image, legend_info = create_multiclass_overlay(
        image_test_sam, class_predictions, MEDICAL_COLORS
    )

    ref_image_overlays = []
    for img_ref, lbl_ref in zip(image_refs_sam, lbl_refs):
        overlayed_image_ref, _ = create_multiclass_overlay(
            img_ref, lbl_ref, MEDICAL_COLORS
        )
        ref_image_overlays.append(overlayed_image_ref)

    # Create visualization
    if lbl_refs is not None:
        fig, axes = plt.subplots(2, 6, figsize=(15, 10))

        # Top row: Original, Ground Truth, Prediction
        axes[0, 0].imshow(image_test_sam, cmap="gray")
        axes[0, 0].set_title("Test Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(pred_label, cmap="tab10", vmin=0, vmax=n_classes)
        axes[0, 1].set_title("SAM2 Prediction")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(
            lbl_query, cmap="tab10", vmin=0, vmax=len(np.unique(lbl_query)) - 1
        )
        axes[0, 2].set_title("Ground Truth")
        axes[0, 2].axis("off")

        if n_classes >= 1:
            axes[0, 3].imshow(class_predictions[1], cmap="gray")
            axes[0, 3].set_title("Class 1 Prediction")
            axes[0, 3].axis("off")

            axes[0, 4].axis("off")
            axes[0, 5].axis("off")

        if n_classes >= 2:
            axes[0, 4].imshow(class_predictions[2], cmap="gray")
            axes[0, 4].set_title("Class 2 Prediction")
            axes[0, 4].axis("off")
            axes[0, 5].axis("off")

        if n_classes >= 3:
            axes[0, 5].imshow(class_predictions[3], cmap="gray")
            axes[0, 5].set_title("Class 3 Prediction")
            axes[0, 5].axis("off")

        for i, ref_overlay in enumerate(ref_image_overlays):
            axes[1, i].imshow(ref_overlay)
            axes[1, i].set_title(f"Ref Image {i+1}")
            axes[1, i].axis("off")

        axes[1, 5].axis("off")  # Empty space if not enough reference images
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_test_sam, cmap="gray")
        axes[0].set_title("Test Image")
        axes[0].axis("off")

        axes[1].imshow(pred_label, cmap="tab10", vmin=0, vmax=n_classes)
        axes[1].set_title("SAM2 Prediction")
        axes[1].axis("off")

        axes[2].imshow(overlayed_image)
        axes[2].set_title("Multi-class Overlay")
        axes[2].axis("off")

    # Add legend
    legend_elements = []
    for class_name, color in legend_info:
        legend_elements.append(
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=np.array(color) / 255.0, label=class_name
            )
        )

    if legend_elements:
        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98)
        )

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Save individual results
    save_multiclass_results(
        image_test_sam, pred_label, overlayed_image, class_predictions, save_prefix
    )

    return pred_label, overlayed_image


def save_multiclass_results(image, pred_label, overlay, class_predictions, prefix):

    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.imsave(f"{folder}/{prefix}_image.png", image, cmap="gray")
    plt.imsave(f"{folder}/{prefix}_prediction.png", pred_label, cmap="tab10")
    plt.imsave(f"{folder}/{prefix}_overlay.png", overlay)

    for i, class_mask in enumerate(class_predictions):
        if i == 0:  # Background
            plt.imsave(f"{folder}/{prefix}_class_background.png", class_mask, cmap="gray")
        else:
            plt.imsave(f"{folder}/{prefix}_class_{i}.png", class_mask, cmap="gray")

    print(f"All results saved with prefix: {prefix}")


def calculate_class_metrics(pred_label, true_label, n_classes):
    """Calculate metrics for each class"""
    metrics = {}

    for class_id in range(n_classes + 1):  # Include background
        true_mask = true_label == class_id
        pred_mask = pred_label == class_id

        # Calculate IoU
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        iou = intersection / union if union > 0 else 0

        # Calculate Dice coefficient
        dice = (
            2 * intersection / (true_mask.sum() + pred_mask.sum())
            if (true_mask.sum() + pred_mask.sum()) > 0
            else 0
        )

        class_name = "Background" if class_id == 0 else f"Class {class_id}"
        metrics[class_name] = {"IoU": iou, "Dice": dice}

        print(f"{class_name}: IoU={iou:.3f}, Dice={dice:.3f}")

    return metrics


def main_multiclass(predictor, img_query, lbl_query, match_imgs_lbls):

    print(f"LABELS IN QUERY: {np.unique(lbl_query)}")
    image_query_sam = preprocess_imgs_for_sam2(img_query)

    image_refs_sam = []
    lbl_refs = []
    for img_ref, lbl_ref in match_imgs_lbls:

        unique_labels = np.unique(lbl_ref)
        n_classes = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        print(f"Found {n_classes} classes: {unique_labels}")

        image_ref_sam = preprocess_imgs_for_sam2(img_ref)
        image_refs_sam.append(image_ref_sam)
        lbl_refs.append(lbl_ref)

    # Create array for SAM2
    all_images = np.stack(image_refs_sam + [image_query_sam], axis=0)
    last_index = all_images.shape[0] - 1
    try:
        # Initialize SAM2
        inference_state = predictor.init_state_by_np_data(all_images)

        for i, (_, lbl_ref) in enumerate(match_imgs_lbls):
            # Add reference frame masks
            class_masks_ref = extract_class_masks_from_label(lbl_ref, n_classes)
            for j in range(1, n_classes + 1):
                mask_index = j - 1

                if (
                    mask_index < len(class_masks_ref)
                    and class_masks_ref[mask_index].sum() > 0
                ):
                    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=i,
                        obj_id=j,  # Object IDs start from 1
                        mask=class_masks_ref[mask_index],
                    )
                    print(f"Added reference class {j} mask to frame {i}")

        # Propagate to test frame
        out_frame_idx, out_obj_ids, out_mask_logits = next(
            predictor.propagate_in_video(inference_state, start_frame_idx=last_index)
        )

        print(f"Propagated to frame {out_frame_idx}, found {len(out_obj_ids)} objects")

        if out_mask_logits is not None:
            # Process predictions following the reference code pattern
            if isinstance(out_mask_logits, torch.Tensor):
                mask_logits = out_mask_logits.detach().cpu().numpy()
            else:
                mask_logits = out_mask_logits

            # Extract class predictions
            class_preds = []
            for i in range(len(out_obj_ids)):
                class_pred = mask_logits[i] > 0.0
                if len(class_pred.shape) > 2:
                    class_pred = class_pred.squeeze()
                class_preds.append(class_pred)

            # Compute background prediction
            if len(class_preds) > 0:
                background_pred = np.logical_not(np.logical_or.reduce(class_preds))
            else:
                background_pred = np.ones_like(image_query_sam, dtype=bool)

            class_preds.insert(0, background_pred)  # Add background as class 0

            # Create final prediction label
            class_preds_array = np.stack(class_preds, axis=0)
            pred_label = np.argmax(class_preds_array, axis=0)

            print(f"Final prediction shape: {pred_label.shape}")
            print(f"Predicted classes: {np.unique(pred_label)}")

            # Visualize results
            overlayed_image = visualize_multiclass_results(
                image_query_sam, class_preds, lbl_refs, image_refs_sam, lbl_query
            )

            # Calculate metrics if ground truth available
            if lbl_ref is not None:
                metrics = calculate_class_metrics(pred_label, lbl_ref, n_classes)

            return pred_label, overlayed_image, class_preds, metrics, image_query_sam

    except Exception as e:
        print(f"Error during multiclass SAM2 processing: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None, None


def preprocess_imgs_for_dino(images):
    """Preprocess image specifically for DINO"""
    proc_imgs = []
    for image in images:
        # Handle different input shapes
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)  # Remove batch dimension
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(-1)  # Remove channel dimension if single channel

        # Ensure 2D
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1 if image.shape[-1] <= 3 else 0)

        # Normalize to [0, 255]
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image_normalized = (
                (image - image_min) / (image_max - image_min) * 255
            ).astype(np.uint8)
        else:
            image_normalized = np.zeros_like(image, dtype=np.uint8)

        # Convert grayscale to RGB by repeating channels
        image_rgb = np.stack([image_normalized] * 3, axis=-1)

        pil_image = Image.fromarray(image_rgb)
        proc_imgs.append(pil_image)

    return proc_imgs


def get_dino_embeddings(images, processor, model):
    proc_imgs = preprocess_imgs_for_dino(images)

    embeddings = []
    for image in proc_imgs:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.pooler_output.detach().cpu().numpy()
        embeddings.append(embedding)

    return embeddings


def create_faiss_index(index, processor, model):
    # patient_nums=["003", "010", "027", "032", "034"]
    patient_nums = ["076", "091", "062", "055", "042"]
    db_paths = create_dataset_paths(
        "../dataset/ACDC_2d_slices/Training", patient_nums=patient_nums
    )
    # db_paths = create_dataset_paths("../dataset/ACDC_2d_slices/Training")
    imgs_db, lbls_db = load_medical_imgs(db_paths)
    imgs_embds = get_dino_embeddings(imgs_db, processor, model)

    for img_embd in imgs_embds:
        index.add(img_embd)

    id_map = list(zip(imgs_db, lbls_db))
    return index, id_map


def get_query_imgs():
    query_paths = create_validation_paths(
        "../dataset/ACDC_2d_slices/Validation", patient_nums=["019"]
    )
    # query_paths = create_validation_paths("../dataset/ACDC_2d_slices/Validation")
    imgs_query, lbls_query = load_medical_imgs(query_paths)
    print(f"Loaded {len(lbls_query)} query images")
    imgs_embds = get_dino_embeddings(imgs_query, processor, model)
    return imgs_query, imgs_embds, lbls_query


def select_k_closest(index, id_map, img_embedding, k=1):
    _, indices = index.search(img_embedding, k)
    results = []
    for pos, i in enumerate(indices[0]):
        img, lbl = id_map[i]
        results.append((img, lbl))

    return results


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor, model = load_dino_model()
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoints/sam2.1_hiera_tiny.pt",
        device=device,
    )

    index = faiss.IndexFlatIP(768)
    index, id_map = create_faiss_index(index, processor, model)
    query_imgs, query_embds, lbls_query = get_query_imgs()
    match_imgs_lbls = select_k_closest(index, id_map, query_embds[0], k=5)

    query_index = 1
    pred_label, overlayed_image, class_predictions, metrics, img_query_sam = (
        main_multiclass(
            predictor, query_imgs[query_index], lbls_query[query_index], match_imgs_lbls
        )
    )

    if pred_label is not None:
        print("\n=== Segmentation Results ===")
        print(f"Prediction shape: {pred_label.shape}")
        print(f"Classes found: {np.unique(pred_label)}")

        # You can also create custom overlays with specific colors
        custom_overlay, legend = create_multiclass_overlay(
            img_query_sam, class_predictions, MEDICAL_COLORS, alpha=0.5
        )

        # Display custom overlay
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_query_sam, cmap="gray")
        plt.title("Original Test Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(custom_overlay)
        plt.title("Multi-class Prediction Overlay")
        plt.axis("off")

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=np.array(color) / 255.0, label=name)
            for name, color in legend
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig("multiclass_overlay_custom.png", dpi=300, bbox_inches="tight")
        plt.show()
