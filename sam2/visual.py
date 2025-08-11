import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import torch

def overlay_mask_on_image(image, mask, alpha=0.5, color=[255, 0, 0]):
    """
    Overlay mask on image with specified color and transparency
    
    Args:
        image: numpy array of shape (H, W) or (H, W, 3)
        mask: numpy array of shape (H, W) with boolean or 0/1 values
        alpha: transparency of overlay (0.0 to 1.0)
        color: RGB color for the mask overlay [R, G, B]
    
    Returns:
        overlayed_image: numpy array with mask overlayed
    """
    # Ensure image is in the right format
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)
    
    # Normalize image to [0, 255] if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Ensure mask is boolean
    if mask.dtype != bool:
        mask = mask > 0.5
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask] = color
    
    # Blend the image and mask
    overlayed = image.copy()
    mask_area = mask[..., np.newaxis] if len(mask.shape) == 2 else mask
    overlayed = np.where(mask_area, 
                        (1 - alpha) * image + alpha * colored_mask, 
                        image)
    
    return overlayed.astype(np.uint8)

def create_side_by_side_visualization(original_image, mask, prediction_mask, 
                                    original_title="Original", 
                                    mask_title="Ground Truth", 
                                    pred_title="Prediction"):
    """
    Create side-by-side visualization of original, ground truth, and prediction
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    if len(original_image.shape) == 2:
        axes[0].imshow(original_image, cmap='gray')
    else:
        axes[0].imshow(original_image)
    axes[0].set_title(original_title)
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(mask_title)
    axes[1].axis('off')
    
    # Prediction mask
    axes[2].imshow(prediction_mask, cmap='gray')
    axes[2].set_title(pred_title)
    axes[2].axis('off')
    
    # Overlay
    overlayed = overlay_mask_on_image(original_image, prediction_mask, alpha=0.4, color=[255, 0, 0])
    axes[3].imshow(overlayed)
    axes[3].set_title("Prediction Overlay")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    return fig

def visualize_sam2_results(image_test_sam, out_mask_logits, label_train=None, threshold=0.0):
    """
    Visualize SAM2 results with overlay
    
    Args:
        image_test_sam: the test image (numpy array)
        out_mask_logits: SAM2 output mask logits
        label_train: ground truth label for comparison (optional)
        threshold: threshold for converting logits to binary mask
    """
    # Convert logits to binary mask
    if out_mask_logits is not None:
        # Handle different output formats
        if isinstance(out_mask_logits, torch.Tensor):
            mask_logits = out_mask_logits.detach().cpu().numpy()
        else:
            mask_logits = out_mask_logits
        
        # If there are multiple objects, take the first one
        if len(mask_logits.shape) == 3:  # (num_objects, H, W)
            mask_logits = mask_logits[0]
        elif len(mask_logits.shape) == 4:  # (batch, num_objects, H, W)
            mask_logits = mask_logits[0, 0]
        
        # Convert logits to binary mask
        prediction_mask = mask_logits > threshold
        
        print(f"Prediction mask shape: {prediction_mask.shape}")
        print(f"Prediction mask unique values: {np.unique(prediction_mask)}")
        
        # Create visualizations
        if label_train is not None:
            create_side_by_side_visualization(
                image_test_sam, 
                label_train, 
                prediction_mask,
                "Test Image", 
                "Ground Truth", 
                "SAM2 Prediction"
            )
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original image
            axes[0].imshow(image_test_sam, cmap='gray')
            axes[0].set_title("Test Image")
            axes[0].axis('off')
            
            # Prediction mask
            axes[1].imshow(prediction_mask, cmap='gray')
            axes[1].set_title("SAM2 Prediction")
            axes[1].axis('off')
            
            # Overlay
            overlayed = overlay_mask_on_image(image_test_sam, prediction_mask, alpha=0.4, color=[255, 0, 0])
            axes[2].imshow(overlayed)
            axes[2].set_title("Prediction Overlay")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return prediction_mask, overlayed
    else:
        print("No mask logits to visualize")
        return None, None

def save_results(image, mask, overlay, filename_prefix="result"):
    """Save the visualization results"""
    # Save original image
    if len(image.shape) == 2:
        plt.imsave(f"{filename_prefix}_image.png", image, cmap='gray')
    else:
        plt.imsave(f"{filename_prefix}_image.png", image)
    
    # Save mask
    plt.imsave(f"{filename_prefix}_mask.png", mask, cmap='gray')
    
    # Save overlay
    plt.imsave(f"{filename_prefix}_overlay.png", overlay)
    
    print(f"Results saved with prefix: {filename_prefix}")


def calculate_class_metrics(pred_label, true_label, n_classes):
    """Calculate metrics for each class"""
    metrics = {}
    
    for class_id in range(n_classes + 1):  # Include background
        true_mask = (true_label == class_id)
        pred_mask = (pred_label == class_id)
        
        # Calculate IoU
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        iou = intersection / union if union > 0 else 0
        
        # Calculate Dice coefficient
        dice = 2 * intersection / (true_mask.sum() + pred_mask.sum()) if (true_mask.sum() + pred_mask.sum()) > 0 else 0
        
        class_name = "Background" if class_id == 0 else f"Class {class_id}"
        metrics[class_name] = {'IoU': iou, 'Dice': dice}
        
        print(f"{class_name}: IoU={iou:.3f}, Dice={dice:.3f}")
    
    return metrics

def save_multiclass_results(image, pred_label, overlay, class_predictions, prefix):
    """Save all multiclass results"""
    # Save main results
    plt.imsave(f"{prefix}_image.png", image, cmap='gray')
    plt.imsave(f"{prefix}_prediction.png", pred_label, cmap='tab10')
    plt.imsave(f"{prefix}_overlay.png", overlay)
    
    # Save individual class masks
    for i, class_mask in enumerate(class_predictions):
        if i == 0:  # Background
            plt.imsave(f"{prefix}_class_background.png", class_mask, cmap='gray')
        else:
            plt.imsave(f"{prefix}_class_{i}.png", class_mask, cmap='gray')
    
    print(f"All results saved with prefix: {prefix}")


def visualize_multiclass_results(image_test_sam, class_predictions, label_train=None, 
                                class_colors=None, save_prefix="multiclass_sam2"):
    """
    Comprehensive visualization for multi-class segmentation results
    """
    n_classes = len(class_predictions) - 1 if class_predictions else 0  # Exclude background
    
    if class_predictions is None or len(class_predictions) == 0:
        print("No predictions to visualize")
        return None, None
    
    # Create combined prediction label
    class_preds_array = np.stack(class_predictions, axis=0)
    pred_label = np.argmax(class_preds_array, axis=0)
    
    # Create multiclass overlay
    overlayed_image, legend_info = create_multiclass_overlay(
        image_test_sam, class_predictions, class_colors
    )
    
    # Create visualization
    if label_train is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Top row: Original, Ground Truth, Prediction
        axes[0, 0].imshow(image_test_sam, cmap='gray')
        axes[0, 0].set_title("Test Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(label_train, cmap='tab10', vmin=0, vmax=n_classes)
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_label, cmap='tab10', vmin=0, vmax=n_classes)
        axes[0, 2].set_title("SAM2 Prediction")
        axes[0, 2].axis('off')
        
        # Bottom row: Individual class predictions and overlay
        axes[1, 0].imshow(overlayed_image)
        axes[1, 0].set_title("Multi-class Overlay")
        axes[1, 0].axis('off')
        
        # Show individual class masks
        if n_classes >= 1:
            axes[1, 1].imshow(class_predictions[1], cmap='gray')
            axes[1, 1].set_title("Class 1 Prediction")
            axes[1, 1].axis('off')
        
        if n_classes >= 2:
            axes[1, 2].imshow(class_predictions[2], cmap='gray')
            axes[1, 2].set_title("Class 2 Prediction")
            axes[1, 2].axis('off')
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(image_test_sam, cmap='gray')
        axes[0].set_title("Test Image")
        axes[0].axis('off')
        
        axes[1].imshow(pred_label, cmap='tab10', vmin=0, vmax=n_classes)
        axes[1].set_title("SAM2 Prediction")
        axes[1].axis('off')
        
        axes[2].imshow(overlayed_image)
        axes[2].set_title("Multi-class Overlay")
        axes[2].axis('off')
    
    # Add legend
    legend_elements = []
    for class_name, color in legend_info:
        legend_elements.append(plt.Rectangle((0,0),1,1, 
                              facecolor=np.array(color)/255.0, 
                              label=class_name))
    
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual results
    save_multiclass_results(image_test_sam, pred_label, overlayed_image, 
                           class_predictions, save_prefix)
    
    return pred_label, overlayed_image

def create_multiclass_overlay(image, class_predictions, class_colors=None, alpha=0.6):
    """
    Create overlay with different colors for each class
    
    Args:
        image: original image (H, W) or (H, W, 3)
        class_predictions: list of boolean masks for each class (including background)
        class_colors: list of RGB colors for each class
        alpha: transparency of overlay
    
    Returns:
        overlayed_image: image with colored class overlays
        legend_info: list of (class_name, color) for legend
    """
    # Default colors for classes
    if class_colors is None:
        class_colors = [
    [0, 0, 0],          # Background (black)
    [255, 0, 0],        # Class 1: e.g., Left Ventricle (red)
    [0, 255, 0],        # Class 2: e.g., Myocardium (green)  
    [0, 0, 255],        # Class 3: e.g., Right Ventricle (blue)
    [255, 255, 0],      # Class 4: (yellow)
    [255, 0, 255],      # Class 5: (magenta)
]
    
    # Ensure image is in RGB format
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Normalize image to [0, 255] if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    overlayed = image.copy().astype(np.float32)
    legend_info = []
    
    # Apply each class overlay (skip background class 0)
    for i, class_mask in enumerate(class_predictions):
        if i == 0:  # Skip background
            continue
            
        if class_mask.sum() > 0:  # Only overlay if class has pixels
            color = class_colors[i % len(class_colors)]
            class_overlay = np.zeros_like(overlayed)
            
            # Create colored overlay for this class
            mask_3d = class_mask[..., np.newaxis]
            class_overlay[class_mask] = color
            
            # Blend with existing overlay
            overlayed = np.where(mask_3d, 
                               (1 - alpha) * overlayed + alpha * class_overlay, 
                               overlayed)
            
            legend_info.append((f"Class {i}", color))
    
    return overlayed.astype(np.uint8), legend_info


def extract_class_masks_from_label(label, n_classes):
    """
    Extract individual class masks from multi-class label
    
    Args:
        label: numpy array with integer class labels (0=background, 1=class1, 2=class2, etc.)
        n_classes: number of classes (excluding background)
    
    Returns:
        class_masks: list of boolean masks for each class
    """
    class_masks = []
    for class_id in range(1, n_classes + 1):  # Start from 1, skip background (0)
        class_mask = (label == class_id)
        class_masks.append(class_mask)
        print(f"Class {class_id} mask: {class_mask.sum()} pixels")
    
    return class_masks
