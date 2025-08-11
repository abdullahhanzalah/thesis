import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import faiss
import torch
from sam2.build_sam import build_sam2_video_predictor
import h5py
from visual import *

def load_dino_model():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    return processor, model

def get_dino_embeddings(image, processor, model):
    if isinstance(image, np.ndarray):
        image = preprocess_img_for_dino(image)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.pooler_output.detach().cpu().numpy()
    return embeddings


def load_image():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    print(image.height, image.width)  # [480, 640]
    return image


# def preprocess_img(image):
#     image_min, image_max = image.min(), image.max()
#     if image_max > image_min:
#         image_normalized = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
#     else:
#         image_normalized = np.zeros_like(image, dtype=np.uint8)
    
#     # Convert grayscale to RGB by repeating channels
#     image_rgb = np.stack([image_normalized] * 3, axis=-1)
    
#     # Convert to PIL Image for processor compatibility
#     pil_image = Image.fromarray(image_rgb)

#     print(f"HERE: {pil_image.size}")  # Print size to verify
#     return pil_image

def preprocess_img_for_dino(image):
    """Preprocess image specifically for DINO"""
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
        image_normalized = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
    else:
        image_normalized = np.zeros_like(image, dtype=np.uint8)
    
    # Convert grayscale to RGB by repeating channels
    image_rgb = np.stack([image_normalized] * 3, axis=-1)
    
    # Convert to PIL Image for processor compatibility
    pil_image = Image.fromarray(image_rgb)
    print(f"DINO preprocessed image size: {pil_image.size}")
    return pil_image

def preprocess_img_for_sam2(image):
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
    
    print(f"SAM2 preprocessed image shape: {image_normalized.shape}, dtype: {image_normalized.dtype}")
    return image_normalized.astype(np.float32)


def load_medical_img():
    h5f = h5py.File("../dataset/ACDC_2d_slices/Training/patient003_SA_ED_slice_2.h5", "r")
    image_train = h5f["image"][:]
    label_train = h5f["label"][:]
    h5f = h5py.File("../dataset/ACDC_2d_slices/Training/patient006_SA_ES_slice_4.h5", "r")
    image_test = h5f["image"][:]
    return image_train, label_train, image_test

n_classes = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor, model = load_dino_model()
# image = load_image()
image_train, label_train, image_test = load_medical_img()
print(image_train.dtype, image_train.shape)
print(np.unique(label_train))  # Check unique labels in training data

class_masks = extract_class_masks_from_label(label_train, n_classes)

dino_embeddings = get_dino_embeddings(image_train, processor, model)

index = faiss.IndexFlatIP(768)
index.add(dino_embeddings)

predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt", device=device)

image_train_sam = preprocess_img_for_sam2(image_train)
image_test_sam = preprocess_img_for_sam2(image_test)

# Create array for SAM2 (it expects a batch of images)
all_images = np.stack([image_train_sam, image_test_sam])  # Shape: (2, H, W)
print(f"All images shape for SAM2: {image_test_sam.shape}")

# all_images = np.array([image_train_sam, image_test_sam]) # create array with [sup_img, query_img]
inference_state = predictor.init_state_by_np_data(all_images)

for i in range(n_classes):
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=1, mask=class_masks[i])

out_frame_idx, out_obj_ids, out_mask_logits = next(predictor.propagate_in_video(inference_state, start_frame_idx=1))

# prediction_mask, overlayed_image = visualize_sam2_results(
            # image_test_sam, out_mask_logits, label_train
        # )

# Save results
# if prediction_mask is not None:
#     save_results(image_test_sam, prediction_mask, overlayed_image, "sam2_medical")

# class_preds = [(out_mask_logits[i] > 0.0).squeeze().cpu().numpy() for i in range(n_classes)]            
# background_pred = np.logical_not(np.logical_or.reduce(class_preds)) # Compute background predictions
# class_preds.insert(0, background_pred)
# class_preds = np.stack(class_preds, axis=0)
# pred_label = np.argmax(class_preds, axis=0)
# true_label = np.argmax(true_label, axis=0)

if out_mask_logits is not None:
    # Process predictions following the reference code pattern
    if isinstance(out_mask_logits, torch.Tensor):
        mask_logits = out_mask_logits.detach().cpu().numpy()
    else:
        mask_logits = out_mask_logits
    
    # Extract class predictions
    class_preds = []
    for i in range(len(out_obj_ids)):
        class_pred = (mask_logits[i] > 0.0)
        if len(class_pred.shape) > 2:
            class_pred = class_pred.squeeze()
        class_preds.append(class_pred)
    
    # Compute background prediction
    if len(class_preds) > 0:
        background_pred = np.logical_not(np.logical_or.reduce(class_preds))
    else:
        background_pred = np.ones_like(image_test_sam, dtype=bool)
    
    class_preds.insert(0, background_pred)  # Add background as class 0
    
    # Create final prediction label
    class_preds_array = np.stack(class_preds, axis=0)
    pred_label = np.argmax(class_preds_array, axis=0)
    
    print(f"Final prediction shape: {pred_label.shape}")
    print(f"Predicted classes: {np.unique(pred_label)}")
    
    # Visualize results
    overlayed_image = visualize_multiclass_results(
        image_test_sam, class_preds, label_train
    )
    
    # Calculate metrics if ground truth available
    if label_train is not None:
        metrics = calculate_class_metrics(pred_label, label_train, n_classes)
