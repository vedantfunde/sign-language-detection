#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Extraction Module
-----------------------
Extract features from video frames using pre-trained models.
"""

import torch
import numpy as np
import os
import argparse
import pickle
import logging
from PIL import Image
from torchvision import transforms, models
import sys

# Add parent directory to path to allow imports from other project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loaders import save_pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_feature_extractor(model_name='resnet50'):
    """
    Load a pre-trained model for feature extraction.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        
    Returns:
        nn.Module: Feature extraction model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if model_name == 'resnet50':
            # Load pre-trained ResNet and remove final FC layer
            model = models.resnet50(pretrained=True)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
        elif model_name == 'vit':
            # ViT base model
            model = models.vit_b_16(pretrained=True)
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_dim = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        feature_extractor.eval().to(device)
        logger.info(f"Loaded {model_name} feature extractor (output dim: {feature_dim})")
        
        return feature_extractor, feature_dim
    
    except Exception as e:
        logger.error(f"Error loading feature extractor: {str(e)}")
        raise

def get_image_transform(model_name='resnet50'):
    """
    Get preprocessing transformations based on the model.
    
    Args:
        model_name (str): Name of the pre-trained model
        
    Returns:
        transforms.Compose: Composed transformations
    """
    if model_name in ['resnet50', 'resnet101']:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'vit':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def extract_features_from_image(image_path, feature_extractor, transform):
    """
    Extract features from a single image.
    
    Args:
        image_path (str): Path to the image file
        feature_extractor (nn.Module): Model for feature extraction
        transform (transforms.Compose): Preprocessing transformation
        
    Returns:
        numpy.ndarray: Extracted features
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(image)
        
        # Flatten the features
        features = features.squeeze().cpu().numpy()
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {str(e)}")
        raise

def extract_features_from_folder(folder_path, feature_extractor, transform, max_samples=None):
    """
    Extract features from all images in a folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        feature_extractor (nn.Module): Model for feature extraction
        transform (transforms.Compose): Preprocessing transformation
        max_samples (int, optional): Maximum number of samples to process
        
    Returns:
        numpy.ndarray: Extracted features of shape (n_images, feature_dim)
    """
    features_list = []
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    try:
        # Get all image files and sort them
        image_files = [
            f for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        image_files.sort()
        
        if max_samples:
            image_files = image_files[:max_samples]
            
        logger.info(f"Processing {len(image_files)} images from {folder_path}")
        
        # Extract features from each image
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(folder_path, img_file)
            features = extract_features_from_image(img_path, feature_extractor, transform)
            features_list.append(features)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images")
        
        # Stack features
        features_array = np.stack(features_list) if features_list else np.array([])
        logger.info(f"Extracted features with shape: {features_array.shape}")
        
        return features_array
    
    except Exception as e:
        logger.error(f"Error extracting features from folder {folder_path}: {str(e)}")
        raise

def process_video_folders(base_dir, output_path, model_name='resnet50', max_samples=None):
    """
    Process multiple video folders for feature extraction.
    
    Args:
        base_dir (str): Base directory containing video folders
        output_path (str): Path to save the features
        model_name (str): Name of the pre-trained model
        max_samples (int, optional): Maximum number of frames per video
        
    Returns:
        dict: Dictionary mapping video IDs to features
    """
    try:
        # Load feature extractor and transform
        feature_extractor, _ = load_feature_extractor(model_name)
        transform = get_image_transform(model_name)
        
        all_features = {}
        
        # Process each subdirectory as a separate video
        for video_id in os.listdir(base_dir):
            video_dir = os.path.join(base_dir, video_id)
            
            if os.path.isdir(video_dir):
                logger.info(f"Processing video: {video_id}")
                features = extract_features_from_folder(
                    video_dir, feature_extractor, transform, max_samples
                )
                all_features[video_id] = features
        
        # Save features to file
        save_pickle(all_features, output_path)
        logger.info(f"Saved features for {len(all_features)} videos to {output_path}")
        
        return all_features
    
    except Exception as e:
        logger.error(f"Error processing video folders: {str(e)}")
        raise

def main():
    """Command line interface for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features from video frames")
    parser.add_argument("-i", "--input", required=True, 
                       help="Input directory containing video frame folders")
    parser.add_argument("-o", "--output", required=True,
                       help="Output path for feature pickle file")
    parser.add_argument("-m", "--model", default="resnet50",
                       choices=["resnet50", "resnet101", "vit"],
                       help="Model to use for feature extraction")
    parser.add_argument("-l", "--log", help="Path to log file")
    parser.add_argument("--max-samples", type=int, 
                       help="Maximum number of frames to process per video")
    
    args = parser.parse_args()
    
    # Configure file logging if specified
    if args.log:
        file_handler = logging.FileHandler(args.log)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    process_video_folders(
        args.input, args.output, args.model, args.max_samples
    )

if __name__ == "__main__":
    main() 