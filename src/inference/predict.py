#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sign Language Gesture Prediction
-------------------------------
Module for predicting sign language gestures from video frames or features.
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
from models.transformer import GestureTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_label_mapping(mapping_file):
    """
    Load the word-to-label mapping from a pickle file.
    
    Args:
        mapping_file (str): Path to the pickle file with word-to-label mapping
        
    Returns:
        tuple: (word_to_label, label_to_word) dictionaries
    """
    try:
        with open(mapping_file, "rb") as f:
            word_to_label = pickle.load(f)
        
        # Create reverse mapping
        label_to_word = {v: k for k, v in word_to_label.items()}
        return word_to_label, label_to_word
    
    except Exception as e:
        logger.error(f"Error loading label mapping: {str(e)}")
        raise

def load_model(model_path, num_classes):
    """
    Load a trained GestureTransformer model.
    
    Args:
        model_path (str): Path to the model checkpoint file
        num_classes (int): Number of sign language classes
        
    Returns:
        GestureTransformer: Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize model
        model = GestureTransformer(num_classes=num_classes).to(device)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"Model loaded from {model_path} (using {device})")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_feature_extractor():
    """
    Load ResNet50 model for feature extraction.
    
    Returns:
        nn.Module: ResNet model without the final classification layer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load pre-trained ResNet and remove final FC layer
        resnet = models.resnet50(pretrained=True)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        resnet.eval().to(device)
        
        return resnet
    
    except Exception as e:
        logger.error(f"Error loading feature extractor: {str(e)}")
        raise

def get_image_transform():
    """
    Get the preprocessing transformation for input images.
    
    Returns:
        transforms.Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_features_from_image(image_path, feature_extractor, transform):
    """
    Extract features from a single image.
    
    Args:
        image_path (str): Path to the image file
        feature_extractor (nn.Module): ResNet model for feature extraction
        transform (transforms.Compose): Preprocessing transformation
        
    Returns:
        numpy.ndarray: Extracted features of shape (2048,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(image)
        
        return features.view(-1, 2048).cpu().numpy()
    
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {str(e)}")
        raise

def extract_features_from_folder(folder_path, feature_extractor=None, transform=None):
    """
    Extract features from all images in a folder.
    
    Args:
        folder_path (str): Path to the folder containing images
        feature_extractor (nn.Module, optional): ResNet model
        transform (transforms.Compose, optional): Preprocessing transformation
        
    Returns:
        numpy.ndarray: Extracted features of shape (n_images, 2048)
    """
    if feature_extractor is None:
        feature_extractor = load_feature_extractor()
        
    if transform is None:
        transform = get_image_transform()
    
    features_list = []
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    try:
        # Get all image files and sort them to maintain sequence
        image_files = [f for f in os.listdir(folder_path) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        image_files.sort()
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        # Extract features from each image
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            features = extract_features_from_image(img_path, feature_extractor, transform)
            features_list.append(features)
        
        # Convert list to array
        return np.vstack(features_list) if features_list else np.zeros((1, 2048))
    
    except Exception as e:
        logger.error(f"Error extracting features from folder {folder_path}: {str(e)}")
        raise

def predict_gesture(model, features, label_to_word):
    """
    Predict gesture class from features.
    
    Args:
        model (GestureTransformer): Trained model
        features (numpy.ndarray): Features of shape (seq_len, feature_dim)
        label_to_word (dict): Mapping from label indices to words
        
    Returns:
        str: Predicted gesture word
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Get prediction from model
        predicted_idx = model.predict(features, device)
        
        # Map index to word
        if predicted_idx in label_to_word:
            return label_to_word[predicted_idx]
        else:
            logger.warning(f"Unknown label index: {predicted_idx}")
            return f"unknown_{predicted_idx}"
    
    except Exception as e:
        logger.error(f"Error predicting gesture: {str(e)}")
        raise

def load_features_from_file(features_file):
    """
    Load pre-extracted features from a pickle file.
    
    Args:
        features_file (str): Path to the pickle file with features
        
    Returns:
        numpy.ndarray: Loaded features
    """
    try:
        with open(features_file, "rb") as f:
            features = pickle.load(f)
        
        logger.info(f"Loaded features from {features_file}, shape: {features.shape}")
        return features
    
    except Exception as e:
        logger.error(f"Error loading features from {features_file}: {str(e)}")
        raise

def predict_from_features(model_path, features_path, mapping_file, output_file=None):
    """
    Predict gesture from pre-extracted features.
    
    Args:
        model_path (str): Path to the model checkpoint
        features_path (str): Path to the features file
        mapping_file (str): Path to the label mapping file
        output_file (str, optional): Path to save the prediction result
        
    Returns:
        str: Predicted gesture
    """
    try:
        # Load label mapping
        word_to_label, label_to_word = load_label_mapping(mapping_file)
        
        # Load model
        model = load_model(model_path, num_classes=len(word_to_label))
        
        # Load features
        features = load_features_from_file(features_path)
        
        # Make prediction
        prediction = predict_gesture(model, features, label_to_word)
        
        # Output result
        logger.info(f"Predicted Gesture: {prediction}")
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"Predicted Gesture: {prediction}\n")
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error in prediction from features: {str(e)}")
        raise

def predict_from_images(model_path, images_folder, mapping_file, output_file=None):
    """
    Predict gesture from a folder of images.
    
    Args:
        model_path (str): Path to the model checkpoint
        images_folder (str): Path to the folder with images
        mapping_file (str): Path to the label mapping file
        output_file (str, optional): Path to save the prediction result
        
    Returns:
        str: Predicted gesture
    """
    try:
        # Load label mapping
        word_to_label, label_to_word = load_label_mapping(mapping_file)
        
        # Load model
        model = load_model(model_path, num_classes=len(word_to_label))
        
        # Extract features
        feature_extractor = load_feature_extractor()
        transform = get_image_transform()
        features = extract_features_from_folder(images_folder, feature_extractor, transform)
        
        # Make prediction
        prediction = predict_gesture(model, features, label_to_word)
        
        # Output result
        logger.info(f"Predicted Gesture: {prediction}")
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"Predicted Gesture: {prediction}\n")
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error in prediction from images: {str(e)}")
        raise

def main():
    """Command line interface for prediction."""
    parser = argparse.ArgumentParser(description="Predict sign language gestures")
    parser.add_argument("--model", required=True, help="Path to the model checkpoint")
    parser.add_argument("--labels", required=True, help="Path to the label mapping file")
    parser.add_argument("--output", help="Path to save prediction output")
    
    # Exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--features", help="Path to pre-extracted features file")
    input_group.add_argument("--images", help="Path to folder with image frames")
    
    args = parser.parse_args()
    
    # Configure file logging if output specified
    if args.output:
        log_file = os.path.splitext(args.output)[0] + ".log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # Run prediction based on input type
    if args.features:
        predict_from_features(args.model, args.features, args.labels, args.output)
    elif args.images:
        predict_from_images(args.model, args.images, args.labels, args.output)

if __name__ == "__main__":
    main() 