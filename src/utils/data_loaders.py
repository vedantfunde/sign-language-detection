#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Loading Utilities
--------------------
Utilities for loading and preprocessing data files.
"""

import pickle
import os
import pandas as pd
import numpy as np
import logging
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_pickle(pickle_path):
    """
    Load data from a pickle file.
    
    Args:
        pickle_path (str): Path to the pickle file
        
    Returns:
        object: Content of the pickle file
    """
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded: {pickle_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file {pickle_path}: {str(e)}")
        raise

def save_pickle(data, output_path):
    """
    Save data to a pickle file.
    
    Args:
        data (object): Data to save
        output_path (str): Path to save the pickle file
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Successfully saved: {output_path}")
    except Exception as e:
        logger.error(f"Error saving pickle file {output_path}: {str(e)}")
        raise

def load_word_labels(csv_path):
    """
    Load word labels from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file with word labels
        
    Returns:
        dict: Mapping from video IDs to word labels
    """
    try:
        word_labels = {}
        
        # Check file extension
        if csv_path.endswith('.csv'):
            # Use pandas for standard CSV
            df = pd.read_csv(csv_path)
            
            # Assuming format: video_id, word_label
            for _, row in df.iterrows():
                if len(row) >= 2:
                    video_id = str(row[0])
                    word_label = str(row[1])
                    word_labels[video_id] = word_label
        else:
            # Try to read as plain text for custom formats
            with open(csv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        video_id = parts[0]
                        word_label = parts[1]
                        word_labels[video_id] = word_label
        
        logger.info(f"Loaded {len(word_labels)} word labels from {csv_path}")
        return word_labels
    
    except Exception as e:
        logger.error(f"Error loading word labels from {csv_path}: {str(e)}")
        raise

def create_word_to_label_mapping(word_labels):
    """
    Create a word-to-label index mapping.
    
    Args:
        word_labels (dict): Dictionary of video_id -> word_label
        
    Returns:
        dict: Mapping from words to label indices
    """
    unique_words = sorted(set(word_labels.values()))
    word_to_label = {word: idx for idx, word in enumerate(unique_words)}
    
    logger.info(f"Created mapping for {len(word_to_label)} unique words")
    return word_to_label

def load_features_and_labels(features_path, labels_path):
    """
    Load both features and labels for training or evaluation.
    
    Args:
        features_path (str): Path to features pickle file
        labels_path (str): Path to labels CSV file
        
    Returns:
        tuple: (features, labels, word_to_label)
    """
    try:
        # Load features
        features_data = load_pickle(features_path)
        
        # Load word labels
        word_labels = load_word_labels(labels_path)
        
        # Create word-to-label mapping
        word_to_label = create_word_to_label_mapping(word_labels)
        
        # Map word labels to numerical indices
        labels = []
        for video_id, word in word_labels.items():
            if video_id in features_data:
                label_idx = word_to_label[word]
                labels.append((video_id, label_idx))
        
        # Extract features in same order as labels
        feature_list = []
        for video_id, _ in labels:
            if video_id in features_data:
                feature_list.append(features_data[video_id])
        
        # Convert labels to array (just the indices)
        label_indices = np.array([label_idx for _, label_idx in labels])
        
        logger.info(f"Loaded {len(feature_list)} feature-label pairs")
        return feature_list, label_indices, word_to_label
    
    except Exception as e:
        logger.error(f"Error loading features and labels: {str(e)}")
        raise

def combine_features(feature_folders, output_path=None):
    """
    Combine features from multiple folders into a single feature set.
    
    Args:
        feature_folders (list): List of folders containing feature files
        output_path (str, optional): Path to save the combined features
        
    Returns:
        dict: Combined features
    """
    try:
        combined_features = {}
        
        for folder in feature_folders:
            for filename in os.listdir(folder):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(folder, filename)
                    features = load_pickle(file_path)
                    
                    # Assuming features is a dictionary with video IDs as keys
                    combined_features.update(features)
        
        logger.info(f"Combined {len(combined_features)} feature sets")
        
        # Save if output path is provided
        if output_path:
            save_pickle(combined_features, output_path)
        
        return combined_features
    
    except Exception as e:
        logger.error(f"Error combining features: {str(e)}")
        raise 