"""
Example: Complete workflow demonstration for computer vision project.
This script demonstrates how to use all components together.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from data_preprocessing import (
    get_preprocessing_pipeline,
    CustomImageDataset,
    create_data_loaders,
    denormalize_image
)


def create_dummy_data(num_samples: int = 100, num_classes: int = 10):
    """
    Create dummy image data for demonstration.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        
    Returns:
        List of images and labels
    """
    print(f"Creating {num_samples} dummy images...")
    
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        images.append(img)
        
        # Random label
        label = np.random.randint(0, num_classes)
        labels.append(label)
    
    return images, labels


def create_simple_model(num_classes: int = 10):
    """
    Create a simple CNN model for demonstration.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    model = nn.Sequential(
        # Convolutional layers
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        
        # Flatten
        nn.Flatten(),
        
        # Fully connected layers
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes)
    )
    
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (CPU or GPU)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, loader, criterion, device):
    """
    Validate model.
    
    Args:
        model: PyTorch model
        loader: Data loader
        criterion: Loss function
        device: Device (CPU or GPU)
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    """
    Main function demonstrating the complete workflow.
    """
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 60 + "\n")
    
    # Configuration
    NUM_SAMPLES = 100
    NUM_CLASSES = 10
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    
    # 1. Create dummy data
    print("Step 1: Creating dummy data...")
    images, labels = create_dummy_data(NUM_SAMPLES, NUM_CLASSES)
    print(f"✓ Created {len(images)} images with {NUM_CLASSES} classes\n")
    
    # 2. Split data into train and validation
    print("Step 2: Splitting data...")
    split_idx = int(0.8 * NUM_SAMPLES)
    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    val_images = images[split_idx:]
    val_labels = labels[split_idx:]
    print(f"✓ Train: {len(train_images)} samples")
    print(f"✓ Validation: {len(val_images)} samples\n")
    
    # 3. Create preprocessing pipelines
    print("Step 3: Creating preprocessing pipelines...")
    train_preprocessor = get_preprocessing_pipeline(mode='train')
    val_preprocessor = get_preprocessing_pipeline(mode='eval')
    print("✓ Train preprocessor: resize=224x224, augmentation=True, normalize=ImageNet")
    print("✓ Val preprocessor: resize=224x224, augmentation=False, normalize=ImageNet\n")
    
    # 4. Create datasets
    print("Step 4: Creating datasets...")
    train_dataset = CustomImageDataset(
        images=train_images,
        labels=train_labels,
        transform=train_preprocessor
    )
    val_dataset = CustomImageDataset(
        images=val_images,
        labels=val_labels,
        transform=val_preprocessor
    )
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples\n")
    
    # 5. Create data loaders
    print("Step 5: Creating data loaders...")
    loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Use 0 for compatibility
        shuffle_train=True
    )
    print(f"✓ Train loader: batch_size={BATCH_SIZE}, shuffle=True")
    print(f"✓ Val loader: batch_size={BATCH_SIZE}, shuffle=False\n")
    
    # 6. Setup device
    print("Step 6: Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}\n")
    
    # 7. Create model
    print("Step 7: Creating model...")
    model = create_simple_model(NUM_CLASSES).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters\n")
    
    # 8. Setup training
    print("Step 8: Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"✓ Loss function: CrossEntropyLoss")
    print(f"✓ Optimizer: Adam (lr={LEARNING_RATE})\n")
    
    # 9. Training loop
    print("Step 9: Training model...")
    print("-" * 60)
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, loaders['val'], criterion, device
        )
        
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        print("-" * 60)
    
    # 10. Summary
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("  ✓ Standard preprocessing pipeline (224x224, ImageNet normalization)")
    print("  ✓ Data augmentation for training")
    print("  ✓ Data loaders with shuffle enabled for training")
    print("  ✓ GPU/CPU automatic detection and usage")
    print("  ✓ Complete training and validation loop")
    print("\nYou can now use these components for your own computer vision project!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
