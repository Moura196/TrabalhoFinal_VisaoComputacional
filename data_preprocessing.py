"""
Data preprocessing pipeline for computer vision tasks.
This module provides standardized preprocessing for image data including:
- Resize to 224x224
- Normalization by channel (ImageNet statistics)
- Data loaders with shuffle for training
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Optional, Tuple, Callable
import numpy as np
from PIL import Image


# ImageNet normalization statistics (commonly used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StandardImagePreprocessor:
    """
    Standard image preprocessing pipeline for computer vision tasks.
    
    Applies:
    - Resize to 224x224
    - Convert to tensor
    - Normalize using ImageNet statistics
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: list = None,
        std: list = None,
        augmentation: bool = False
    ):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target image size (default: 224)
            mean: Normalization mean values (default: ImageNet mean)
            std: Normalization std values (default: ImageNet std)
            augmentation: Whether to apply data augmentation (default: False)
        """
        self.image_size = image_size
        self.mean = mean if mean is not None else IMAGENET_MEAN
        self.std = std if std is not None else IMAGENET_STD
        self.augmentation = augmentation
        
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        """
        Build the transformation pipeline.
        
        Returns:
            Composed transforms
        """
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Data augmentation (only for training)
        if self.augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
            ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        return transforms.Compose(transform_list)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply preprocessing to an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        return self.transform(image)
    
    def get_transform(self) -> transforms.Compose:
        """
        Get the transform pipeline.
        
        Returns:
            Transform pipeline
        """
        return self.transform


class CustomImageDataset(Dataset):
    """
    Custom dataset for image data.
    
    This is a template that can be adapted to specific datasets.
    """
    
    def __init__(
        self,
        images: list,
        labels: Optional[list] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            images: List of image paths or image arrays
            labels: Optional list of labels
            transform: Optional transform to apply to images
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image tensor, label)
        """
        # Load image
        image = self.images[idx]
        
        # Convert to PIL Image if necessary
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get label if available
        label = self.labels[idx] if self.labels is not None else -1
        
        return image, label


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    shuffle_test: bool = False
) -> dict:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        shuffle_train: Whether to shuffle training data (default: True)
        shuffle_val: Whether to shuffle validation data (default: False)
        shuffle_test: Whether to shuffle test data (default: False)
        
    Returns:
        Dictionary containing data loaders
    """
    loaders = {}
    
    # Training loader with shuffle
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    # Validation loader
    if val_dataset is not None:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    # Test loader
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    return loaders


def get_preprocessing_pipeline(
    mode: str = 'train',
    image_size: int = 224,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> StandardImagePreprocessor:
    """
    Get a standard preprocessing pipeline.
    
    Args:
        mode: 'train' or 'eval' - determines if augmentation is applied
        image_size: Target image size (default: 224)
        mean: Normalization mean (default: ImageNet mean)
        std: Normalization std (default: ImageNet std)
        
    Returns:
        StandardImagePreprocessor instance
    """
    augmentation = (mode == 'train')
    
    return StandardImagePreprocessor(
        image_size=image_size,
        mean=mean,
        std=std,
        augmentation=augmentation
    )


def denormalize_image(
    tensor: torch.Tensor,
    mean: list = None,
    std: list = None
) -> np.ndarray:
    """
    Denormalize an image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean values used for normalization (default: ImageNet mean)
        std: Std values used for normalization (default: ImageNet std)
        
    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    mean = mean if mean is not None else IMAGENET_MEAN
    std = std if std is not None else IMAGENET_STD
    
    # Clone tensor to avoid modifying original
    tensor = tensor.clone()
    
    # Denormalize
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clip values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose to (H, W, C)
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    
    return image


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE - EXAMPLE USAGE")
    print("=" * 60)
    print()
    
    # Create preprocessors for train and eval
    train_preprocessor = get_preprocessing_pipeline(mode='train')
    eval_preprocessor = get_preprocessing_pipeline(mode='eval')
    
    print("✓ Created training preprocessor (with augmentation)")
    print("✓ Created evaluation preprocessor (without augmentation)")
    print()
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (256, 256), color='red')
    
    # Apply preprocessing
    train_tensor = train_preprocessor(dummy_image)
    eval_tensor = eval_preprocessor(dummy_image)
    
    print(f"Input image size: {dummy_image.size}")
    print(f"Preprocessed tensor shape: {train_tensor.shape}")
    print(f"Tensor dtype: {train_tensor.dtype}")
    print(f"Tensor range: [{train_tensor.min().item():.3f}, {train_tensor.max().item():.3f}]")
    print()
    
    # Create dummy dataset
    dummy_images = [Image.new('RGB', (256, 256), color='blue') for _ in range(10)]
    dummy_labels = list(range(10))
    
    train_dataset = CustomImageDataset(
        images=dummy_images,
        labels=dummy_labels,
        transform=train_preprocessor
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print()
    
    # Create data loader
    loaders = create_data_loaders(
        train_dataset=train_dataset,
        batch_size=4,
        num_workers=0,  # Use 0 for this example
        shuffle_train=True
    )
    
    print("✓ Created data loader with shuffle enabled")
    print(f"Batch size: 4")
    print()
    
    # Test data loader
    for batch_idx, (images, labels) in enumerate(loaders['train']):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        if batch_idx == 0:  # Only show first batch
            break
    
    print()
    print("=" * 60)
    print("✓ Data preprocessing pipeline is working correctly!")
    print("=" * 60)
