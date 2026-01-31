import numpy as np
import cv2

def preprocess_observation(observation: np.ndarray, target_size: tuple = (84, 84)) -> np.ndarray:
    """
    Combined preprocessing: Grayscale + Normalize + Resize
    
    Pipeline:
        1. Convert RGB to grayscale (3 channels -> 1 channel)
        2. Resize to smaller dimensions (reduces computation)
        3. Normalize to [0, 1] range (training stability)
        4. Convert to float32
    
    Args:
        observation: Raw RGB image from environment
                    Shape: (320, 320, 3)
                    dtype: uint8
                    Range: [0, 255]
        target_size: Target dimensions (height, width)
                    Default: (84, 84) - standard from Atari DQN paper
    
    Returns:
        Preprocessed image
        Shape: (84, 84, 1)
        Range: [0.0, 1.0]
    
    Note:
        Formula for grayscale: Gray = 0.299*R + 0.587*G + 0.114*B
    """
    # Validate input
    assert observation.ndim == 3, \
        f"Expected 3D array (H, W, C), got shape {observation.shape}"
    assert observation.shape[2] == 3, \
        f"Expected RGB (3 channels), got {observation.shape[2]} channels"
    assert observation.dtype == np.uint8, \
        f"Expected uint8 dtype, got {observation.dtype}"
    
    
    grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    # cv2.resize expects (width, height), so swap target_size
    resized = cv2.resize(grayscale, (target_size[1], target_size[0]), 
                        interpolation=cv2.INTER_AREA)  # INTER_AREA best for downsampling
    
    # Add channel dimension back, Shape: (84, 84, 1)
    resized = np.expand_dims(resized, axis=-1)
    
    #  Normalize to [0, 1] and convert to float32
    normalized = resized.astype(np.float32) / 255.0
    # Range: [0.0, 1.0]
    
    return normalized


def preprocess_observation_basic(observation: np.ndarray) -> np.ndarray:
    """
    Basic preprocessing (your original): Grayscale only, no resize, no normalize
    
    Use this if you want to keep 320x320 resolution.
    """
    assert observation.ndim == 3, \
        f"Expected 3D array (H, W, C), got shape {observation.shape}"
    assert observation.shape[2] == 3, \
        f"Expected RGB (3 channels), got {observation.shape[2]} channels"
    assert observation.dtype == np.uint8, \
        f"Expected uint8 dtype, got {observation.dtype}"
    
    grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    grayscale = np.expand_dims(grayscale, axis=-1)
    
    return grayscale


def preprocess_observation_normalize(observation: np.ndarray) -> np.ndarray:
    """
    Grayscale + Normalize (no resize): 320x320 resolution
    
    Use this if you want normalization but don't want to resize.
    """
    grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    grayscale = np.expand_dims(grayscale, axis=-1)
    
    # Normalize: [0, 255] -> float32 [0, 1]
    grayscale = grayscale.astype(np.float32) / 255.0
    
    return grayscale


# For testing
if __name__ == "__main__":
    print("Testing preprocessing functions...\n")
    
    # Create synthetic RGB image (320x320x3)
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    print(f"Input: shape={test_rgb.shape}, dtype={test_rgb.dtype}, "
          f"range=[{test_rgb.min()}, {test_rgb.max()}]")
    
    # Test combined preprocessing (grayscale + normalize + resize)
    print("\n--- Testing Combined Preprocessing ---")
    processed = preprocess_observation(test_rgb, target_size=(84, 84))
    print(f"Output: shape={processed.shape}, dtype={processed.dtype}, "
          f"range=[{processed.min():.3f}, {processed.max():.3f}]")
    assert processed.shape == (84, 84, 1), "Shape mismatch!"
    assert processed.dtype == np.float32, "Dtype should be float32!"
    assert 0.0 <= processed.min() <= processed.max() <= 1.0, "Range should be [0, 1]!"
    print("✅ Combined preprocessing works!")
    
    # Test basic preprocessing
    print("\n--- Testing Basic Preprocessing ---")
    basic = preprocess_observation_basic(test_rgb)
    print(f"Output: shape={basic.shape}, dtype={basic.dtype}, "
          f"range=[{basic.min()}, {basic.max()}]")
    assert basic.shape == (320, 320, 1), "Shape mismatch!"
    assert basic.dtype == np.uint8, "Dtype should be uint8!"
    print("✅ Basic preprocessing works!")
    
    # Test normalized preprocessing
    print("\n--- Testing Normalized Preprocessing ---")
    normalized = preprocess_observation_normalize(test_rgb)
    print(f"Output: shape={normalized.shape}, dtype={normalized.dtype}, "
          f"range=[{normalized.min():.3f}, {normalized.max():.3f}]")
    assert normalized.shape == (320, 320, 1), "Shape mismatch!"
    assert normalized.dtype == np.float32, "Dtype should be float32!"
    print("✅ Normalized preprocessing works!")
    
    # Compare memory usage
    print("\n--- Memory Comparison ---")
    memory_original = test_rgb.nbytes / 1024  # KB
    memory_basic = basic.nbytes / 1024
    memory_resized = processed.nbytes / 1024
    
    print(f"Original RGB (320x320x3):    {memory_original:.2f} KB")
    print(f"Basic grayscale (320x320x1): {memory_basic:.2f} KB ({memory_basic/memory_original*100:.1f}%)")
    print(f"Resized (84x84x1):           {memory_resized:.2f} KB ({memory_resized/memory_original*100:.1f}%)")
    print(f"💾 Memory savings: {(1 - memory_resized/memory_original)*100:.1f}%")
    
    print("\n🎉 All preprocessing tests passed!")