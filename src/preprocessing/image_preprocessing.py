import numpy as np
import cv2

def preprocess_observation(observation: np.ndarray, target_size: tuple = (84, 84)) -> np.ndarray:
    """
    Preprocesses an RGB observation for RL agents:
        1. Converts RGB to grayscale (H, W)
        2. Resizes to target_size (default: 84x84)
        3. Normalizes to [0, 1] float32
        4. Converts to PyTorch format (C, H, W)
    
    Args:
        observation: Raw RGB image (H, W, 3), dtype uint8, range [0, 255]
        target_size: (height, width) for resizing
    
    Returns:
        np.ndarray: Preprocessed image (1, H, W), float32, range [0, 1]
    """
    assert observation.ndim == 3 and observation.shape[2] == 3, \
        f"Expected RGB image (H, W, 3), got {observation.shape}"
    assert observation.dtype == np.uint8, \
        f"Expected uint8 dtype, got {observation.dtype}"

    # 1. Grayscale
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)  # (H, W)
    # 2. Resize
    resized = cv2.resize(gray, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)  # (H, W)
    # 3. Normalize
    normalized = resized.astype(np.float32) / 255.0  # (H, W)
    # 4. Add channel and convert to PyTorch format
    pytorch_img = np.expand_dims(normalized, axis=0)  # (1, H, W)
    return pytorch_img

def preprocess_observation_basic(observation: np.ndarray) -> np.ndarray:
    """
    Basic preprocessing: Grayscale only, keeps original size, returns (1, H, W).
    """
    assert observation.ndim == 3 and observation.shape[2] == 3, \
        f"Expected RGB image (H, W, 3), got {observation.shape}"
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    pytorch_img = np.expand_dims(gray, axis=0)  # (1, H, W)
    return pytorch_img

def preprocess_observation_normalize(observation: np.ndarray) -> np.ndarray:
    """
    Grayscale + Normalize (no resize), returns (1, H, W), float32.
    """
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    pytorch_img = np.expand_dims(normalized, axis=0)  # (1, H, W)
    return pytorch_img

# For testing
if __name__ == "__main__":
    print("Testing preprocessing functions...\n")
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    print(f"Input: shape={test_rgb.shape}, dtype={test_rgb.dtype}, "
          f"range=[{test_rgb.min()}, {test_rgb.max()}]")

    print("\n--- Testing Combined Preprocessing ---")
    processed = preprocess_observation(test_rgb, target_size=(84, 84))
    print(f"Output: shape={processed.shape}, dtype={processed.dtype}, "
          f"range=[{processed.min():.3f}, {processed.max():.3f}]")
    assert processed.shape == (1, 84, 84), "Shape mismatch!"
    assert processed.dtype == np.float32, "Dtype should be float32!"
    assert 0.0 <= processed.min() <= processed.max() <= 1.0, "Range should be [0, 1]!"
    print("✅ Combined preprocessing works!")

    print("\n--- Testing Basic Preprocessing ---")
    basic = preprocess_observation_basic(test_rgb)
    print(f"Output: shape={basic.shape}, dtype={basic.dtype}, "
          f"range=[{basic.min()}, {basic.max()}]")
    assert basic.shape == (1, 320, 320), "Shape mismatch!"
    assert basic.dtype == np.uint8, "Dtype should be uint8!"
    print("✅ Basic preprocessing works!")

    print("\n--- Testing Normalized Preprocessing ---")
    normalized = preprocess_observation_normalize(test_rgb)
    print(f"Output: shape={normalized.shape}, dtype={normalized.dtype}, "
          f"range=[{normalized.min():.3f}, {normalized.max():.3f}]")
    assert normalized.shape == (1, 320, 320), "Shape mismatch!"
    assert normalized.dtype == np.float32, "Dtype should be float32!"
    print("✅ Normalized preprocessing works!")

    print("\n--- Memory Comparison ---")
    memory_original = test_rgb.nbytes / 1024
    memory_basic = basic.nbytes / 1024
    memory_resized = processed.nbytes / 1024
    print(f"Original RGB (320x320x3):    {memory_original:.2f} KB")
    print(f"Basic grayscale (1,320,320): {memory_basic:.2f} KB ({memory_basic/memory_original*100:.1f}%)")
    print(f"Resized (1,84,84):           {memory_resized:.2f} KB ({memory_resized/memory_original*100:.1f}%)")
    print(f"💾 Memory savings: {(1 - memory_resized/memory_original)*100:.1f}%")
    print("\n🎉 All preprocessing tests passed!")