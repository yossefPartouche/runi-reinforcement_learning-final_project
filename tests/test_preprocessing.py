import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.preprocessing.image_preprocessing import (
    preprocess_observation,
    preprocess_observation_basic,
    preprocess_observation_normalize
)


def test_preprocessing_output_shape():
    """Test that preprocessing produces correct output shape."""
    print("\n" + "="*60)
    print("Testing preprocessing output shape...")
    print("="*60)
    
    # Create synthetic RGB image
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    print(f"Input:  shape={test_rgb.shape}, dtype={test_rgb.dtype}")
    
    # Apply preprocessing
    processed = preprocess_observation(test_rgb)
    print(f"Output: shape={processed.shape}, dtype={processed.dtype}")
    
    # Verify shape
    assert processed.shape == (84, 84, 1), \
        f"Expected shape (84, 84, 1), got {processed.shape}"
    print("✅ Output shape is correct: (84, 84, 1)")


def test_preprocessing_output_dtype():
    """Test that preprocessing produces correct dtype."""
    print("\n" + "="*60)
    print("Testing preprocessing output dtype...")
    print("="*60)
    
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    processed = preprocess_observation(test_rgb)
    
    assert processed.dtype == np.float32, \
        f"Expected dtype float32, got {processed.dtype}"
    print(f"✅ Output dtype is correct: {processed.dtype}")


def test_preprocessing_output_range():
    """Test that preprocessing normalizes to [0, 1]."""
    print("\n" + "="*60)
    print("Testing preprocessing output range...")
    print("="*60)
    
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    processed = preprocess_observation(test_rgb)
    
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    assert 0.0 <= processed.min() <= processed.max() <= 1.0, \
        f"Expected range [0, 1], got [{processed.min()}, {processed.max()}]"
    print("✅ Output range is correct: [0.0, 1.0]")


def test_preprocessing_grayscale_conversion():
    """Test that RGB is properly converted to grayscale."""
    print("\n" + "="*60)
    print("Testing grayscale conversion...")
    print("="*60)
    
    # Create RGB image with known values
    test_rgb = np.ones((320, 320, 3), dtype=np.uint8) * 128
    processed = preprocess_observation(test_rgb)
    
    # Check that output has single channel
    assert processed.shape[2] == 1, \
        f"Expected 1 channel, got {processed.shape[2]}"
    print("✅ Grayscale conversion successful (1 channel)")


def test_memory_reduction():
    """Test memory savings from preprocessing."""
    print("\n" + "="*60)
    print("Testing memory reduction...")
    print("="*60)
    
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    processed = preprocess_observation(test_rgb)
    
    memory_original = test_rgb.nbytes / 1024  # KB
    memory_processed = processed.nbytes / 1024  # KB
    reduction = (1 - memory_processed / memory_original) * 100
    
    print(f"Original RGB (320x320x3): {memory_original:.2f} KB")
    print(f"Processed (84x84x1):      {memory_processed:.2f} KB")
    print(f"Memory reduction:         {reduction:.1f}%")
    
    assert memory_processed < memory_original, \
        "Preprocessing should reduce memory usage"
    print(f"✅ Memory reduced by {reduction:.1f}%")


def test_basic_preprocessing():
    """Test basic preprocessing (grayscale only, no resize)."""
    print("\n" + "="*60)
    print("Testing basic preprocessing...")
    print("="*60)
    
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    basic = preprocess_observation_basic(test_rgb)
    
    print(f"Output: shape={basic.shape}, dtype={basic.dtype}")
    
    assert basic.shape == (320, 320, 1), f"Expected (320, 320, 1), got {basic.shape}"
    assert basic.dtype == np.uint8, f"Expected uint8, got {basic.dtype}"
    print("✅ Basic preprocessing works correctly")


def test_normalized_preprocessing():
    """Test normalized preprocessing (grayscale + normalize, no resize)."""
    print("\n" + "="*60)
    print("Testing normalized preprocessing...")
    print("="*60)
    
    test_rgb = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    normalized = preprocess_observation_normalize(test_rgb)
    
    print(f"Output: shape={normalized.shape}, dtype={normalized.dtype}")
    print(f"Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    assert normalized.shape == (320, 320, 1), f"Expected (320, 320, 1), got {normalized.shape}"
    assert normalized.dtype == np.float32, f"Expected float32, got {normalized.dtype}"
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0, "Range should be [0, 1]"
    print("✅ Normalized preprocessing works correctly")


def run_all_tests():
    """Run all preprocessing tests."""
    print("\n" + "🧪 "*30)
    print("PREPROCESSING TESTS")
    print("🧪 "*30)
    
    try:
        test_preprocessing_output_shape()
        test_preprocessing_output_dtype()
        test_preprocessing_output_range()
        test_preprocessing_grayscale_conversion()
        test_memory_reduction()
        test_basic_preprocessing()
        test_normalized_preprocessing()
        
        print("\n" + "="*60)
        print("🎉 ALL PREPROCESSING TESTS PASSED! 🎉")
        print("="*60)
        return True
        
    except AssertionError as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()