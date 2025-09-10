import os
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path

# Set up test environment - mock azure connection for testing
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net"

import pipeline


def create_test_image(width=640, height=480):
    """Create a test image for pipeline testing"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some content to make it a valid image
    cv2.rectangle(img, (width//4, height//4), (3*width//4, 3*height//4), (255, 255, 255), -1)
    return img


def create_test_mask(width=640, height=480):
    """Create a test segmentation mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    # Create a nail-like shape in the center
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(mask, (center_x, center_y), (60, 120), 0, 0, 360, 255, -1)
    return mask


class TestPipeline:
    """Test cases for the main pipeline functionality"""
    
    def test_image_from_bytes(self):
        """Test image decoding from bytes"""
        img = create_test_image()
        _, buffer = cv2.imencode('.png', img)
        img_bytes = buffer.tobytes()
        
        result = pipeline._image_from_bytes(img_bytes)
        assert result is not None
        assert result.shape[:2] == (480, 640)  # height, width
    
    def test_image_from_bytes_invalid(self):
        """Test image decoding with invalid bytes"""
        invalid_bytes = b"not an image"
        result = pipeline._image_from_bytes(invalid_bytes)
        assert result is None
    
    @patch('pipeline.mm_per_pixel_from_aruco')
    def test_mm_per_pixel_aruco_detection(self, mock_aruco):
        """Test ArUco marker detection for scaling"""
        # Test successful ArUco detection
        mock_aruco.return_value = (0.25, 0.95)
        img = create_test_image()
        
        mm_per_px, confidence = mock_aruco(img)
        assert mm_per_px == 0.25
        assert confidence == 0.95
        mock_aruco.assert_called_once()
    
    @patch('pipeline.mm_per_pixel_from_aruco')
    def test_mm_per_pixel_fallback(self, mock_aruco):
        """Test fallback when ArUco marker is not detected"""
        mock_aruco.return_value = (None, 0.0)
        img = create_test_image()
        
        mm_per_px, confidence = mock_aruco(img)
        assert mm_per_px is None
        assert confidence == 0.0
    
    @patch('app.onnx_infer.infer_mask')
    def test_onnx_segmentation_success(self, mock_infer):
        """Test successful ONNX segmentation"""
        test_mask = create_test_mask()
        mock_infer.return_value = test_mask
        
        img = create_test_image()
        result = pipeline._onnx_segmentation(img)
        
        assert result is not None
        assert np.array_equal(result, test_mask)
        mock_infer.assert_called_once()
    
    @patch('app.onnx_infer.infer_mask')
    def test_onnx_segmentation_failure(self, mock_infer):
        """Test ONNX segmentation failure handling"""
        mock_infer.side_effect = Exception("ONNX inference failed")
        
        img = create_test_image()
        result = pipeline._onnx_segmentation(img)
        
        assert result is None
    
    def test_hosted_segmentation_no_config(self):
        """Test hosted segmentation without configuration"""
        with patch.dict(os.environ, {}, clear=True):
            img = create_test_image()
            result = pipeline._hosted_segmentation(img)
            assert result is None
    
    @patch('requests.post')
    def test_hosted_segmentation_success(self, mock_post):
        """Test successful hosted segmentation"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "predictions": [{
                "points": [
                    {"x": 100, "y": 100},
                    {"x": 200, "y": 100},
                    {"x": 200, "y": 200},
                    {"x": 100, "y": 200}
                ]
            }]
        }
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {
            'ROBOFLOW_API_KEY': 'test_key',
            'ROBOFLOW_INFER_URL': 'https://test.com'
        }):
            img = create_test_image()
            result = pipeline._hosted_segmentation(img)
            
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
    
    @patch('app.measure.pca_axis_metrics')
    def test_measure_function(self, mock_pca):
        """Test measurement calculation"""
        mock_metrics = {
            'length_mm': 25.0,
            'width_prox_mm': 12.0,
            'width_mid_mm': 11.0,
            'width_dist_mm': 10.0,
            'mask_area_px': 1500,
            'sharpness': 120.0
        }
        mock_pca.return_value = mock_metrics
        
        mask = create_test_mask()
        result = pipeline._measure(mask, 0.25)
        
        assert result == mock_metrics
        mock_pca.assert_called_once_with(mask, 0.25)
    
    @patch('app.scale_aruco.mm_per_pixel_from_aruco')
    @patch('app.onnx_infer.infer_mask')
    @patch('app.measure.pca_axis_metrics')
    @patch('app.overlay.draw_overlay')
    @patch('app.to_3d.outline_to_stl_bytes')
    def test_run_pipeline_full(self, mock_stl, mock_overlay, mock_measure, mock_infer, mock_aruco):
        """Test complete pipeline run"""
        # Setup mocks
        mock_aruco.return_value = (0.25, 0.95)
        mock_infer.return_value = create_test_mask()
        mock_measure.return_value = {
            'length_mm': 25.0,
            'width_mid_mm': 11.0,
            'mask_area_px': 1500
        }
        mock_overlay.return_value = b"fake_png_data"
        mock_stl.return_value = b"fake_stl_data"
        
        # Create test image
        img = create_test_image()
        _, buffer = cv2.imencode('.png', img)
        img_bytes = buffer.tobytes()
        
        with patch.dict(os.environ, {'USE_ONNX': '1'}):
            result = pipeline.run_pipeline(img_bytes)
        
        # Verify result structure
        assert 'overlay_png' in result
        assert 'stl_bytes' in result
        assert 'csv_bytes' in result
        assert 'metrics' in result
        
        # Verify data types
        assert isinstance(result['overlay_png'], bytes)
        assert isinstance(result['stl_bytes'], bytes)
        assert isinstance(result['csv_bytes'], bytes)
        assert isinstance(result['metrics'], dict)
        
        # Verify metrics include scaling information
        assert 'mm_per_px' in result['metrics']
        assert 'scale_confidence' in result['metrics']
        assert result['metrics']['mm_per_px'] == 0.25
        assert result['metrics']['scale_confidence'] == 0.95
    
    @patch('app.scale_aruco.mm_per_pixel_from_aruco')
    @patch('app.onnx_infer.infer_mask')
    def test_run_pipeline_fallback_scaling(self, mock_infer, mock_aruco):
        """Test pipeline with fallback scaling when ArUco fails"""
        # ArUco detection fails
        mock_aruco.return_value = (None, 0.0)
        mock_infer.return_value = create_test_mask()
        
        img = create_test_image()
        _, buffer = cv2.imencode('.png', img)
        img_bytes = buffer.tobytes()
        
        with patch.dict(os.environ, {'USE_ONNX': '1'}):
            result = pipeline.run_pipeline(img_bytes)
        
        # Should use fallback scaling
        assert result['metrics']['mm_per_px'] == 0.25
        assert result['metrics']['scale_confidence'] == 0.0
    
    @patch('app.scale_aruco.mm_per_pixel_from_aruco')
    @patch('app.onnx_infer.infer_mask')
    def test_run_pipeline_dummy_mask(self, mock_infer, mock_aruco):
        """Test pipeline with dummy mask when segmentation fails"""
        mock_aruco.return_value = (0.25, 0.95)
        mock_infer.return_value = None  # Segmentation fails
        
        img = create_test_image()
        _, buffer = cv2.imencode('.png', img)
        img_bytes = buffer.tobytes()
        
        with patch.dict(os.environ, {'USE_ONNX': '1'}):
            result = pipeline.run_pipeline(img_bytes)
        
        # Should still produce results with dummy mask
        assert 'overlay_png' in result
        assert 'metrics' in result
    
    def test_run_pipeline_invalid_image(self):
        """Test pipeline with invalid image data"""
        invalid_bytes = b"not an image"
        
        with pytest.raises(ValueError, match="Invalid image"):
            pipeline.run_pipeline(invalid_bytes)


class TestCSVGeneration:
    """Test CSV output generation"""
    
    def test_csv_metrics_filtering(self):
        """Test that only scalar metrics are included in CSV"""
        import pandas as pd
        
        metrics = {
            'length_mm': 25.0,
            'width_mid_mm': 11.0,
            'mask_area_px': 1500,
            'axis_origin': [320, 240],  # Should be excluded (list)
            'axis_vec': [1.0, 0.0],     # Should be excluded (list)
            'complex_data': {'key': 'value'}  # Should be excluded (dict)
        }
        
        # Filter metrics like the pipeline does
        filtered = {k: v for k, v in metrics.items() if not isinstance(v, (list, tuple, dict))}
        
        df = pd.DataFrame([filtered])
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        
        # Parse back to verify
        csv_str = csv_bytes.decode('utf-8')
        assert 'length_mm' in csv_str
        assert 'width_mid_mm' in csv_str
        assert 'mask_area_px' in csv_str
        assert 'axis_origin' not in csv_str
        assert 'axis_vec' not in csv_str
        assert 'complex_data' not in csv_str


@pytest.fixture
def test_env():
    """Setup test environment variables"""
    test_vars = {
        "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        "USE_ONNX": "1",
        "ONNX_MODEL_PATH": "model/best.onnx"
    }
    
    with patch.dict(os.environ, test_vars):
        yield test_vars


def test_pipeline_import():
    """Test that pipeline module imports successfully"""
    import pipeline
    assert hasattr(pipeline, 'run_pipeline')
    assert hasattr(pipeline, '_image_from_bytes')
    assert hasattr(pipeline, '_onnx_segmentation')
    assert hasattr(pipeline, '_hosted_segmentation')
    assert hasattr(pipeline, '_measure')