#!/usr/bin/env python3
"""
Local demonstration script for im2fit pipeline.
Tests the complete pipeline end-to-end by starting a local server and posting an image.
"""

import os
import sys
import time
import requests
import subprocess
import signal
import argparse
from pathlib import Path
from typing import Optional
import json


def start_server(port: int = 8000) -> subprocess.Popen:
    """Start the FastAPI server in the background"""
    print(f"Starting server on port {port}...")
    
    # Set required environment variables for local testing
    env = os.environ.copy()
    env.setdefault("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=demo;AccountKey=ZGVtbw==;EndpointSuffix=core.windows.net")
    env.setdefault("USE_ONNX", "1")
    env.setdefault("ONNX_MODEL_PATH", "model/best.onnx")
    
    # Start uvicorn server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app", 
        "--host", "127.0.0.1", 
        "--port", str(port),
        "--reload"
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=Path(__file__).parent
    )
    
    # Wait for server to start
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
            if response.status_code == 200:
                print(f"✅ Server started successfully on port {port}")
                return proc
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            raise RuntimeError("Server failed to start")
    
    raise RuntimeError(f"Server did not respond after {max_attempts} seconds")


def stop_server(proc: subprocess.Popen):
    """Stop the FastAPI server"""
    print("Stopping server...")
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("✅ Server stopped")


def test_pipeline(image_path: Path, server_url: str) -> dict:
    """Test the pipeline with an image file"""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"📤 Processing image: {image_path}")
    
    # Read image file
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/png')}
        
        # Post to /process endpoint
        response = requests.post(
            f"{server_url}/process",
            files=files,
            timeout=30
        )
    
    if response.status_code != 200:
        raise RuntimeError(f"Pipeline failed with status {response.status_code}: {response.text}")
    
    return response.json()


def create_sample_image(output_path: Path):
    """Create a sample test image with an ArUco marker"""
    try:
        import cv2
        import numpy as np
        
        # Create a 800x600 image
        img = np.ones((600, 800, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add some nail-like content in the center
        center_x, center_y = 400, 300
        cv2.ellipse(img, (center_x, center_y), (80, 160), 0, 0, 360, (180, 120, 100), -1)
        
        # Add ArUco marker if available
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 100)
            # Place marker in top-left corner
            img[50:150, 50:150] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            print("✅ Added ArUco marker for scaling")
        except AttributeError:
            print("⚠️  ArUco not available, using fallback scaling")
        
        # Add some text
        cv2.putText(img, "Demo Image", (center_x-80, center_y-200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        
        cv2.imwrite(str(output_path), img)
        print(f"✅ Created sample image: {output_path}")
        
    except ImportError:
        print("❌ OpenCV not available, cannot create sample image")
        raise


def main():
    parser = argparse.ArgumentParser(description="Demonstrate im2fit pipeline locally")
    parser.add_argument("image", nargs='?', help="Path to image file to process")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-server", action="store_true", help="Don't start server, assume it's running")
    parser.add_argument("--create-sample", action="store_true", help="Create a sample test image")
    args = parser.parse_args()
    
    server_url = f"http://127.0.0.1:{args.port}"
    server_proc = None
    
    try:
        # Handle sample image creation
        if args.create_sample:
            sample_path = Path("sample_nail.png")
            create_sample_image(sample_path)
            if not args.image:
                args.image = str(sample_path)
        
        # Require image path
        if not args.image:
            print("❌ No image provided. Use --create-sample to generate one, or provide a path.")
            return 1
        
        image_path = Path(args.image)
        
        # Start server if needed
        if not args.no_server:
            server_proc = start_server(args.port)
        else:
            # Check if server is running
            try:
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code != 200:
                    raise RuntimeError("Server health check failed")
                print(f"✅ Using existing server at {server_url}")
            except requests.exceptions.RequestException:
                print(f"❌ No server running at {server_url}")
                return 1
        
        # Test the pipeline
        print("\n" + "="*50)
        print("🚀 TESTING PIPELINE")
        print("="*50)
        
        result = test_pipeline(image_path, server_url)
        
        print("\n✅ PIPELINE SUCCESS!")
        print("📊 Results:")
        print(f"  Backend: {result.get('backend', 'unknown')}")
        print(f"  Scale confidence: {result.get('scale_confidence', 'N/A')}")
        
        metrics = result.get('metrics', {})
        if metrics:
            print("📏 Measurements:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'mm' in key:
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n🔗 Generated artifacts:")
        for key in ['overlay_url', 'csv_url', 'stl_url']:
            if key in result:
                print(f"  {key.replace('_', ' ').title()}: {result[key]}")
        
        print(f"\n📋 Full response saved to: demo_result.json")
        with open("demo_result.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
        
    finally:
        if server_proc:
            stop_server(server_proc)


if __name__ == "__main__":
    sys.exit(main())