Welcome to the **im2fit** project. This repository contains the source code for a modular and scalable computer vision and 3D modeling pipeline that transforms a simple 2D image into a custom-fit 3D model. While this project focuses on the beauty-tech industry, its core architecture is designed for a wide range of custom product manufacturing and design applications.

---

## Business Value & Use Cases

This project demonstrates a robust, end-to-end solution for a significant business challenge: translating real-world, physical dimensions into digital, customizable products.

### Value Proposition

The pipeline provides:

- **Precision and Customization**: Replaces manual sizing with an automated, accurate computer vision system, ensuring a perfect fit for every customer.
- **Scalability**: The system can process a high volume of images at a low cost, enabling mass production of personalized items.
- **Efficiency**: The fully automated pipeline drastically reduces design time, allowing for faster product delivery and a more responsive supply chain.
- **Enhanced User Experience**: Offers a seamless, interactive experience where users can see a digital preview of their custom product before it's created.

### Broader Applications

The underlying technology is broadly applicable to any industry that requires the conversion of 2D images to custom 3D products, such as:

- **Manufacturing**: Detecting and measuring product defects.
- **Medical Devices**: Creating custom prosthetics or orthotics.
- **Dental**: Designing custom crowns or aligners.
- **Footwear**: Generating personalized shoe insoles.

---

## Technical Stack & Capabilities

| Component                | Description                                                                                                                           |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Web Framework**        | **FastAPI** on **Python** with **Gunicorn** and **Uvicorn** for a high-performance, developer-friendly backend.                       |
| **Image Processing**     | **OpenCV** & **Numpy** for industry-standard computer vision tasks, including accurate real-world scaling via ArUco marker detection. |
| **Segmentation**         | A **YOLOv11** segmentation model exported to the `.onnx` format for fast, portable inference on CPU or GPU.                           |
| **3D Modeling**          | **Trimesh** & **Shapely** to reliably convert 2D shapes into solid 3D meshes for manufacturing.                                       |
| **Cloud Infrastructure** | **Azure App Service** and **Azure Blob Storage** for a scalable, cost-effective, and robust deployment.                               |
| **DevOps**               | **Docker**, **GitHub Actions**, and **Azure Bicep** for automated, reproducible, and secure deployments.                              |

### Project Capabilities

This repository showcases:

- **End-to-end pipeline implementation**: From image upload to final 3D artifact generation.
- **Expertise in image segmentation**: The core functionality of the pipeline relies on a custom-trained model for accurate finger/nail segmentation.
- **Robust geometry inference**: The system normalizes input images to derive precise, real-world dimensions in millimeters.
- **Proficiency with frameworks**: The project demonstrates hands-on experience with Python, OpenCV, PyTorch/TensorFlow (via ONNX), and 3D modeling libraries.

<br>
