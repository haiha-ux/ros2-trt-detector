from setuptools import find_packages, setup

package_name = "trt_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/detector.launch.py"]),
        ("share/" + package_name + "/config", ["config/detector.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ha Phan Thanh Hai",
    maintainer_email="haiha-ux@users.noreply.github.com",
    description="TensorRT YOLO detector node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector_node = trt_detector.detector_node:main",
            "video_publisher = trt_detector.video_publisher:main",
        ],
    },
)
