from glob import glob

from setuptools import find_packages, setup

package_name = "vla_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Installed, not just built: `ros2 launch vla_bridge x.launch.py` reads them from
        # share/, so a launch file that is only in the source tree is invisible.
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    # pyzmq and msgpack are not packaged as ROS deps; install them into the container
    # with pip. numpy ships with the Jazzy image.
    install_requires=["setuptools", "pyzmq", "msgpack"],
    zip_safe=True,
    maintainer="Trish",
    maintainer_email="trishlee3032@gmail.com",
    description="ROS2 bridge to a remote VLA policy server (GR00T N1.6 / DiffusionVLA).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vla_policy_node = vla_bridge.vla_policy_node:main",
            "droid_slam_node = vla_bridge.slam_node:main",
        ],
    },
)
