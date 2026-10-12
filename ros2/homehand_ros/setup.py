from glob import glob

from setuptools import setup

setup(
    name="homehand_ros",
    version="0.1.0",
    packages=["homehand_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/homehand_ros"]),
        ("share/homehand_ros", ["package.xml"]),
        ("share/homehand_ros/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "sim_bridge = homehand_ros.sim_bridge:main",
        "tidy_controller = homehand_ros.tidy_controller:main",
        "inspire_bridge = homehand_ros.inspire_bridge:main",
    ]},
)
