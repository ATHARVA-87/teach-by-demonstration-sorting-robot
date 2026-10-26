from glob import glob

from setuptools import find_packages, setup

package_name = "sorting_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["py.typed"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Atharva Thombare",
    maintainer_email="thombare.atharva16@gmail.com",
    description="MuJoCo simulated sorting cell behind ros2_control-compatible interfaces.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_node = sorting_sim.sim_node:main",
        ]
    },
)
