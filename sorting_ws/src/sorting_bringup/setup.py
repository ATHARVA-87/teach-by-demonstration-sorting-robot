from glob import glob

from setuptools import find_packages, setup

package_name = "sorting_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Atharva Thombare",
    maintainer_email="thombare.atharva16@gmail.com",
    description="Launch files for the sorting cell.",
    license="Apache-2.0",
)
