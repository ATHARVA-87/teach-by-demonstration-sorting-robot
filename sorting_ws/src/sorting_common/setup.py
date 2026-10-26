from glob import glob

from setuptools import find_packages, setup

package_name = "sorting_common"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["py.typed"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Barlow fonts (SIL OFL), shared by the AR overlay and the dashboard.
        ("share/" + package_name + "/assets/fonts", glob("assets/fonts/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Atharva Thombare",
    maintainer_email="thombare.atharva16@gmail.com",
    description="Shared helpers for the sorting cell.",
    license="Apache-2.0",
)
