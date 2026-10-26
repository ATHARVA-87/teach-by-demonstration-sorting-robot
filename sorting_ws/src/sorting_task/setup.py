from setuptools import find_packages, setup

package_name = "sorting_task"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["py.typed"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Atharva Thombare",
    maintainer_email="thombare.atharva16@gmail.com",
    description="Robot-independent sorting tasks.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["export_schema = sorting_task.task_file:export_schema_main"]},
)
