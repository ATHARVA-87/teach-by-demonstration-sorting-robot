from glob import glob

from setuptools import find_packages, setup

package_name = "rule_learner"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["py.typed"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Atharva Thombare",
    maintainer_email="thombare.atharva16@gmail.com",
    description="Rule learning from demonstrations.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rule_learner_node = rule_learner.node:main",
            "benchmark = rule_learner.benchmark:main",
            "plot_results = rule_learner.plots:main",
        ]
    },
)
