# Copyright 2020 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup for pip package."""
from setuptools import setup

REQUIRED_PACKAGES = (
    "absl-py>=0.12.0",
    "numpy>=1.16.4",
    "typing>=3.7.4.3",
    "scipy>=1.7.1",
    "open-spiel>=1.0.1",
    "tensorflow>=2.6.0",
    "tensorflow-datasets>=4.4.0",
    "jax==0.2.20",
    "jaxlib==0.1.71"
)

LONG_DESCRIPTION = "\n".join([(
    "A suite of 17 datasets with phase space, high dimensional (visual) "
    "observations and other measurement where appropriate that are based on "
    "physical systems, exhibiting a Hamiltonian dynamics"
)])


setup(
    name="dm_hamiltonian_dynamics_suite",
    version="0.0.1",
    description="A collection of 17 datasets based on Hamiltonian physical "
                "systems.",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/deepmind/dm_hamiltonian_dynamics_suite",
    author="DeepMind",
    packages=[
        "dm_hamiltonian_dynamics_suite",
        "dm_hamiltonian_dynamics_suite.hamiltonian_systems",
        "dm_hamiltonian_dynamics_suite.molecular_dynamics",
        "dm_hamiltonian_dynamics_suite.multiagent_dynamics",
    ],
    install_requires=REQUIRED_PACKAGES,
    platforms=["any"],
    license="Apache License, Version 2.0",
)
