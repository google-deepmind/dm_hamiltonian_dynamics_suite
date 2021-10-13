# DeepMind Hamiltonian Dynamics Suite
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/dm_hamiltonian_dynamics_suite/blob/master/visualize_datasets.ipynb)

The repository contains the code used for generating the
[DM Hamiltonian Dynamics Suite].

The code for the models and experiments in our paper can be found 
[here](https://github.com/deepmind/deepmind-research/tree/master/physics_inspired_models), 
together with the code used for our concurrent publication on how to measure the 
quality of the learnt dynamics in models using the Hamiltonian prior when 
learning from pixels.


## Datasets

The suite contains 17 datasets ranging from simple physics problems
(Toy Physics) datasets, to more realistic datasets of molecular dynamics
(Molecular Dynamics), learning dynamics in non-transitive zero-sum games
(Multi Agent), and motion in 3D simulated environments (Mujoco Room). The
datasets vary in terms of the complexity of simulated dynamics and visual
richness.

For each dataset we created 50k training trajectories, and 20k test trajectories
, with each trajectory including image observations, ground truth phase state
used to generate the data, the first time derivative of the ground truth state,
and any hyper-parameters of individual trajectories. For a few of the datasets
we generate a small number of long trajectories which are used purely for
evaluation.


### Toy Physics

For all simulated systems we take the trajectories samples at every
`Δt = 0.05` intervals. For any of the non-conservative variants of each
dataset we set the friction coefficient to `0.05`. All hyper-parameters which
can be randomized are always sampled and kept fixed throughout each trajectory.
The colors of the particles, when being randomized, are always sampled uniformly
from some fixed number of options. The exact configurations used for generating
the datasets in the suite can be found in the [datasets.py] file. All systems
were simulated using `scipy.integrate.solve_ivp` .

#### Mass Spring

This dataset describes a simple harmonic motion of a particle attached to a
spring. The system has two hyper-parameters - the spring force coefficient `k`
and the mass of the particle `m`. The initial positions and momenta are sampled
jointly from an annulus, where the radius is in the interval
[r<sub>low</sub>, r<sub>high</sub>]. One can choose whether the distribution to
sample from is uniform in the annulus, or otherwise to sample uniformly the
length of the radius. To render the system on an image we visualize just the
particle as a circle, with a radius proportional to the square root of its mass.
When rendering, there are also a three additional hyper-parameters - whether to
randomize the horizontal position of the particle, since its motion is only
vertical, whether to also shift around in any direction the anchor point of the
spring, and how many possible colors can the circle representing the particle
can take. Finally, we one can in addition simulate a non-conservative system by
setting the friction coefficient to non-zero.

#### Pendulum

This dataset describes  the evolution of a particle attached to a pivot, such
that it can move freely. The system is simulated in angle space, such that it is
one dimensional. It has three hyper-parameters - the mass of the particle `m`,
the gravitational constant `g` and the pivot length `l`. The initial positions
and momenta are sampled jointly from an annulus, where the radius is in the
interval [r<sub>low</sub>, r<sub>high</sub>]. One can choose whether the
distribution to sample from is uniform in the annulus, or otherwise to sample
uniformly the length of the radius. To render the system on an image we
visualize just the particle as a circle, with radius proportioned to the square
root of its mass. When rendering, there are also a two additional
hyper-parameters - whether to also shift around in any direction the anchor
point of the pivot and how many possible colors can the circle representing the
particle can take. Finally, we one can in addition simulate a non-conservative
system by setting the friction coefficient to non-zero.

#### Double Pendulum

This dataset describes the evolution of two coupled pendulums, where the second
one's anchor point of its pivot is the center of the particle of the first one.
This leads to significantly more complicated dynamics [2]. All the
hyper-parameters are equivalent to those in the Pendulum dataset and follow the
exact same protocol.

#### Two Body

This dataset describes the gravitational motion of two particles in the plane.
The system has three hyper-parameters - the masses of the two particles `m_1`
and `m_2` and the gravitational constant `g`.  The positions and momenta of each
particle are sampled jointly from an annulus, where the radius is in the
interval [r<sub>low</sub>, r<sub>high</sub>]. To render the system on an image
we visualize just each particle as a circle, with radius proportioned to the
square root of its mass. When rendering, there are also a two additional
hyper-parameters - whether to also shift around in any direction the center of
mass of the system and how many possible colors can the circles representing the
particles can take.


### Multi Agent

These datasets describe the dynamics of non-transitive zero-sum games. Here we
consider two prominent examples of such games: **matching pennies** and
**rock-paper-scissors**. We use the well-known continuous-time multi-population
replicator dynamics to drive the learning process. The ground-truth trajectories
are generated by integrating the coupled set of ODEs using an improved Euler
scheme or RK45. In both cases the ground-truth state, i.e., joint strategy
profile (joint policy), and its first order time derivative, is recorded at
regular time intervals `Δt = 0.1`. Trajectories start from uniformly sampled
points on the product of the policy simplexes. No noise is added to the
trajectories.

As all other datasets use images as inputs, we define the observation as the
outer product of the strategy profiles of the two players. The resulting matrix
captures the probability mass that falls on each pure joint strategy profile
(joint action). In this dataset, the observations are a loss-less representation
of the ground-truth state and are upsampled to `32 x 32 x 3` images through
tiling.


### Mujoco Room

These datasets are composed of multiple scenes each consisting of a camera
moving around a room with 5 randomly placed objects. The objects were sampled
from four shape types: a sphere, a capsule, a cylinder and a box. Each room was
different due to the randomly sampled colors of the wall, floor and objects. The
dynamics were created by motion and rotation of the camera. The **cirlce**
dataset is generated by rotating the camera around a single randomly sampled
parallel of the unit hemisphere centered around the middle of the room. The
**spiral** dataset is generated by  rotating the camera on a spiral moving down
the unit hemisphere. For each trajectory an initial radius and angle are sampled
and then converted into the Cartesian coordinates of the camera. The dynamics
are discretised by moving the camera using step size of `0.1` degrees in a way
that keeps the camera on the unit hemisphere while facing the center of the
room. For the **spiral** dataset, the camera path traces out a golden spiral
starting at the height corresponding to the originally sampled radius on the
unit hemisphere. The rendered scenes are used as observations, and the Cartesian
coordinates of the camera and its velocities estimated through finite
differences as the state. Each trajectory was generated using [MuJoCo].

### Molecular Dynamics

These datasets comprise a type of interaction potential commonly studied
using computer simulation techniques, such as molecular dynamics or Monte Carlo
simulations. In particular, we generated two datasets employing a Lennard-Jones
potential of increasing complexity: one comprising only 4 particles at a very
low density and another one for a 16-particle liquid at a higher density. For
rendering these datasets we used the same scheme as for the Toy Physics datasets.
All masses are set to unity and we represent particles by circles of equal size
with a radius value adjusted to fit the canvas well. The illustrations are
therefore not representative of the density of the system. In addition, we
assigned different colors to the particles to facilitate tracking their
trajectories.

We created the datasets in two steps: we first generated the raw molecular
dynamics data using the simulation software [LAMMPS], and then converted the
resulting trajectories into a trainable format. For the final datasets available
for download, we combined simulation data from 100 different molecular dynamics
trajectories, each corresponding to a different random initialization
(see Appendix 1.3 of the paper for details). Here we provide a LAMMPS input
script [lj_16.lmp] to generate data for a single seed and a script
[generate_dataset.py] to turn the text-based simulation output into
a trainable format. By default, the simulation is set up for the 16-particle
system, but we provide inline comments on which lines need changing for the
4-particle dataset.



## Installation

All package requirements are listed in `requirements.txt`. To install the code
run in your shell the following commands:

```shell
git clone https://github.com/deepmind/dm_hamiltonian_dynamics_suite
pip install -r ./dm_hamiltonian_dynamics_suite/requirements.txt
pip install ./dm_hamiltonian_dynamics_suite
pip install --upgrade "jax[XXX]"
```

where `XXX` is the correct type of accelerator that you have on your machine.
Note that if you are using a GPU you might need `XXX` to also include the
correct version of CUDA and cuDNN installed on your machine.
For more details please read [here](https://github.com/google/jax#installation).


## Usage

You can find an example of how to generate a dataset and the load and visualize
them in the [Colab notebook provided].


## References

**Which priors matter? Benchmarking models for learning latent dynamics**

Aleksandar Botev, Drew Jaegle, Peter Wirnsberger, Daniel Hennes and Irina
Higgins

URL: https://openreview.net/forum?id=qBl8hnwR0px

**SyMetric: Measuring the Quality of Learnt Hamiltonian Dynamics Inferred from Vision**

Irina Higgins, Peter Wirnsberger, Andrew Jaegle, Aleksandar Botev

URL: https://openreview.net/forum?id=9Qu0U9Fj7IP

## Disclaimer

This is not an official Google product.

[DM Hamiltonian Dynamics Suite]: https://console.cloud.google.com/storage/browser/dm-hamiltonian-dynamics-suite
[Colab notebook provided]: https://colab.research.google.com/github/deepmind/dm_hamiltonian_dynamics_suite/blob/master/visualize_datasets.ipynb
[datasets.py]: https://github.com/deepmind/dm_hamiltonian_dynamics_suite/blob/master/dm_hamiltonian_dynamics_suite/datasets.py
[lj_16.lmp]: https://github.com/deepmind/dm_hamiltonian_dynamics_suite/blob/master/dm_hamiltonian_dynamics_suite/molecular_dynamics/lj_16.lmp
[generate_dataset.py]: https://github.com/deepmind/dm_hamiltonian_dynamics_suite/blob/master/dm_hamiltonian_dynamics_suite/generate_dataset.py
[LAMMPS]: https://lammps.sandia.gov/
[MuJoCo]: http://www.mujoco.org/

