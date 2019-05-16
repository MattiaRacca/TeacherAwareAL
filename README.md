# Teacher-Aware Active Robot Learning
## Content
Repository for the paper *M. Racca, A. Oulasvirta and V. Kyrki, Teacher-Aware Active Robot Learning*, HRI 2019 ([preprint available here](https://mattiaracca.files.wordpress.com/2018/12/hri19_cr_v2.pdf)).

Repo structure:
- *caal*: python package containing the core code behind the different active learners
- *catkin_ws*: catkin workspace containing all the ROS related components: *nao_interface*, *learner* (GUI used by the subject in the experiment and interface to the learners in *caal*), *control_gui* (GUI for the experimenter), *support* (useful launch files)
- *dataset*: contains AwA2 list of attributes and entities + ground truth (no photos)
- *notebooks*: contains Jupyter notebooks for the simulation

## External Dependencies
- ROS (I use Kinetic but I don't see why it should not work with Indigo or Melodic)
- Python (2.7 as it is the one supported by ROS; non-ROS code runs also on Python 3)
- NAO SDK: NAOqi 2.1.2 (see *NAOqi_guide*)
- Python Packages: NLTK, numpy, scipy, anytree, bidict
- PyQT 5 (for the experiment GUI)

## How to run the simulation experiment
- run the Jupyter notebook `CategoryAttributeLearning.ipynb`.

## How to run the experiment (requires a NAO robot)
- first, compile the catkin workspace with `caktin_make` (*catkin_make* should also do *`catkin_init_workspace`* for you) and sourced it `source devel/setup.bash`
- add the *caal* folder to your `$PYTHONPATH`, with `export PYTHONPATH=/path_to_caal_folder/caal:$PYTHONPATH`
- inside the catkin workspace, under *support/launch* there are a couple of launch files. The *experiment.launch* is for the experiment as reported in the paper
- the launch file loads various ROS parameters (where to log, where to find the dataset, time budget), including the IP of your NAO. These have to be changed according to your liking and setup
- start with `roslaunch support experiment.launch`
- the launch file opens connection with NAO and spawns the 2 GUIs (one for the subject, one for experimenter). The two GUIs were displayed on two different screens during the experiment

## License
Everything authored by me is released under the GNU GPL 3.0 license.
