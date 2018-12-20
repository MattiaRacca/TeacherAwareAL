# Content
Repository for the paper *Teacher-Aware Active Robot Learning.*, HRI 2019.

Structure:
- *caal*: contains the core code behind the different active learners
- *catkin_ws*: it is a catkin workspace containing all the ROS related stuff namely: *nao_interface*, *learner* (GUI used by the subject in the experiment), *control_gui* (GUI for the experimenter), *support* (mainly useful launch files). It has to be initialized `catkin_init_workspace` and compiled `caktin_make`.
- *dataset*: contains AwA2 list of attributes and entities + ground truth (no photos)
- *notebooks*: contains Jupyter notebooks for the simulation and the analysis of experiment data

# External Dependencies
- ROS (I use Kinetic but I don't see why it should not work with Indigo or Melodic)
- Python (2.7 as it is the one supported by ROS; non-ROS code runs also on Python 3)
- NAO SDK: NAOqi 2.1.2 (see *NAOqi_guide*)
- Python Packages: NLTK, numpy, scipy, anytree, bidict
- PyQT 5 (for the experiment GUI)

# How to run the simulation experiment
- run the Jupyter notebook `CategoryAttributeLearning.ipynb`

# How to run the experiment (requires a NAO robot)
- first, create the catkin workspace with `catkin_init_workspace`, compiled it `caktin_make` and source it `source devel/set`.
- also, the *caal* folder has to be in your `$PYTHONPATH`
- inside the catkin workspace, under *support/launch* there are a couple of launch files. The *experiment.launch* is for the experiment as reported in the paper.
- the launch file loads a couple of parameters (where to log, where to find the dataset, time budget) plus the IP of your NAO. These have to be changed accordingly.
- start with `roslaunch support experiment.launch`
- the launch file opens connection with NAO and spawns the 2 GUIs (one for the subject, one for experimenter - to be running on different screens)

# License
Everything authored by me is released under the GNU GPL 3.0 license.
