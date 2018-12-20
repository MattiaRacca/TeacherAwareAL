#!/usr/bin/python

import os
import errno
import datetime
import rospy
import rospkg
import std_msgs
import roslaunch

from support.msg import ExperimentTrigger

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QPushButton, \
    QLineEdit, QMessageBox
from functools import partial


class ControlPlugin(Plugin):

    def __init__(self, context):
        super(ControlPlugin, self).__init__(context)
        self.experiment_type = None
        self.skip = -1
        self.skip_check = True

        # Give QObjects reasonable names
        self.setObjectName('Control plugin')

        # Create publisher
        self.start_publisher = rospy.Publisher('start_experiment',
                                                ExperimentTrigger,
                                                queue_size=10)
        # Create QWidget
        self._widget = QWidget()

        # Get path to UI file which should be in the "resource"
        # folder of this package
        ui_file = os.path.join(rospkg.RosPack().get_path('control_gui'),
                               'resources', 'control_gui.ui')

        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget)

        # Give QObjects reasonable names
        self._widget.setObjectName('Control  widget')

        # Add widget to the user interface
        context.add_widget(self._widget)

        # Create list of Radiobuttons
        self.radiobuttons = list()
        self.radiobuttons.append(self._widget.r123)
        self.radiobuttons.append(self._widget.r132)
        self.radiobuttons.append(self._widget.r213)
        self.radiobuttons.append(self._widget.r231)
        self.radiobuttons.append(self._widget.r312)
        self.radiobuttons.append(self._widget.r321)
        
        # Create list of Skip boxes
        self.skipradio = list()
        self.skipradio.append(self._widget.skip0)
        self.skipradio.append(self._widget.skip1)
        self.skipradio.append(self._widget.skip2)

        # Connect buttons to slots
        self._widget.startButton.pressed.connect(self._start_experiment)

    def _start_experiment(self):
        for i, rb in enumerate(self.radiobuttons):
            if rb.isChecked():
                self.experiment_type = i
        
        for i, rb in enumerate(self.skipradio):
            if rb.isChecked():
                self.skip = i
        
        if self.skip != -1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Are you sure you want to skip phases?")
            msg.setWindowTitle("Careful")
            msg.exec_()
            if self.skip_check:
                self.skip_check = False
                return
        
        if self.experiment_type == None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Select which experiment to run!")
            msg.setWindowTitle("Careful")
            msg.exec_()
            return

        rospy.loginfo('Starting the experiment ' + str(self.experiment_type))
        
        # Create the message
        message = ExperimentTrigger()
        message.type = self.experiment_type
        message.finnish = self._widget.finnish_box.isChecked()
        message.skip = self.skip
        message.name = self._widget.name_text.text()
        
        self.start_publisher.publish(message)
        self._widget.startButton.setEnabled(False)

    def shutdown_plugin(self):
        self.start_publisher.unregister()
