#!/usr/bin/python
# coding=utf-8

# OS and ROS
import os
import rospy
import rospkg
import std_msgs
from support.srv import speech, speechRequest, led, ledRequest
from support.msg import ExperimentTrigger

# Utils
from functools import partial
import pickle
import datetime
import time
from copy import copy
import random

# Learner
from caal import attributelearning as al
import numpy as np
import anytree as at

# PyQT 5
from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QSizePolicy, QPushButton
from python_qt_binding.QtGui import QFont, QPixmap
from python_qt_binding.QtCore import QWaitCondition, QMutex, Qt, pyqtSignal, \
    QThread

"""
Subclassing of QWidget that is responsive to arrow keyboard events 
"""

class ResponsiveWidget(QWidget):
    press_enter = pyqtSignal()
    def __init__(self, plugin):
        super(ResponsiveWidget, self).__init__()
        self.gui = plugin

    def keyPressEvent(self, event):
        k = event.key()
        rospy.loginfo("Key pressed: {}".format(k))
        if k == Qt.Key_Left or k == Qt.Key_Right or k == Qt.Key_Down:
            self.gui.keyboard_lock.lock()
            self.gui.last_key_pressed = k
            self.gui.keyboard_lock.unlock()
            rospy.loginfo("Arrow key pressed")
            self.gui.keyboard_condition.wakeAll()
        if k == Qt.Key_Enter or k ==  Qt.Key_Return:
            rospy.loginfo("Enter key pressed")
            self.gui.keyboard_lock.lock()
            self.gui.last_key_pressed = k
            self.gui.keyboard_lock.unlock()
            self.press_enter.emit()

"""
QThread that runs the Learner code
"""


class LearnerThread(QThread):
    show_question = pyqtSignal(str)
    show_fixed = pyqtSignal(str)
    show_tab = pyqtSignal(int)
    show_score = pyqtSignal(str)
    show_score_outro = pyqtSignal(str)
    flash_question = pyqtSignal(int)
    flash_label = pyqtSignal(int)

    def __init__(self, plugin):
        super(LearnerThread, self).__init__()
        self.gui = plugin
        self.current_phase = None
        self.learners = None
        self.nao_talk = rospy.ServiceProxy('nao_speech', speech)
        self.nao_led = rospy.ServiceProxy('nao_led', led)
        
        # Initialize log dictionary
        self.experiment_data = dict()
        # Time - to name the log folder
        self.ts = time.time()
        # Initialize learner
        self._initialize_learner()

    def run(self):
        rospy.loginfo("Thread started")

        while True:
            self.gui.phase_lock.lock()
            try:
                rospy.loginfo("Wait for new phase")
                self.gui.phase_condition.wait(self.gui.phase_lock)
                rospy.loginfo("New phase!")
                self.current_phase = self.gui.sm
                self.current_learner = self.gui.training_counter
            finally:
                self.gui.phase_lock.unlock()
            if self.current_phase == 1:
                self._training_phase()
            elif self.current_phase == 4:
                self._learning_phase(self.current_learner)
            elif self.current_phase == 6:
                self._experiment_log()
                rospy.loginfo("Exit learner thread")
                break

    def _training_phase(self):
        
        fixed_question = "Are these animals MAMMALS?"
        if self.gui.finnish_help:
            fixed_question += '\n' + 'Ovatko nämä eläimet NISÄKKÄITÄ?'
        
        mixed_animals = ['Zebra', 'Lion', 'Squirrel', 'Dolphin', 'Blue whale',\
            'Penguin', 'Python', 'Hawk', 'Salmon', 'Frog']
        mixed_animals_fi = ['Seepra', 'Leijona', 'Orava', 'Delfiini', \
            'Sinivalas', 'Pingviini', 'Python', 'Haukka', 'Lohi', 'Sammakko']
        
        # randomize these questions
        combined = list(zip(mixed_animals, mixed_animals_fi))
        random.shuffle(combined)
        mixed_animals[:], mixed_animals_fi[:] = zip(*combined)
        
        # add training question to log, create empty answer list
        self.experiment_data['training_questions'] = mixed_animals
        self.experiment_data['training_answers'] = list()
        
        self.show_fixed.emit("")
        self.show_question.emit("Training task")
        time.sleep(2)
        resp = self.nao_talk("Let's start the training!\\eos=1\\ " + \
            "Please reply to my questions by pressing the arrow keys on the" +\
            " keyboard.\\pau=800\\" +\
            "Right arrow for Yes,\\eos=1\\Left arrow for No\\eos=1\\and" +\
            "\\eos=1\\Down arrow \\eos=1\\ if you don't know." \
            , True, True)
            
        self.show_question.emit("")
        self.show_fixed.emit(fixed_question)
        
        resp = self.nao_talk("Let's start!", False, True)
        time.sleep(2)
        
        verbose_starting = ['What about ', 'How about ', 'And ']

        for i, animale in enumerate(mixed_animals):
            # Ask question
            if i == 0:
                question = verbose_starting[0] +\
                    animale + '?'
                self.nao_talk(question, False, True)
                if self.gui.finnish_help:
                    question += '\n' '(' + mixed_animals_fi[i] + ')'
                first_question = False
            else:
                r = np.argmax(np.random.multinomial(1,\
                    [0.2, 0.2, 0.6]))
                question = verbose_starting[r] +\
                    animale + '?'
                self.nao_talk(question, False, True)
                if self.gui.finnish_help:
                    question += '\n' '(' + mixed_animals_fi[i] + ')'
            
            self.show_question.emit(question)

            self.gui.keyboard_lock.lock()
            try:
                rospy.loginfo("Wait for keystroke")
                self.gui.keyboard_condition.wait(self.gui.keyboard_lock)
                rospy.loginfo("Received keystroke!")
                answer = self.gui.last_key_pressed
                # Decode the answer
                if answer == Qt.Key_Right:
                    answer = 1
                elif answer == Qt.Key_Left:
                    answer = 0
                elif answer == Qt.Key_Down:
                    answer = -1
                else:
                    answer = -1
                    rospy.logwarn("Received invalid answer")
            finally:
                self.gui.keyboard_lock.unlock()
            
            # log the answer
            self.experiment_data['training_answers'].append(answer)
            
            self.answer_feedback(answer)
            self.show_question.emit("")
        
        self.show_question.emit("")
        self.show_fixed.emit("")
        
        self.nao_talk("Cool, that ends the training session.\\eos=1\\" +\
        " I hope now it is clear how to answer my questions.\\eos=1\\" +\
        " If you have some questions, please ask the experimenter.\\eos=1\\" +\
        " Press Enter when you are ready to continue.", True, True)
        
        # Partial log after training session
        self._experiment_log(partial=True)
        self.show_question.emit("Training completed. Press Enter to continue.")
        
        self.gui.phase_lock.lock()
        self.gui.sm = 2
        rospy.logwarn('SM: --> 2')
        self.gui.phase_lock.unlock()
    
    def _initialize_learner(self):
        # list of names following the wn nomenclature
        self.entities = list()
        # list of names and ids following the awa nomenclature
        self.entities_awa = list()
        self.entities_id = list()
        # list of names in finnish
        self.entities_fin = list()
        # list of names and ids of the attributes
        self.attributes = list()
        self.attributes_id = list()
        
        # Path of the dataset
        if rospy.has_param('/dataset_path'):
            dataset_path = rospy.get_param('/dataset_path')
        
        # Collect the entities of AwA2: ids and names
        with open(dataset_path + '/classes_wn.txt', 'r') as f:
              for line in f:
                    entity = line.split()[1].replace('+','_')
                    entity_id = int(line.split()[0]) - 1
                    self.entities.append(entity)
                    self.entities_id.append(entity_id)

        with open(dataset_path + '/classes.txt', 'r') as f:
              for line in f:
                    entity = line.split()[1].replace('+',' ')
                    entity_id = int(line.split()[0]) - 1
                    self.entities_awa.append(entity)
                    
        with open(dataset_path + '/classes_finnish.txt', 'r') as f:
              for line in f:
                    entity = line.split()[1].replace('+',' ')
                    entity_id = int(line.split()[0]) - 1
                    self.entities_fin.append(entity)
        
        # Build the category tree
        self.ct = al.CategoryTree('mammal.n.01', similarity_tree_gamma=0.7)
        self.ct.add_leaves(self.entities)
        self.ct.simplify_tree()
        
        # Collect the attributes of AwA2
        with open(dataset_path + '/predicates.txt', 'r') as f:
              for line in f:
                    attribute = line.split()[1]
                    attribute_id = int(line.split()[0]) - 1
                    self.attributes.append(attribute)
                    self.attributes_id.append(attribute_id)
        full_table = np.loadtxt(open(dataset_path +\
            '/predicate-matrix-binary.txt', 'r'))
        binary_table = full_table[np.asarray(self.entities_id, dtype=int), :]
        self.binary_table = binary_table[:, np.asarray(self.attributes_id,\
         dtype=int)]
        
        # Compute distance for the hybrid learner
        w = at.Walker()
        paths = list()
        for entity in self.entities:
            for other in self.entities:
                if entity is not other:
                    paths.append(w.walk(self.ct.node_dictionary[\
                    self.ct.leaves_to_wn[entity]],self.ct.node_dictionary\
                    [self.ct.leaves_to_wn[other]]))

        distances = [(len(path[0]) + len(path[2])) for path in paths]

        self.min_distance = min(distances)
        self.max_distance = max(distances)
        
        # Experiment parameters from rosparam
        parameters_available = \
            rospy.has_param('/time_bag')
        
        if parameters_available:
            self.time_bag = rospy.get_param('/time_bag')
        else:
            rospy.logerr('Parameters needed are not available')
            exit(5)
            
        self.selected_attributes = np.array([[22, 51], [30, 52], [31, 54]])
        
        self.verbose_attributes = np.array([\
            ['Do these animals have PAWS?','Do these animals EAT FISH?'],\
              ['Do these animals have HORNS?', 'Do these animals EAT MEAT?'],\
              ['Do these animals have CLAWS?','Are these animals HERBIVORE?']])
        
        self.verbose_attributes_fi = np.array([\
            ['Onko näillä eläimillä TASSUT?','Syövätkö nämä eläimet KALAA?'],\
            ['Onko näillä eläimillä SARVET?', 'Syövätkö nämä eläimet LIHAA?'],\
            ['Onko näillä eläimillä KYNNET?','Ovatko nämä eläimet KASVISSYÖJIÄ?']])
        self.verbose_attributes_nao = np.array([\
            ['Do these animals have\\pau=200\\ paws?','Do these animals\\pau=200\\ eat fish?'],\
            ['Do these animals have\\pau=200\\ horns?', 'Do these animals\\pau=200\\ eat meat?'],\
            ['Do these animals have\\pau=200\\ claws?','Are these animals\\pau=200\\ herbivore?']])
        
        # Initialize data for logging
        self.experiment_data['attributes'] = self.selected_attributes
        self.experiment_data['time_budget'] = self.time_bag
            
    def _experiment_log(self, partial=False):
    
        if partial:
            rospy.logwarn('PARTIAL LOGGING')
        else:
            rospy.logwarn('COMPLETE LOGGING')
        
        directory_name = \
            datetime.datetime.fromtimestamp(self.ts).strftime('%m%d_%H%M')
        
        # add name to directory_name
        directory_name += self.gui.experiment_name
        
        # path for the log
        if rospy.has_param('/log_path'):
            folderpath = os.path.join(rospy.get_param('/log_path'), \
                directory_name)
        
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
            
        if partial:
            partial_folderpath = os.path.join(folderpath, 'partial')
            if not os.path.exists(partial_folderpath):
                os.makedirs(partial_folderpath)
            filepath = os.path.join(partial_folderpath, 'partial.pkl')
        else:
            filepath = os.path.join(folderpath, 'experiment.pkl')

        f = open(filepath, 'wb')
        pickle.dump(self.experiment_data,f)
        f.close()
        rospy.logwarn('LOGGING DONE')
        
    def _learning_phase(self,id_learner):
        # Initialize variables
        self.experiment_data['learners'] = self.learners
        learner = self.learners[id_learner]
        number_of_questions = len(self.entities)
        hard_performance_threshold = 0.49
        
        verbose_starting = ['What about ', 'How about ', 'And ']
        
        if id_learner == 0:
            self.nao_talk("Let us start with the first teaching session"+\
            "\\eos=1\\ I will ask two sets of questions.", True, True)
        else:
            self.nao_talk("As before, I will ask two sets of questions"\
                +"\\pau=1000\\ Remember, \\eos=1\\ pay attention to how "\
                +"I choose my questions, \\eos=1\\ because I will use "\
                +"a different strategy now!", True, True)
        
        self.show_question.emit("")
        self.show_fixed.emit("")    
        time.sleep(5)
        
        rospy.logwarn('Starting with {} out of {}'.format(\
            learner, self.learners))

        for id_attribute, attribute in \
        enumerate(self.selected_attributes[id_learner,:]):
            
            self.nao_talk("I will now ask you:\\eos=1\\{}".format(\
            self.verbose_attributes_nao[id_learner][id_attribute]), False, True)
            question = self.verbose_attributes[id_learner][id_attribute]
            
            if self.gui.finnish_help:
                question += '\n' + self.verbose_attributes_fi[id_learner][id_attribute]
            
            self.show_fixed.emit(question)
            time.sleep(1)
            self.nao_talk("We have {} seconds of time.\\eos=1\\ Let's start!"\
                .format(self.time_bag), False, True)
            time.sleep(3)
            
            # initialize attribute-learner specific logging data
            self.experiment_data['question_h',learner,attribute] = list()
            self.experiment_data['answer_h',learner,attribute] = list()
            self.experiment_data['response_h',learner,attribute] = list()
            self.experiment_data['response_only_h',learner,attribute] = list()
            self.experiment_data['testset_p',learner,attribute] = list()
            self.experiment_data['dunno_p',learner,attribute] = list()
            self.experiment_data['correct_answers_p',learner,attribute] = 0
            
            # re-initialize the learner
            previous_index = None
            self.ct.reset_learning()
            shuffled_entities = copy(self.entities)
            dunno_entities = list()
            random.shuffle(shuffled_entities)
            
            # vector of the learned attributes from user, ordered as AwA2
            learned_user = -1*np.ones([len(self.entities)])
            # vector of the learned attributes from dataset, ordered as AwA2
            learned_truth = -1*np.ones([len(self.entities)])
            
            # intialize time budget
            time_budget = self.time_bag
            
            ### LEARNING PHASE ###
            first_question = True
            while time_budget > 0 and len(shuffled_entities) > 0:
                # select question
                if learner == 'greedy':
                    rospy.loginfo('greedy')
                    awa_index, tilde = self.ct.select_greedy_query\
                    (shuffled_entities)
                elif learner == 'similar':
                    rospy.loginfo('similar')
                    awa_index, tilde = self.ct.select_closest_query\
                    (shuffled_entities, previous_index)
                elif learner == 'hybrid':
                    rospy.loginfo('hybrid')
                    awa_index, tilde = self.ct.select_hybrid_query(\
                        shuffled_entities, previous_index, -1.6, 0,\
                        self.min_distance, self.max_distance, 0.7)
                elif learner == 'random':
                    awa_index = self.ct.leaves.index(shuffled_entities[-1])
                    shuffled_entities.pop()
                else:
                    rospy.logerr('ERROR: unknown learner')
                    exit(10)

                # store the picked question (for similar and hybrid learners)
                previous_index = awa_index
                
                # log the picked question
                self.experiment_data['question_h',learner,attribute].append(\
                    awa_index)
                
                # Ask question
                if first_question:
                    question = verbose_starting[0] +\
                        self.entities_awa[awa_index] + '?'
                    first_question = False
                else:
                    r = np.argmax(np.random.multinomial(1,\
                        [0.2, 0.2, 0.6]))
                    question = verbose_starting[r] +\
                        self.entities_awa[awa_index] + '?'
                
                # Start the timer
                text_question = copy(question)
                if self.gui.finnish_help:
                    text_question += ' (' + self.entities_fin[awa_index] + ')'
                
                t0 = time.time()
                self.nao_talk(question, False, True)
                self.show_question.emit(text_question)
                
                t1 = time.time()
                # Wait for an answer
                self.gui.keyboard_lock.lock()
                try:
                    rospy.loginfo("Wait for keystroke")
                    self.gui.keyboard_condition.wait(self.gui.keyboard_lock)
                    answer = self.gui.last_key_pressed
                    rospy.loginfo("Received keystroke!")
                finally:
                    self.gui.keyboard_lock.unlock()
                
                # Stop the timer
                t2= time.time()
                
                # Decode the answer
                if answer == Qt.Key_Right:
                    answer = 1
                elif answer == Qt.Key_Left:
                    answer = 0
                elif answer == Qt.Key_Down:
                    answer = -1
                    dunno_entities.append(awa_index)
                else:
                    answer = -1
                    rospy.logwarn("Received invalid answer")
                
                # Flash the label of question with the answer
                self.answer_feedback(answer)
                
                # Store answer
                learned_user[awa_index] = answer           
                self.experiment_data['response_h',learner,\
                    attribute].append(t2-t0)
                self.experiment_data['response_only_h',learner,\
                    attribute].append(t2-t1)    
                self.experiment_data['answer_h',learner,attribute].append(\
                    learned_user[awa_index])
                
                # Update the time budget
                time_budget -= (t2-t0)
                
                learned_truth[awa_index] = self.binary_table[awa_index,\
                    attribute]
                if learned_truth[awa_index] == learned_user[awa_index]:
                    self.experiment_data['correct_answers_p',learner,\
                        attribute] += 1
                
                # Update the model with the answer
                if answer != -1:
                    self.ct.node_dictionary[self.ct.leaves_to_wn[\
                        self.entities[awa_index]]].push_information(\
                        learned_user[awa_index] == 1)

            ### EVALUATION PHASE ###
            
            # performance on the unseen entities
            performance = -1*np.ones([len(shuffled_entities)])
            for i, entity in enumerate(shuffled_entities):
                theta = self.ct.node_dictionary[\
                    self.ct.leaves_to_wn[entity]].theta
                if theta > 1 - hard_performance_threshold or \
                theta < hard_performance_threshold:
                    if theta > 1 - hard_performance_threshold:
                        performance[i] = 1 if \
                            self.binary_table[self.entities.index(entity), \
                            attribute] == 1 else 0
                    else:
                        performance[i] = 1 if \
                            self.binary_table[self.entities.index(entity), \
                            attribute] == 0 else 0
                else:
                    performance[i] = 0
            
            # performance on the unknown entities
            dunno_performance = -1*np.ones([len(dunno_entities)])
            for i, entity in enumerate(dunno_entities):
                theta = self.ct.node_dictionary[self.ct.leaves_to_wn\
                    [self.entities[entity]]].theta
                if theta > 1 - hard_performance_threshold or \
                theta < hard_performance_threshold:
                    if theta > 1 - hard_performance_threshold:
                        dunno_performance[i] = 1 if \
                        self.binary_table[entity, attribute] == 1 else 0
                    else:
                        dunno_performance[i] = 1 if \
                        self.binary_table[entity, attribute] == 0 else 0
                else:
                    dunno_performance[i] = 0
            
            # log the performance
            self.experiment_data['testset_p',learner,attribute].append(\
                np.sum(performance))
            self.experiment_data['dunno_p',learner,attribute].append(\
                np.sum(dunno_performance))
            
            self.show_question.emit("")
            self.show_fixed.emit("")
            
            if id_attribute == 0:
                self.nao_talk("Time is over!\\eos=1\\"+\
                " We are done with this set of questions!"+\
                    "\\eos=1\\ Let's have a small break!", True, True)
                self.show_question.emit("Small Break")
                time.sleep(10)
            else:
                time.sleep(2)
        
        self.nao_talk('This teaching session is over!\\eos=1\\Thank you'+\
        ' for your patience!', True ,True)
        
        self.show_score.emit("")
        self.show_score_outro.emit("")
        self.show_tab.emit(2)
        
        if id_learner== 0:
            self.nao_talk("I will take a test now, to check if"+\
            " I learned something.\\eos=1\\ "+\
            "Let's see how well I learned with this technique"+\
            "\\pau=1000\\Wish me luck!", True, True)
        else:
            self.nao_talk("I will take the test again.\\eos=1\\ "+\
            "Let's see how well I learned with this different technique"+\
            "\\pau=1000\\Wish me luck!", True, True)
        
        # Robot "thinking" led animations
        self.nao_led(1, 0, 5, True)
        
        self.nao_talk("I am done with the test!", True, True)
        time.sleep(1)
        
        score = 0
        total = 0
        for a in self.selected_attributes[id_learner,:]:
            score += self.experiment_data['testset_p',learner,a][0]
            score += self.experiment_data['dunno_p',learner,a][0]
            score += self.experiment_data['correct_answers_p',learner,attribute] # is this suppose to be just 'a'?
            total += number_of_questions
        
        self.nao_talk("I replied correctly to {0:.1f} % of the questions!".format(100*score/total), True, True)
        self.show_score.emit("Percentage \n {0:.1f}%".format(100*score/total))
        time.sleep(3)
        
        self.nao_talk('Please now fill the questionnaires. \\eos=1\\'+\
            'When you have filled them, please continue by pressing ENTER',\
            True, True)
        self.show_score_outro.emit("Fill the questionnaire now. \n Then, press ENTER to continue")
        
        ## PARTIAL LOG
        self._experiment_log(partial=True)
        
        ## Pause to avoid skipping questionnaire
        time.sleep(10)
        
        ### ADVANCE THE STATE MACHINE
        self.gui.phase_lock.lock()
        self.gui.sm = 5
        rospy.logwarn('SM: --> 5')
        self.gui.phase_lock.unlock()
    
    def answer_feedback(self, answer):
        # the led feedback is disabled (because creepy)
        # self.nao_led(2, answer, 0.3, False)
        self.flash_question.emit(answer+1)
        self.flash_label.emit(answer+1)
        time.sleep(0.3)
        self.flash_question.emit(-1)
        self.flash_label.emit(-1)

"""
Plugin to be launched with rqt
"""

class QueryPlugin(Plugin):
    def __init__(self, context):
        super(QueryPlugin, self).__init__(context)
        
        self.learner_list = None
        self.finnish_help = None
        
        # Initialize subscriber for experiment starting
        self.experiment_subscriber = rospy.Subscriber('start_experiment',\
                                                      ExperimentTrigger,\
                                                      self._start_experiment)
                                                      
        parameters_available = \
            rospy.has_param('/experiment_config') and\
            rospy.has_param('/experiment_intro') and\
            rospy.has_param('/training_intro') and\
            rospy.has_param('/teaching_intro') and\
            rospy.has_param('/teaching_continuation') and\
            rospy.has_param('/experiment_outro')
        
        if parameters_available:
            self.experiment_config = rospy.get_param('/experiment_config')
            self.experiment_intro = rospy.get_param('/experiment_intro')
            self.training_intro = rospy.get_param('/training_intro')
            self.teaching_intro = rospy.get_param('/teaching_intro')
            self.teaching_continuation = \
                rospy.get_param('/teaching_continuation')
            self.experiment_outro = rospy.get_param('/experiment_outro')
        else:
            rospy.logerr('Parameters needed are not available')
            exit(5)
        
        # Synchronization primitives for the Keyboard event
        self.keyboard_condition = QWaitCondition()
        self.keyboard_lock = QMutex()
        self.last_key_pressed = None

        # Synchronization primitives for the Phase event
        self.phase_condition = QWaitCondition()
        self.phase_lock = QMutex()
        self.sm = -1 # ID of the state in the State Machine
        self.training_counter = -1 # how many learners have been used

        # Initialize and start the LearnerThread
        self.learner_thread = LearnerThread(self)
        self.learner_thread.show_question.connect(self._set_question_text)
        self.learner_thread.flash_question.connect\
            (self._flash_question_background)
        self.learner_thread.flash_label.connect(self._flash_label)
        self.learner_thread.show_fixed.connect(self._set_fixed_text)
        self.learner_thread.show_tab.connect(self._set_tab)
        self.learner_thread.show_score.connect(self._set_score)
        self.learner_thread.show_score_outro.connect(self._set_score_outro)
        self.learner_thread.start()

        # Give QObjects reasonable names
        self.setObjectName('Query plugin')

        # Create the Widget
        self._widget = ResponsiveWidget(self)
        self._widget.press_enter.connect(self._step_state_machine)

        # Get the UI file
        ui_file = os.path.join(rospkg.RosPack().get_path('learner'), \
            'resources', 'learner.ui')
        loadUi(ui_file, self._widget)
        
        # Get the pictures for the instruction label
        picture_paths = list()
        neutral_picture_path = os.path.join(rospkg.RosPack().get_path('learner'), \
            'resources', 'drawing.png')

        picture_paths.append(os.path.join(rospkg.RosPack().get_path('learner'), \
            'resources', 'drawing_dunno.png'))
        picture_paths.append(os.path.join(rospkg.RosPack().get_path('learner'), \
            'resources', 'drawing_no.png'))
        picture_paths.append(os.path.join(rospkg.RosPack().get_path('learner'), \
            'resources', 'drawing_yes.png'))
        
        self.pictures = list()
        self.neutral_picture = QPixmap(neutral_picture_path)
        
        for p in picture_paths:
            self.pictures.append(QPixmap(p))

        # Give QObjects reasonable names
        self._widget.setObjectName('Query widget')

        # Add widget to the user interface
        context.add_widget(self._widget)

        # Create the response buttons
        button_size_policy = QSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Expanding)

        # Set up the labels
        self._widget.fixed_label.setWordWrap(True)
        self._widget.fixed_label.setFont(QFont("Sans Serif", 30))
        self._widget.question_label.setWordWrap(True)
        self._widget.question_label.setFont(QFont("Sans Serif", 30))
        self._widget.intro_label.setWordWrap(True)
        self._widget.intro_label.setFont(QFont("Sans Serif", 30))
        self._widget.score_intro_label.setWordWrap(True)
        self._widget.score_intro_label.setFont(QFont("Sans Serif", 20))
        self._widget.score_label.setWordWrap(True)
        self._widget.score_label.setFont(QFont("Sans Serif", 38))
        self._widget.score_outro_label.setWordWrap(True)
        self._widget.score_outro_label.setFont(QFont("Sans Serif", 24))
        
        # Clear the first page label
        self._widget.intro_label.setText("")

        # Activate starting tab
        self._widget.tab_manager.setCurrentIndex(1)

        # Activate keyboard grabber
        self._widget.grabKeyboard()

    def _set_question_text(self, string):
        self._widget.question_label.setText(string)
    
    def _flash_question_background(self, color_code):
        # 0 is yellow, 1 is red, 2 is green
        # -1 is the standard gray
        if color_code == 0:
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(254, 229, 0)')
        elif color_code == 1:
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(255, 120, 103)')
        elif color_code == 2:
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(145, 228, 141)')
        elif color_code == -1:
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(236, 236, 236)')
                
    def _flash_label(self, color_code):
        # 0 is not sure, 1 is no, 2 is yes
        # -1 is the standard label
        
        if color_code == -1:
            self._widget.instruction_label.setPixmap(self.neutral_picture)
        else:
            self._widget.instruction_label.setPixmap(self.pictures[color_code])
        if color_code == 0:
            
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(254, 229, 0)')
        elif color_code == 1:
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(255, 120, 103)')
        elif color_code == 2:
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(145, 228, 141)')
        elif color_code == -1:
            self._widget.question_label.setStyleSheet\
                ('background-color: rgb(236, 236, 236)')
            
        
    def _set_tab(self, int):
        self._widget.tab_manager.setCurrentIndex(int)
        
    def _set_fixed_text(self, string):
        self._widget.fixed_label.setText(string)
        
    def _set_score(self, string):
        self._widget.score_label.setText(string)
    
    def _set_score_outro(self, string):
        self._widget.score_outro_label.setText(string)
        
    def _set_intro_text(self, string):
        self._widget.intro_label.setText(string)
        
    def _start_experiment(self, msg):
        self.phase_lock.lock()
        self.learner_list = self.experiment_config[msg.type]
        self.finnish_help = msg.finnish
        self.learner_thread.learners = self.learner_list
            
        # Initialize service with NAO
        self.nao_talk = rospy.ServiceProxy('nao_speech', speech)
        
        # Need to skip some phases?
        if msg.skip == -1:
            self.sm = 0
            self.training_counter = 0
        elif msg.skip == 0:
            self.sm = 3
            self.training_counter = 0
        else:
            self.sm = 3
            self.training_counter = msg.skip
        
        self.experiment_name = msg.name
        
        rospy.logwarn("Start Interaction '{}': order {} with finnish: {}"\
            .format(self.experiment_name, msg.type, msg.finnish))
        
        self.nao_talk(self.experiment_intro ,True,True)
        
        self._set_intro_text("Press ENTER to start the training.")
        self._set_question_text("")
        self._set_fixed_text("")
        self.phase_lock.unlock()

    def _step_state_machine(self):
        try:
            rospy.loginfo('State Machine function entered')
            self.phase_lock.lock()
            if self.sm == 0:
                # I am at the beginning of training and I received 
                # Enter on keyboard --> Start training
                self._set_tab(0)
                rospy.logwarn("SM: 0 --> 1")
                self.sm = 1
                self.phase_lock.unlock()
                self.phase_condition.wakeAll()
                
                self.nao_talk(self.training_intro ,True,True)
                return
            elif self.sm == 2:
                # The training is done and I received an Enter
                self._set_tab(1)
                rospy.logwarn("SM: 2 --> 3")
                self.sm = 3
                self.phase_lock.unlock()
                self.phase_condition.wakeAll()
                
                self._set_fixed_text("")
                self._set_question_text("")
                self._set_intro_text("Press Enter to start the experiment")
                
                if self.training_counter == 0:
                    self.nao_talk(self.teaching_intro, True, True)
                else:
                    self.nao_talk(self.teaching_continuation, True, True)
                
                return
            elif self.sm == 3:
                # I am in the intro page before exp and I receive an Enter
                # --> Start the experiment
                self._set_tab(0)
                rospy.logwarn("SM: 3 --> 4")
                self.sm = 4
                self.phase_lock.unlock()
                self.phase_condition.wakeAll()
                
                return
            elif self.sm == 5:
                if self.training_counter < 2:
                    # one phase of experiment is done, still more to go
                    self._set_tab(1)
                    rospy.logwarn("SM: 5 --> 3")
                    self.sm = 3
                    self.training_counter += 1
                    rospy.loginfo('training counter: {}'\
                        .format(self.training_counter))
                    self.phase_lock.unlock()
                    self.phase_condition.wakeAll()
                    
                    if self.training_counter == 0:
                        self.nao_talk(self.teaching_intro, True, True)
                    else:
                        self.nao_talk(self.teaching_continuation, True, True)
                    return
                else:
                    # all phases of experiment are done
                    self._set_tab(1)
                    rospy.logwarn("SM: 5 --> 6")
                    self.sm = 6
                    self.phase_lock.unlock()
                    self.phase_condition.wakeAll()
                    
                    self.nao_talk(self.experiment_outro, True, True)
                    self._set_intro_text("")
                    return
        finally:
            self.phase_lock.unlock()
