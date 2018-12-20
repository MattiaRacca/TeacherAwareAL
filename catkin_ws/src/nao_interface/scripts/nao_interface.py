#!/usr/bin/env python
import rospy
import random
import almath
import time
from support.srv import speech, speechResponse, led, ledResponse
from naoqi import ALProxy


class NaoLearner:
    def __init__(self):
        self.IP = rospy.get_param('/nao_IP')

        try:
            self.tts = ALProxy("ALTextToSpeech", self.IP, 9559)
            self.atts = ALProxy("ALAnimatedSpeech", self.IP, 9559)
            
            self.tts.setParameter("speed", 85)
            self.atts.setBodyLanguageModeFromStr("contextual")
            
            self.motion = ALProxy("ALMotion", self.IP, 9559)

            self.awareness = ALProxy("ALBasicAwareness", self.IP, 9559)
            self.awareness.stopAwareness()
            
            self._sitting_setup()

            self.faceProxy = ALProxy("ALFaceDetection", self.IP, 9559)
            self.faceProxy.enableTracking(True)

            self.leds = ALProxy("ALLeds", self.IP, 9559)
            self.color_list = ['yellow','red','green']
        except RuntimeError as e:
            rospy.logerr('Not able to connect to NAO at IP ' + self.IP)
            exit(5)
        
        # Speech service
        speech_srv = rospy.Service('nao_speech', speech, self._speech_callback)
        # LEDs service
        led_srv = rospy.Service('nao_led', led, self._led_callback)
    
    def _sitting_setup(self):
        # NAO's leg will be compliant
        # For animated speech when sitting
        
        legs = ['LHipYawPitch','LHipRoll','LHipPitch','LKneePitch',\
        'LAnklePitch','LAnkleRoll',  'RHipYawPitch','RHipRoll',\
        'RHipPitch','RKneePitch','RAnklePitch','RAnkleRoll']
        
        self.motion.setStiffnesses(legs, 0.0)
        
    def _speech_callback(self, req):
        sentence = req.sentence
        self._nao_look(0, 5)
                
        if req.bodylanguage:
            if req.blocking:
                self.atts.say(sentence)
            else:
                self.atts.post.say(sentence)
        else:
            if req.blocking:
                self.tts.say(sentence)
            else:
                self.tts.post.say(sentence)
        self._nao_look(0, 5)
        return speechResponse(True)
        
    def _led_callback(self, req):
        self._nao_look(0, 5)
        if req.type == 1:
            # concentrated animation
            if req.blocking:
                self.leds.rasta(req.duration)
            else:
                self.leds.post.rasta(req.duration)
        elif req.type == 2:
            # answer feedback
            if req.blocking:
                self.leds.fadeRGB("FaceLeds",\
                    self.color_list[int(req.answer)+1], req.duration)
            else:
                self.leds.post.fadeRGB("FaceLeds",\
                    self.color_list[int(req.answer)+1], req.duration)
        self._nao_look(0, 5)
        return ledResponse(True)

    # Makes NAO move head to given angles in degrees (Absolute not incremental)
    # Pitch is positive towards robots body
    # Yaw is positive towards robots left   
    def _nao_look(self, yaw, pitch):
        stiffness = 0.9
        fractionMaxSpeed = 0.1
        angles = [yaw * almath.TO_RAD, pitch * almath.TO_RAD]
        self.motion.setStiffnesses("Head", stiffness)
        self.motion.setAngles("Head", angles, fractionMaxSpeed)

    # Makes NAO nod at a given pitch
    def _nao_affirmative_nod(self, pitch):
        pitch_RAD = pitch * almath.TO_RAD
        angle_list = [0, -pitch_RAD, 0, pitch_RAD, 0]
        time_list = [0.4, 0.8, 1.2, 1.6, 3.0]
        isAbsolute = True
        self.motion.post.angleInterpolation("HeadPitch", angle_list, time_list,
                                            isAbsolute)
    # Makes NAO nod at a given pitch
    def _nao_sorry_nod(self, pitch):
        pitch_RAD = pitch * almath.TO_RAD
        angle_list = [0, pitch_RAD, 0]
        time_list = [0.4, 1.2, 1.6]
        isAbsolute = True
        self.motion.post.angleInterpolation("HeadPitch", angle_list, time_list,
                                            isAbsolute)


if __name__ == "__main__":
    rospy.init_node('nao_learner_node')
    nl = NaoLearner()
    rospy.spin()
