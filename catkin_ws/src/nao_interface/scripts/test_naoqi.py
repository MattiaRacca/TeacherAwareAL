#!/usr/bin/env python
from naoqi import ALProxy

tts = ALProxy("ALTextToSpeech", "169.254.232.44", 9559)
tts.say("Hello, world!")
