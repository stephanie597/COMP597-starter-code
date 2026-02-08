"""Thin wrapper around the datasets module for ResNet152.

This module is a very thin wrapper around the Hugging Face datasets module. It 
makes the translation between the config object and the datasets module for 
ResNet152 training, specifically handling imagefolder format and FakeImageNet data.

"""
from src.data.resnet152.data import *


