# -*- coding: utf-8 -*-
"""
BatteryBERT
===================

BatteryBERT: A Pre-trained Language Model for Battery Database Enhancement
"""

import logging

__title__ = 'BatteryBERT'
__version__ = '1.0.0'
__author__ = 'Shu Huang'
__email__ = 'sh2009@cam.ac.uk'


logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

from . import apps
from . import finetune
from . import pretrain
