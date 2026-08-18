from .configuration_xyz import *
from .modeling_xyz import *
from .processing_xyz import *

XYZConfig.register_for_auto_class()
XYZModel.register_for_auto_class()
XYZForCausalLM.register_for_auto_class("AutoModelForCausalLM")
XYZProcessor.register_for_auto_class()
