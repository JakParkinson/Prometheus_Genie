from enum import Enum

class Interactions(Enum):
    """Enum of known interactions"""
    GLASHOW_RESONANCE = 0
    CHARGED_CURRENT = 1
    NEUTRAL_CURRENT = 2
    DIMUON = 3
    UNKNOWN = 4  # Added UNKNOWN interaction type if GENIE output doesnt fit