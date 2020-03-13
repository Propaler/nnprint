from abc import abstractmethod

from nnprint.visualizations.abstract import AbstractVisualization
from nnprint.drawing.object_drawable import ObjectDrawable


class BaseVisualization(AbstractVisualization, ObjectDrawable):
    """Defines the functions in common with any display 
    such as positioning of subtitles and description data.
    
    Args:
        model: Model to be used.
        mask: Mask structure to represent weights, filters or layers to be hidden.
    """

    def __init__(self, model, mask=None):
        self.model = model
        # TODO automatically identify the type of mask structure given.
        self.mask = mask

    def title(self, text, position=None):
        """Add a title to visualization"""
        # TODO
        pass

    @abstractmethod
    def print():
        """Shows the output visualization"""
        # TODO
        pass

    @abstractmethod
    def save(path):
        """Saves the visualization to a given path"""
        # TODO
        pass

    def set_mask(mask):
        pass
