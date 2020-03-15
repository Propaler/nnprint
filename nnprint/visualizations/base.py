from abc import abstractmethod

from nnprint.visualizations.abstract import AbstractVisualization
from nnprint.drawing.object_drawable import ObjectDrawable

from PIL import Image
from PIL import ImageDraw

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
    def print(self):
        """Shows the output visualization"""
        # TODO
        pass

    @abstractmethod
    def save(self, path):
        """Saves the visualization to a given path"""
        # TODO
        pass

    def set_mask(mask):
        # TODO
        pass

    @classmethod
    def create_whiteboard(cls):
        return Image.new(cls._image_type, cls._shape, cls._color)
        
    def draw_square(
        self,
        base,
        topleft=(16, 16),
        size=16,
        fill="red",
        outline="lightgrey",
        width=1,
        inner_square_margin=1,
    ):
        """Draws a simple square on a base image.
        
        Args:
            base: A PIL Image instance.
            topleft: The top-left to start the drawing.
            size: Square size.
            fill: Square color.
            outline: Square border color.
            width: Square border width.
            inner_square_margin: Margin size between squares.
        
        Returns:
            The Cartesian coordinate (tuple) of the bottom-right 
            point of the drawing.

        """
        draw = ImageDraw.Draw(base)
        draw.rectangle(
            [topleft, topleft[0] + size, topleft[1] + size], fill, outline, width
        )
        del draw
        return (
            topleft[0] + size + inner_square_margin,
            topleft[1] + size + inner_square_margin,
        )

    def draw_text(self, base, topleft, text, fill="black", position="left"):
        """Draws plain text on a base image.
        
        Args:
            base: A PIL Image instance.
            topleft: The top-left to start the drawing.
            text: Text to be drawn.
            fill: Font color:
            position: Position where the text should be in relation to 
                the top-left coordinate.
        """
        draw = ImageDraw.Draw(base)
        hmargin = 10
        text = str(text)
        textsize = draw.multiline_textsize(text)
        offset = 0
        if position == "right":
            offset = textsize[0] + hmargin
        elif position == "left":
            offset = -(textsize[0] + hmargin)

        draw.multiline_text(
            (topleft[0] + offset, topleft[1]), text, fill=fill, align="left",
        )
        del draw
