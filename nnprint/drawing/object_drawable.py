from abc import abstractmethod


class ObjectDrawable:
    """Defines how to fit any visualization into an output image"""

    _image_type = "RGB"
    _shape = (600, 400)  # FIXME set default shape
    _position = (0, 0)  # FIXME set default top-left position
    _color = "white"

    def shape(self, shape):
        """Sets the size of the drawable object"""
        self._shape = shape

    def position(self, position):
        """Sets the position of the drawable object"""
        self._position = position
