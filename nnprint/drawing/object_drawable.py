from abc import abstractmethod


class ObjectDrawable:
    """Defines how to fit any visualization into an output image"""

    def __init__(self):
        super(ObjectDrawable, self).__init__()
        self._size = (600, 400)  # FIXME set default size
        self._position = (0, 0)  # FIXME set default position

    def size(self, size):
        """Sets the size of the drawable object"""
        self._size = size

    def position(self, position):
        """Sets the position of the drawable object"""
        self._position = position
