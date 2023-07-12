from tempfile import TemporaryDirectory

with TemporaryDirectory() as aaa:
    breakpoint()
    pass


    def __enter__(self) -> DocxReader:
        """Do nothing. The zipfile will open itself when needed.

        :return: self
        """
        return self

    def __exit__(
        self,
        exc_type: Any,  # None | Type[Exception], but py <= 3.9 doesn't like it.
        exc_value: Any,  # None | Exception, but py <= 3.9 doesn't like it.
        exc_traceback: Any,  # None | TracebackType, but py <= 3.9 doesn't like it.
    ):
        """Close the zipfile.

        :param exc_type: Python internal use
        :param exc_value: Python internal use
        :param exc_traceback: Python internal use
        """
        self.close()
