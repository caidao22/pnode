class Tee(object):
    # Based on https://gist.github.com/327585 by Anand Kunal
    def __init__(self, outstream, filestream):
        self.outstream = outstream
        self.filestream = filestream
        self.__missing_method_name = None # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        filecallable = getattr(self.filestream, self.__missing_method_name)
        filecallable(*args, **kwargs)

        # Emit method call to stdout
        outcallable = getattr(self.outstream, self.__missing_method_name)
        return outcallable(*args, **kwargs)
