import os
import sys
import aspectlib

class Aspects:
    @aspectlib.Aspect
    def param_validator(*args, **kwargs):
        print(u"Called method with args: {} kwargs: {}".format(args, kwargs))
        yield aspectlib.Proceed

    @aspectlib.Aspect(bind=True)
    def exception_logger(cutpoint, *args):
        try:
            value = yield aspectlib.Proceed
        except Exception as e:
            print(u"Raised exception {} for function {} called with arguments: {}".format(cutpoint.__name__, e, args))
            raise
        else:
            print(u"Returned {} for {}".format(value, args))
            yield aspectlib.Return(value)

    @aspectlib.Aspect
    def result_logger(*args):
        result = yield aspectlib.Proceed
        print(u"[LOGGER] The returned result is: {}".format(result))
        yield aspectlib.Return(result)
