# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this


class CancelFitException(Exception):
    pass


class CancelAllBatchException(Exception):
    pass


class CancelOneBatchException(Exception):
    pass

