import random


# PICK NEXT BATCH STRATEGIES
def strategy_random(images, k=1, **kwargs):
    return random.sample(images, k)


# STOPPING STRATEGIES
def stopping_nostop(history=None, **kwargs):
    return False


# EPOCH STRATEGIES
def epochs_constant(current, max_epochs=0, **kwargs):
    return current <= max_epochs

