import random


# PICK NEXT BATCH STRATEGIES
def strategy_random(images, k=1, **kwargs):
    return random.sample(images, k)


def strategy_least_uncertainty(images, k=1, **kwargs):
    pass


# STOPPING STRATEGIES
def stopping_nostop(histories=None, **kwargs):
    return False


#def stopping_early_val_loss(histories=None, **kwargs):
#    not_decreased = 0
#    start_loss = start_loss
#    for h in reversed(histories):
#        if h.history['val_loss']
#    return


# EPOCH STRATEGIES
def epochs_constant(current, max_epochs=0, **kwargs):
    return current <= max_epochs

