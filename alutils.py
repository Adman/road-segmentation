import random


# SAMPLING STRATEGIES ===============================================
def strategy_random(images, k=1, **kwargs):
    return random.sample(images, k)


def strategy_least_uncertainty(images, k=1, **kwargs):
    pass


# STOPPING STRATEGIES ===============================================
def stopping_nostop(model=None, **kwargs):
    return False


def stopping_early_val_loss(model=None, epochs=20, **kwargs):
    """Stop training if val loss has not decreased for some time"""
    best_ep = model.best_val_loss.epoch
    return best_ep < model.epochs_trained - epochs


# EPOCH STRATEGIES ==================================================
def epochs_constant(current, max_epochs=0, **kwargs):
    return current <= max_epochs

