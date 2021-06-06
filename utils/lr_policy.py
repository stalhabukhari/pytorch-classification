import numpy as np
# from torch.optim import lr_scheduler

class CosineAnnealLR(object):
    def __init__(self, cycles, total_steps, cycle_decay=None):
        self.steps_per_cycle = np.ceil(total_steps/cycles)
        if cycle_decay:
            self.cycle_decay = cycle_decay
            self.op = self.decayed_cosanneal
        else:
            self.op = self.vanilla_cosanneal

    def __call__(self, step):
        """ base cosine annealing schedule """
        return self.op(step)

    def vanilla_cosanneal(self, step):
        return (np.cos(np.pi * np.mod(step-1, self.steps_per_cycle) \
                /(1. * self.steps_per_cycle)) + 1.) * 0.5

    def decayed_cosanneal(self, step):
        return (np.cos(np.pi * np.mod(step-1, self.steps_per_cycle) \
                /(1. * self.steps_per_cycle)) + 1.) * 0.5 * \
                self.cycle_decay ** np.floor(step/self.steps_per_cycle)


def lr_cosAnnealDecoderwise(_base_lr, batch_cos_cycles, total_train_iterations):
    """ LR values used are multiplied by the initial learning rate previously set. """

    print('Switching LR Schedule!')
    # _base_lr = 5e-3 # will become 5e-4
    # batch_cos_cycles = 15
    cos_decoderwise_iters = total_train_iterations
    _batch_cos_steps_per_cycle = np.ceil(cos_decoderwise_iters/batch_cos_cycles)
    print('[Base] CosineAnnealLR cycle length:', _batch_cos_steps_per_cycle)

    def cosine_anneal_lr(step):
        """ base cosine annealing schedule """
        return (np.cos(np.pi * np.mod(step-1, _batch_cos_steps_per_cycle) \
                /(1. * _batch_cos_steps_per_cycle)) + 1.) * 0.5

    def cos_anneal_branch_a(step):
        """ Decoder A: phase = 0 degrees """
        if (0 <= step) and (step < _batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)
        elif (3*_batch_cos_steps_per_cycle <= step) and (step < 4*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75
        elif (6*_batch_cos_steps_per_cycle <= step) and (step < 7*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**2
        elif (9*_batch_cos_steps_per_cycle <= step) and (step < 10*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**3
        elif (step <= 12*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**4
        else:
            return _base_lr

    def cos_anneal_branch_b(step):
        """ Decoder B: phase = 120 degrees """
        if (_batch_cos_steps_per_cycle <= step) and (step < 2*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)
        elif (4*_batch_cos_steps_per_cycle <= step) and (step < 5*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75
        elif (7*_batch_cos_steps_per_cycle <= step) and (step < 8*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**2
        elif (10*_batch_cos_steps_per_cycle <= step) and (step < 11*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**3
        elif (step <= 12*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**4
        else:
            return _base_lr

    def cos_anneal_branch_c(step):
        """ Decoder C: phase = 240 degrees """
        if (2*_batch_cos_steps_per_cycle <= step) and (step < 3*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)
        elif (5*_batch_cos_steps_per_cycle <= step) and (step < 6*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75
        elif (8*_batch_cos_steps_per_cycle <= step) and (step < 9*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**2
        elif (11*_batch_cos_steps_per_cycle <= step) and (step < 12*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**3
        elif (step <= 12*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**4
        else:
            return _base_lr

    def cos_anneal_stem(step):
        """ Encoder (decay after one main cycle, mean after 4 main cycles) """
        if (0 <= step) and (step < 3*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)
        elif (3*_batch_cos_steps_per_cycle <= step) and (step < 6*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75
        elif (6*_batch_cos_steps_per_cycle <= step) and (step < 9*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**2
        elif (9*_batch_cos_steps_per_cycle <= step) and (step < 12*_batch_cos_steps_per_cycle):
            return cosine_anneal_lr(step)*0.75**3
        else:
            return (cos_anneal_branch_a(step) + \
                    cos_anneal_branch_b(step) + \
                    cos_anneal_branch_c(step)
                    )/3.

    #
    # scheduler = lr_scheduler.LambdaLR(optimizer,
    #                             lr_lambda=[
    #                                 cos_anneal_stem,
    #                                 cos_anneal_branch_a,
    #                                 cos_anneal_branch_b,
    #                                 cos_anneal_branch_c,
    #                             ],
    #                             last_epoch=-1, verbose=False)
    # set_LR_mode('batch')
    # return scheduler
    return [
        cos_anneal_stem,
        cos_anneal_branch_a,
        cos_anneal_branch_b,
        cos_anneal_branch_c,
    ]
