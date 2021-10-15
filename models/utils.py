import torch.nn as nn
from misc.utils import PRINT
from torch.nn import init
import math


def print_debug(feed, layers, file=None, append=''):
    if isinstance(feed, (tuple, list)):
        for i in feed:
            PRINT(file, i.size())
    else:
        PRINT(file, feed.size())
    for layer in layers:
        try:
            if isinstance(feed, (tuple, list)):
                feed = layer(*feed)
            else:
                feed = layer(feed)
            if isinstance(feed, (tuple, list)):
                feed_str = feed[0]
            else:
                feed_str = feed
        except BaseException:
            feed = layer(*feed)
            raise BaseException(
                "Type of layer {} not compatible with input {}.".format(
                    layer, feed.shape))
        try:
            _str = '{}, {}'.format(str(layer), feed_str.size())
            # _str = '{}, {}'.format(str(layer).split('(')[0], feed_str.size())
            # _str = '{}, {}'.format(layer.__name__, feed_str.size())
        except AttributeError:
            _str = '{}, {}'.format(layer.__class__.__name__, feed_str.size())
        PRINT(file, _str)
    if append:
        PRINT(file, append)
    PRINT(file, ' ')
    return feed


# ==================================================================#
# Weights Init
# ==================================================================#
class weights_init(object):
    def __init__(self, init_type='kaiming', a=0, nonlinearity='relu'):
        self.a = 0
        self.init_type = init_type
        self.nonlinearity = nonlinearity

    def assign(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'weight'):
                if self.init_type == 'gaussian':
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif self.init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif self.init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data,
                                         a=self.a,
                                         mode='fan_in',
                                         nonlinearity=self.nonlinearity)
                elif self.init_type == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight.data,
                                          a=self.a,
                                          mode='fan_in',
                                          nonlinearity=self.nonlinearity)
                elif self.init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif self.init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(
                        self.init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
