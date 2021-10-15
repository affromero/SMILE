from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from munch import Munch
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
# setup plot details

linestyle_tuple = {
    'solid': 'solid',      # Same as (0, ()) or '-'
    'dotted': 'dotted',    # Same as (0, (1, 1)) or '.'
    'dashed': 'dashed',    # Same as '--'
    'dashdot': 'dashdot',  # Same as '-.'    

    'dotted': (0, (1, 1)),
    'densely dotted': (0, (1, 1)),

    'dashed': (0, (5, 5)),
    'densely dashed': (0, (5, 1)),

    'dashdotted': (0, (3, 5, 1, 5)),
    'densely dashdotted': (0, (3, 1, 1, 1)),

    'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
}

colors = ['navy', 'darkorange', 'turquoise', 'red', 'cornflowerblue', 'teal', 'orchid']
# marks = cycle(['+', 'x', 'v', 's', 'D', 'o'])
linestyles = ['dotted', 'dashdot', 'dashed', 'densely dotted', 'densely dashed', 'dashdotted', 'densely dashdotted']
marks = ['+', 'x', '1', '2', '3', '4', '*']

def compute_f1(dict_labels, classes):
    real = dict_labels['real']
    fake = dict_labels['fake']
    attrs = real.keys()
    result = Munch(P=Munch(), R=Munch(), ap=Munch(), f1=Munch())
    # result = Munch(ap=Munch, f1=Munch)
    # PR = Munch(p=Munch(), r=Munch())
    for key in attrs:
        precision = dict()
        recall = dict()
        average_precision = dict()    
        f1 = dict()        
        _real = torch.cat(real[key], dim=0).cpu().numpy()
        _fake = torch.cat(fake[key], dim=0).cpu().numpy()
        # import ipdb; ipdb.set_trace()
        for i in range(_real.shape[1]):
            _precision, _recall, _ = precision_recall_curve(_real[:, i], _fake[:, i])
            _average_precision = average_precision_score(_real[:, i], _fake[:, i])            
            # f1[i] = f1_score(_real[:, i].astype(np.uint8), _fake[:, i], average='weighted')
            _f1 = max(2*_precision*_recall / (_precision+_recall))
            precision[classes[i]] = _precision
            recall[classes[i]] = _recall
            average_precision[classes[i]] = _average_precision
            f1[classes[i]] = _f1
        result['P'][key] = precision
        result['R'][key] = recall
        result['ap'][key] = average_precision
        result['f1'][key] = f1
        #     r=recall,
        #     ap=average_precision,
        #     f1=f1
        # )
    return result

def plot_PR(data_munch, folder_to_save, classes, attr, mask=False):
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))    
    lines.append(l)
    labels.append('iso-f1 curves')
    for i, color, linestyle, marker in zip(classes, colors, linestyles, marks):
        kwargs = {'linestyle': linestyle_tuple[linestyle], 'marker': marker}
        kwargs['markeredgewidth'] = .3
        kwargs['markevery'] = data_munch['P'][i].shape[0]//10 + 1
        l, = plt.plot(data_munch['R'][i], data_munch['P'][i], color=color, **kwargs, lw=1)
        # l, = plt.plot(data_munch['r'][i], data_munch['p'][i], color=color, marker=marker, lw=1)
        lines.append(l)
        labels.append('PR curve for class {} (AP = {:0.2f} - F1 max = {:0.2f})'
                    ''.format(i, data_munch['ap'][i], data_munch['f1'][i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    manipulation = 'removing' if 'NOT_' in attr else 'adding'
    attr_title = attr.replace('NOT_', '')
    plt.title(f'Precision-Recall curve for {manipulation} {attr_title} manipulation.')
    # import ipdb; ipdb.set_trace()
    # plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=10))  
    plt.legend(lines, labels, loc=(0.135, -.375), prop=dict(size=9), framealpha=0.8)
    _mask = '_mask' if mask else ''
    plt.savefig(os.path.join(folder_to_save, f'PRcurve_{attr}{_mask}.png'), dpi=500)