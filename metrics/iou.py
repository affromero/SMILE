import torch

def IoU(gt, out):
    from sklearn.metrics import jaccard_similarity_score as jsc
    gt = gt.cpu().numpy().reshape(-1)
    out = out.cpu().numpy().reshape(-1)    
    return jsc(out,gt)

def get_mean_for_iou(images):
    mode_sem = torch.zeros_like(images[0])
    assert mode_sem.size(0)==1, "Only list of tensors with batch=1"
    images = torch.cat(images, dim=0)
    # calculate the semantic mean per pixel
    # among all random outputs
    for i in range(images.size(1)):
        mode_sem[0,i] = images[:,i].mean(0).round()
    return mode_sem
