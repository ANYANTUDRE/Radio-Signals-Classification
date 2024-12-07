import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from spec_augment import TimeMask, FreqMask


def print_one_image(dataset, idx):
    row = dataset.iloc[idx]

    image_pixels = np.array(row[:8192], dtype=np.float64)
    label = row.labels

    image = np.resize(image_pixels, (64, 128))  # 64 * 128 = 8192

    plt.imshow(image)
    plt.title(label)
    plt.show()


def get_train_tranform():
    return T.Compose([
        TimeMask(T = 15, num_masks = 4),
        FreqMask(F = 15, num_masks = 4)
    ])


def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))
    
    
def view_classify(img, ps, true_label = None):
    classes = ["squiggle", "narrowband",  "narrowbanddrd", "noise"]

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy()
   
    fig, (ax1, ax2) = plt.subplots(figsize=(12,8), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    if true_label != None:
        ax1.set_title(f'Ground-Truth : {true_label}')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None
