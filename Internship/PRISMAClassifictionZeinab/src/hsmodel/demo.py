import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config
import numpy as np
import time
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd



def load_tiff_input_and_labels(input_path, train_label_path, test_label_path):
    with rasterio.open(input_path) as src:
        input_data = src.read()
    input_data = np.transpose(input_data, (1, 2, 0))

    with rasterio.open(train_label_path) as src:
        TR = src.read(1)

    with rasterio.open(test_label_path) as src:
        TE = src.read(1)

    return input_data, TR, TE


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['aborashed'], default='aborashed', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='test', help='testing mark')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=300, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band
#-------------------------------------------------------------------------------
# y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()       

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre

def test_epoch(model, test_loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.cuda()
            outputs = model(batch_data)
            preds = outputs.argmax(dim=1)  # faster than topk(1)
            all_preds.append(preds.cpu())

    # Concatenate once at the end — much faster
    return torch.cat(all_preds).numpy()

#-------------------------------------------------------------------------------
# def output_metric(tar, pre):
#     matrix = confusion_matrix(tar, pre)
#     report = classification_report(tar, pre)
#     OA, AA_mean, Kappa, AA = cal_results(matrix)
#     return OA, AA_mean, Kappa, AA, report
# #-------------------------------------------------------------------------------
# def cal_results(matrix):
#     shape = np.shape(matrix)
#     number = 0
#     sum = 0
#     AA = np.zeros([shape[0]], dtype=float)
#     for i in range(shape[0]):
#         number += matrix[i, i]
#         AA[i] = matrix[i, i] / np.sum(matrix[i, :])
#         sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
#     OA = number / np.sum(matrix)
#     AA_mean = np.mean(AA)
#     pe = sum / (np.sum(matrix) ** 2)
#     Kappa = (OA - pe) / (1 - pe)
#     return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------



def output_metric(tar, pre, save_dir):
    """
    Compute and visualize classification metrics, with class names from a shapefile.
    
    Args:
        tar: NumPy array of target (ground truth) labels.
        pre: NumPy array of predicted labels.
        shapefile_path: Path to shapefile containing class names (e.g., 'path/to/classes.shp').
        class_field: Field name in shapefile with class names (e.g., 'class_name').
        save_dir: Directory to save visualization files (default: 'E:/PRISMAClassifiction/results').
    
    Returns:
        OA: Overall accuracy.
        AA_mean: Mean average accuracy.
        Kappa: Cohen's Kappa coefficient.
        AA: Per-class accuracy.
        report: Classification report as a string.
    """

    matrix = confusion_matrix(tar, pre)
    
    # Normalize confusion matrix by row (true labels)
    matrix_normalized = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)  # Percentage per true class
    # Alternative: Normalize by column (predicted labels)
    # matrix_normalized = matrix.astype(float) / matrix.sum(axis=0, keepdims=True) * 100
    # Alternative: Normalize by total
    # matrix_normalized = matrix.astype(float) / matrix.sum() * 100

    
    # Compute classification report
    report = classification_report(tar, pre)
    
    # Compute metrics
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    
    # Display text summary
    print("Overall Accuracy (OA): {:.4f}".format(OA))
    print("Mean Average Accuracy (AA_mean): {:.4f}".format(AA_mean))
    print("Kappa Coefficient: {:.4f}".format(Kappa))
    print("\nPer-Class Accuracy (AA):")
    print("\nClassification Report:\n", report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_normalized, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=config.class_names, yticklabels=config.class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    unique_classes = 11
    # Visualize per-class accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(range(unique_classes), AA, tick_label=config.class_names, color='skyblue')
    plt.title('Per-Class Accuracy (AA)')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(AA):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return OA, AA_mean, Kappa, AA, report

def cal_results(matrix):
    """
    Calculate OA, AA_mean, Kappa, and per-class accuracy from confusion matrix.
    
    Args:
        matrix: Confusion matrix (numpy array).
    
    Returns:
        OA: Overall accuracy.
        AA_mean: Mean average accuracy.
        Kappa: Cohen's Kappa coefficient.
        AA: Per-class accuracy.
    """
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :]) if np.sum(matrix[i, :]) > 0 else 0
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix) if np.sum(matrix) > 0 else 0
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2) if np.sum(matrix) > 0 else 0
    Kappa = (OA - pe) / (1 - pe) if (1 - pe) != 0 else 0
    return OA, AA_mean, Kappa, AA

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False


if args.dataset == 'aborashed':
    input, TR, TE = load_tiff_input_and_labels(
        config.tifpath,
        config.train_mask_path,
        config.test_mask_path
    )
else:
    raise ValueError("Unknown dataset")


# === Load label image and its color map
with rasterio.open(config.label_tif_path) as color_mat:
    label_data = color_mat.read(1)
    cmap = color_mat.colormap(1)  # Get the colormap of band 1


label = TR + TE
num_classes = np.max(TR)

# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
#-------------------------------------------------------------------------------
# load data
x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
Label_train=Data.TensorDataset(x_train,y_train)
x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test=Data.TensorDataset(x_test,y_test)
x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
y_true=torch.from_numpy(y_true).type(torch.LongTensor)
Label_true=Data.TensorDataset(x_true,y_true)

label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

#-------------------------------------------------------------------------------
# create model
model = ViT(
    image_size = args.patches,
    near_band = args.band_patches,
    num_patches = band,
    num_classes = num_classes,
    dim = 64,
    depth = 5,
    heads = 4,
    mlp_dim = 8,
    dropout = 0.1,
    emb_dropout = 0.1,
    mode = args.mode
)
model = model.cuda()
# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)

#-------------------------------------------------------------------------------
if args.flag_test == 'test':
    model_path = os.path.join(config.save_model_path, 'best_model.pt')
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)

     # Call output_metric with shapefile parameters
    # OA2, AA_mean2, Kappa2, AA2, report = output_metric(
    #     tar_v,
    #     pre_v,
    #     save_dir='E:/AboReshead_Data/01 data for train/PRS_L2D_STD_20200725083506_20200725083510_0001/outputs'
    # )
    # OA2, AA_mean2, Kappa2, AA2, report = output_metric(tar_v, pre_v)
    

    # output classification maps
    pre_u = test_epoch(model, label_true_loader)
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1

    # --- Save prediction map as a GeoTIFF file ---
 
    # Open the original input file to get its metadata
    with rasterio.open(config.tifpath) as src:
        profile = src.profile

    # Update the profile for the new single-band output TIFF
    profile.update(
        dtype=rasterio.int16,
        count=1,
        compress='lzw'
    )

    # Write the prediction matrix to a new GeoTIFF file
    with rasterio.open(config.output_tif_path, 'w', **profile) as dst:
        dst.write(prediction_matrix.astype(rasterio.int16), 1)

    print(f"✅ Saved prediction map to: {config.output_tif_path}")

elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    for epoch in range(args.epoches): 
        # scheduler.step()
        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        
        OA2, AA_mean2, Kappa2, AA2, report = output_metric(
            tar_t,
            pre_t,
            save_dir='E:/AboReshead_Data/01 data for train/PRS_L2D_STD_20200725083506_20200725083510_0001/outputs'
        ) 
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                        .format(epoch+1, train_obj, train_acc))
        

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            # Call output_metric with shapefile parameters


    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

            # === Save Model as .pt ===
    os.makedirs(config.save_model_path, exist_ok=True)
    save_path = os.path.join(config.save_model_path, 'best_model.pt')
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to: {save_path}")



print("**************************************************")

OA2, AA_mean2, Kappa2, AA2, report = output_metric(
    tar_v,
    pre_v,
    save_dir=config.base_data_outputs
)

print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")
print(report)
print("**************************************************")
print("Parameter:")

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))

print("**************************************************")
print("**************************************************")
print("End of the program")
print("**************************************************")