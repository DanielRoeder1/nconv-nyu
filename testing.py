from utils import create_data_loaders
import torch
import numpy as np
import math
from tqdm import tqdm
import os
import pickle
from scipy.stats import binned_statistic

class arguments:
  def __init__(self):
    self.max_depth = -1
    self.num_samples = 500
    self.evaluate = False
    self.workers = 8
    self.batch_size = 8
    self.modality = "rgbd"
    self.datatype = "nyudepthv2"
    model = "model_best_orb.pth.tar" #@param ["model_best_orb.pth.tar", "model_best_uniform.pth.tar"]
    self.chkpt_path = os.path.join("/content/nconv-nyu", model)
    self.sparsifier = "orb_sampler" #@param ["orb_sampler", "uar"]
    self.data_path = "/content/nyudepthv2" #@param ["/content/nyudepthv2","/content/tum_h5", "/content/tum2_h5","/content/tum3_h5"]
    self.test_depth_variety = False #@param ["False", "True"] {type:"raw"}
    self.save_dir = "/content/nconv-nyu/eval_results"

  def get_name(self):
    if "orb" in self.chkpt_path:
      model_name = "model_orb"
    else:
      model_name = "model_uni"
    
    if "orb" in self.sparsifier:
      sparsifier = "orb"
    else:
      sparsifier = "uni"
    
    if "nyu" in self.data_path:
      data = "nyu"
    else:
      data = "tum"
    
    model_name = f"{model_name}_{sparsifier}_{data}_{self.num_samples}.pickle"
    model_path = os.path.join(self.save_dir, model_name)
    return model_path

class DepthAnalyzer():
  def __init__(self):
    self.bin_edges = [0,1,2,3,4,5,6,7]
    self.gt_depth = []
    self.error = []
    self.save_dir = "/content/sparse-to-dense.pytorch/eval_results"

  def add_data(self,pred, gt):
    mask = gt > 1e-3
    rmse_error = np.absolute(pred[mask] - gt[mask]).flatten()
    gt_depth = gt[mask].flatten()
    self.gt_depth = np.append(self.gt_depth, gt_depth)
    self.error = np.append(self.error, rmse_error)

  def get_results(self, args):
    bin_means, bin_edges, binnumber = binned_statistic(self.gt_depth, self.error, bins=self.bin_edges)
    bin_stds, bin_edges, binnumber = binned_statistic(self.gt_depth, self.error, statistic="std",bins=self.bin_edges)

    bin_idx, counts = np.unique(binnumber, return_counts = True)
    rel_counts = counts / len(self.gt_depth)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    results = {"mean": bin_means, "std": bin_stds, "counts": rel_counts, "center": bin_centers}
    with open(args.get_name() , 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return bin_means, bin_stds, rel_counts, bin_centers 


def rmse(groundtruth, pred):
  mask = groundtruth > 1e-3
  return np.sqrt(np.mean((pred[mask]-groundtruth[mask])**2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_mse = 0
        self.sum_lg10 = 0
        self.sum_absrel = 0
        self.sum_d1 = 0 
        self.sum_d2 = 0 
        self.sum_d3 = 0
        self.count = 0

    def update(self, loss_dict, n=1):
        self.sum_rmse += loss_dict.get("rmse") * n
        self.sum_mae += loss_dict.get("mae") * n
        self.sum_mse += loss_dict.get("mse") * n
        self.sum_lg10 += loss_dict.get("lg10") * n
        self.sum_absrel += loss_dict.get("absrel") * n
        self.sum_d1 += loss_dict.get("d1") * n
        self.sum_d2 += loss_dict.get("d2") * n
        self.sum_d3 += loss_dict.get("d3") * n
        
        self.count += n
    
    def get_average(self):
      avg_rmse = self.sum_rmse / self.count
      avg_mae = self.sum_mae / self.count
      avg_mse = self.sum_mse / self.count
      avg_lg10 = self.sum_lg10 / self.count
      avg_absrel = self.sum_absrel / self.count
      avg_d1 = self.sum_d1 / self.count
      avg_d2 = self.sum_d2 / self.count
      avg_d3 = self.sum_d3 / self.count
      return {"rmse": avg_rmse, "mae": avg_mae, "mse": avg_mse, "lg10": avg_lg10, "absrel": avg_absrel, "d1": avg_d1, "d2": avg_d2, "d3": avg_d3}

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


def evaluate(output, target):
    valid_mask = target> 1e-3
    output = output[valid_mask]
    target = target[valid_mask]

    abs_diff = (output - target).abs()

    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    mae = float(abs_diff.mean())
    lg10 = float((log10(output) - log10(target)).abs().mean())
    absrel = float((abs_diff / target).mean())

    maxRatio = torch.max(output / target, target / output)
    delta1 = float((maxRatio < 1.25).float().mean())
    delta2 = float((maxRatio < 1.25 ** 2).float().mean())
    delta3 = float((maxRatio < 1.25 ** 3).float().mean())
    return {"rmse": rmse, "mae":mae, "mse":mse, "lg10": lg10, "absrel":absrel, "d1":delta1, "d2":delta2, "d3":delta3}



def run_test(net, testloader,args, test_depth = False):
  Avg = AverageMeter()
  d_analyzer = DepthAnalyzer()
  #sum_loss = 0
  net.eval()
  for batch_idx, (input, target) in enumerate(testloader):
    input, target = input.cuda(), target.cuda()
    with torch.no_grad():
      pred = net(input)

    loss_dict = evaluate(pred.data, target.data)
    Avg.update(loss_dict)

    pred = pred.squeeze().cpu().numpy()
    groundtruth = target.squeeze().cpu().numpy()

    if args.test_depth_variety:
      d_analyzer.add_data(pred, groundtruth)
    #loss = rmse(groundtruth, pred)
    #sum_loss += loss

  #print(f"Average RMSE for testset: {sum_loss/len(testloader)}")
  #rmse_new = Avg.get_average().get("rmse")
  #print(f"average from dict: {rmse_new}")
  
  if args.test_depth_variety:
    print(d_analyzer.get_results())
  return Avg.get_average(args)

if __name__ == '__main__':
  args = arguments()
  args.sparsifier = "uar"
  #args.sparsifier = "orb_sampler"
  trainloader, testloader = create_data_loaders(args)

  chkpt_path = "nconv-nyu/model_best_orb.pth.tar"
  checkpoint = torch.load(chkpt_path)
  net = checkpoint['model']
  net.cuda()

  print(f"Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

  loss_test = {"rmse": [], "mae": [], "mse": [], "lg10": [], "absrel": [], "d1": [], "d2": [], "d3": []}

  for n_samples in tqdm([500,400,300,200,100,0]):
    args.num_samples = n_samples
    testloader.dataset.sparsifier.num_samples = n_samples
    loss_dict = run_test(net, testloader, args)
    loss_test.get("rmse").append(loss_dict.get("rmse"))
    loss_test.get("mae").append(loss_dict.get("mae"))
    loss_test.get("mse").append(loss_dict.get("mse"))
    loss_test.get("lg10").append(loss_dict.get("lg10"))
    loss_test.get("absrel").append(loss_dict.get("absrel"))
    loss_test.get("d1").append(loss_dict.get("d1"))
    loss_test.get("d2").append(loss_dict.get("d2"))
    loss_test.get("d3").append(loss_dict.get("d3"))

  print(loss_test)
  print("-")
