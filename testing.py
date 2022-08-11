from main import create_data_loaders
import torch
import numpy as np

class arguments:
  def __init__(self):
    self.max_depth = -1
    self.num_samples = 500
    self.evaluate = False
    self.sparsifier = "orb_sampler"
    self.workers = 8
    self.data = 'nyudepthv2'
    self.batch_size = 8
    self.modality = "rgbd"


def rmse(groundtruth, pred):
  mask = groundtruth > 1e-3
  return np.sqrt(np.mean((pred[mask]-groundtruth[mask])**2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_test():
  Avg = AverageMeter()
  sum_loss = 0
  net.eval()
  for batch_idx, (input, target) in enumerate(testloader):
    input, target = input.cuda(), target.cuda()
    with torch.no_grad():
      pred = net(input)

    pred = pred.squeeze().cpu().numpy()
    groundtruth = target.squeeze().cpu().numpy()

    loss = rmse(groundtruth, pred)
    Avg.update(loss,1)
    sum_loss += loss

  
  print(f"Average loss for testset: {sum_loss/len(testloader)}")
  print(Avg.avg)
  return Avg.avg

if __name__ == '__main__':
  args = arguments()
  trainloader, testloader = create_data_loaders(args)

  chkpt_path = "nconv-nyu/model_best_orb.pth.tar"
  checkpoint = torch.load(chkpt_path)
  net = checkpoint['model']
  net.cuda()

  print(f"Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

  loss_test = []
  for n_samples in [500,400,300,200,100,0]:
    testloader.dataset.sparsifier.num_samples = n_samples
    loss_test.append(run_test())
  print(loss_test)

