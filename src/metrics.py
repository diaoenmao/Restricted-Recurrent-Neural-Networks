import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.utils.linear_assignment_ import linear_assignment

import config

device = config.PARAM['device']


def SSIM(output, target, window_size=11, MAX=1, window=None, full=False):
    with torch.no_grad():
        MIN = 0
        if isinstance(output, torch.Tensor):
            output = output.expand(target.size())
            (_, channel, height, width) = output.size()
            if window is None:
                valid_size = min(window_size, height, width)
                sigma = 1.5
                mean = [math.exp(-(x - valid_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(valid_size)]
                gauss = torch.Tensor(mean)
                gauss = gauss / gauss.sum()
                _1D_window = gauss.unsqueeze(1)
                _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
                window = _2D_window.expand(channel, 1, valid_size, valid_size).contiguous().to(device)
            mu1 = F.conv2d(output, window, padding=0, groups=channel)
            mu2 = F.conv2d(target, window, padding=0, groups=channel)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(output * output, window, padding=0, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(target * target, window, padding=0, groups=channel) - mu2_sq
            sigma12 = F.conv2d(output * target, window, padding=0, groups=channel) - mu1_mu2
            C1 = (0.01 * (MAX - MIN)) ** 2
            C2 = (0.03 * (MAX - MIN)) ** 2
            ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim[torch.isnan(ssim)] = 1
            ssim = ssim.mean().item()
            if full:
                cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
                cs[torch.isnan(cs)] = 1
                cs = cs.mean().item()
                return ssim, cs
    return ssim


def MSSIM(output, target, window_size=11, MAX=1):
    with torch.no_grad():
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
        levels = weights.size()[0]
        if isinstance(output, torch.Tensor):
            mssim = []
            mcs = []
            for _ in range(levels):
                ssim, cs = SSIM(output, target, window_size=window_size, MAX=MAX, full=True)
                mssim.append(ssim)
                mcs.append(cs)
                output = F.avg_pool2d(output, 2)
                target = F.avg_pool2d(target, 2)
            mssim = torch.tensor(mssim, device=device)
            mcs = torch.tensor(mcs, device=device)
            pow1 = mcs ** weights
            pow2 = mssim ** weights
            mssim = torch.prod(pow1[:-1] * pow2[-1])
            mssim[torch.isnan(mssim)] = 0
            mssim = mssim.item()
        elif isinstance(output, list):
            mssim = 0
            for i in range(len(output)):
                mssim += MSSIM(output[i].unsqueeze(0), target[i].unsqueeze(0), window_size=window_size, MAX=MAX)
            mssim = mssim / len(output)
        else:
            raise ValueError('Not valid output')
    return mssim


def PSNR(output, target, MAX=1.0):
    if isinstance(output, list):
        psnr = 0
        for i in range(len(output)):
            psnr += PSNR(output[i], target[i], MAX)
        psnr /= len(output)
    else:
        with torch.no_grad():
            criterion = nn.MSELoss().to(device)
            mse = criterion(output, target)
            psnr = (20 * torch.log10(torch.tensor(MAX).to(device)) - 10 * torch.log10(mse)).item()
    return psnr


def BPP(nbytes, img):
    if isinstance(img, list):
        bpp = 0
        for i in range(len(img)):
            bpp += BPP(nbytes[i], img[i])
        bpp /= len(img)
    else:
        with torch.no_grad():
            num_pixel = img.numel() / img.size(1)
            bpp = (8 * nbytes / num_pixel)
    return bpp


def BER(output, target):
    with torch.no_grad():
        ber = (output != target).float().mean().item()
    return ber


def BLER(output, target):
    with torch.no_grad():
        bler = (output != target).any(dim=1).float().mean().item()
    return bler


def ACC(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1, 1).expand_as(pred_k)).float().sum()
        acc = (correct_k * (100.0 / batch_size)).item()
    return acc


def Cluster_ACC(output, target, topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        D = max(pred_k.max(), target.max()) + 1
        w = torch.zeros(D, D)
        for i in range(batch_size):
            w[pred_k[i], target[i]] += 1
        ind = linear_assignment(w.numpy().max() - w)
        correct_k = sum([w[i, j] for i, j in ind])
        cluster_acc = (correct_k * (100.0 / batch_size)).item()
    return cluster_acc


class Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history_val = []
        self.history_avg = [0]
        return

    def update(self, new, n=1):
        if isinstance(new, Meter):
            self.val = new.val
            self.avg = new.avg
            self.sum = new.sum
            self.count = new.count
            self.history_val.extend(new.history_val)
            self.history_avg.extend(new.history_avg)
        elif isinstance(new, numbers.Number):
            self.val = new
            self.sum += new * n
            self.count += n
            self.avg = self.sum / self.count
            self.history_val.append(self.val)
            self.history_avg[-1] = self.avg
        else:
            raise ValueError('Not valid update')
        return


class Metric(object):
    batch_metric_names = ['psnr', 'ssim', 'mssim', 'bpp', 'acc']
    full_metric_names = ['cluster_acc']

    def __init__(self, metric_names):
        self.reset(metric_names)

    def reset(self, metric_names):
        self.metric_names = metric_names
        self.if_save = not set(self.metric_names).isdisjoint(self.full_metric_names)
        self.score = None
        self.label = None
        return

    def eval(self, input, output, metric_names):
        evaluation = {'loss': output['loss'].item()}
        if 'channel' in config.PARAM['tuning_param'] and config.PARAM['tuning_param']['channel'] > 0:
            if 'ber' in metric_names:
                evaluation['ber'] = BER(output['channel']['bits'], input['bits'])
            if 'bler' in metric_names:
                evaluation['bler'] = BLER(output['channel']['bits'], input['bits'])
        if 'compression' in config.PARAM['tuning_param'] and config.PARAM['tuning_param']['compression'] > 0:
            if 'psnr' in metric_names:
                evaluation['psnr'] = PSNR(output['compression']['img'], input['img'])
            if 'ssim' in metric_names:
                evaluation['ssim'] = SSIM(output['compression']['img'], input['img'])
            if 'mssim' in metric_names:
                evaluation['mssim'] = MSSIM(output['compression']['img'], input['img'])
            if 'bpp' in metric_names:
                evaluation['bpp'] = BPP(output['compression']['code_size'], input['img'])
        if 'classification' in config.PARAM['tuning_param'] and config.PARAM['tuning_param']['classification'] > 0:
            if self.if_save:
                self.score = torch.cat((self.score, output['classification'].cpu()), 0) if self.score is not None else \
                    output['classification'].cpu()
                self.label = torch.cat((self.label, input['label'].cpu()), 0) if self.label is not None else input[
                    'label'].cpu()
            if 'acc' in metric_names:
                evaluation['acc'] = ACC(output['classification'], input['label'], topk=config.PARAM['topk'])
            if 'cluster_acc' in metric_names:
                evaluation['cluster_acc'] = Cluster_ACC(self.score, self.label, topk=config.PARAM['topk']) if (
                    config.PARAM['activate_full']) else 0
        return evaluation


class Meter_Panel(object):
    def __init__(self, meter_names):
        self.meter_names = meter_names
        self.panel = {k: Meter() for k in meter_names}
        self.metric = Metric(meter_names)

    def reset(self):
        for k in self.panel:
            self.panel[k].reset()
        self.metric.reset(self.meter_names)
        return

    def update(self, new, n=1):
        if isinstance(new, Meter_Panel):
            for i in range(len(new.meter_names)):
                if new.meter_names[i] in self.panel:
                    self.panel[new.meter_names[i]].update(new.panel[new.meter_names[i]])
                else:
                    self.panel[new.meter_names[i]] = new.panel[new.meter_names[i]]
                    self.meter_names += [new.meter_names[i]]
        elif isinstance(new, dict):
            for k in new:
                if k not in self.panel:
                    self.panel[k] = Meter()
                    self.meter_names += [k]
                if isinstance(n, int):
                    self.panel[k].update(new[k], n)
                else:
                    raise ValueError('Not valid n')
        else:
            raise ValueError('Not valid update')
        return

    def eval(self, input, output, metric_names):
        evaluation = self.metric.eval(input, output, metric_names)
        return evaluation

    def summary(self, names):
        fmt_str = ''
        if 'loss' in names and 'loss' in self.panel:
            fmt_str += '\tLoss: {:.4f}'.format(self.panel['loss'].history_avg[-1])
        if 'ber' in names and 'ber' in self.panel:
            fmt_str += '\tBER: {:.4f}'.format(self.panel['ber'].history_avg[-1])
        if 'bler' in names and 'bler' in self.panel:
            fmt_str += '\tBLER: {:.4f}'.format(self.panel['bler'].history_avg[-1])
        if 'bpp' in names and 'bpp' in self.panel:
            fmt_str += '\tBPP: {:.4f}'.format(self.panel['bpp'].history_avg[-1])
        if 'psnr' in names and 'psnr' in self.panel:
            fmt_str += '\tPSNR: {:.4f}'.format(self.panel['psnr'].history_avg[-1])
        if 'ssim' in names and 'ssim' in self.panel:
            fmt_str += '\tSSIM: {:.4f}'.format(self.panel['ssim'].history_avg[-1])
        if 'mssim' in names and 'mssim' in self.panel:
            fmt_str += '\tMSSIM: {:.4f}'.format(self.panel['mssim'].history_avg[-1])
        if 'acc' in names and 'acc' in self.panel:
            fmt_str += '\tACC: {:.4f}'.format(self.panel['acc'].history_avg[-1])
        if 'cluster_acc' in names and 'cluster_acc' in self.panel:
            fmt_str += '\tACC: {:.4f}'.format(self.panel['cluster_acc'].history_avg[-1])
        if 'perplexity' in names and 'perplexity' in self.panel:
            fmt_str += '\tPerplexity: {}'.format(self.panel['perplexity'].history_avg[-1])
        if 'batch_time' in names and 'batch_time' in self.panel:
            fmt_str += '\tBatch Time: {:.4f}'.format(self.panel['batch_time'].history_avg[-1])
        return fmt_str
