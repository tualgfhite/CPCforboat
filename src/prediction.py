import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cpc")

def prediction_boat( cpc_model, boat_model, device, data_loader,  frame_window):
    logger.info("Starting Evaluation")
    cpc_model.eval() # not training cpc model 
    boat_model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for [data, target] in data_loader:
            data = data.float().unsqueeze(1).to(device) # add channel dimension
            target = target.to(device)
            hidden = cpc_model.init_hidden(len(data))
            output, hidden = cpc_model.predict(data, hidden)
            target = target.view(-1, 1).repeat(1, output.shape[1]).view((-1,))
            data = output.contiguous().view((-1,256))
            output = boat_model.forward(data)
            total_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            total_acc += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(data_loader.dataset)*128 # average loss
    total_acc  /= 1.*len(data_loader.dataset)*128 # average acc

    logger.info("===> Final predictions done. Here is a snippet")
    logger.info('===> Evaluation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))
