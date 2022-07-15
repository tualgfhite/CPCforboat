import numpy as np
import logging
import torch
import torch.nn.functional as F

## Get the same logger from main"
logger = logging.getLogger("cpc")

def testXXreverse( model, device, data_loader):
    logger.info("Starting test")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for [data, data_r] in data_loader:
            data   = data.float().unsqueeze(1).to(device) # add channel dimension
            data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
            hidden1 = model.init_hidden1(len(data))
            hidden2 = model.init_hidden2(len(data))
            acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)
            total_loss += len(data) * loss 
            total_acc  += len(data) * acc

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> test set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss

def test_boat( cpc_model, boat_model, device, data_loader):
    logger.info("Starting test")
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

    logger.info('===> test set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss

def test(model, device, data_loader):
    logger.info("Starting test")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device) # add channel dimension
            hidden = model.init_hidden(len(data), use_gpu=True)
            acc, loss, hidden = model(data, hidden)
            total_loss += len(data) * loss 
            total_acc  += len(data) * acc

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> test set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss
