import os
import time
import torch
from torch import nn
import torch.utils.data as td
from abc import ABC, abstractmethod
import numpy as np
from torch.autograd import Variable


class NeuralNetwork(nn.Module, ABC):
    """An abstract class representing a neural network.

    All other neural network should subclass it. All subclasses should override
    ``forward``, that makes a prediction for its input argument, and
    ``criterion``, that evaluates the fit between a prediction and a desired
    output. This class inherits from ``nn.Module`` and overloads the method
    ``named_parameters`` such that only parameters that require gradient
    computation are returned. Unlike ``nn.Module``, it also provides a property
    ``device`` that returns the current device in which the network is stored
    (assuming all network parameters are stored on the same device).
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()

    @property
    def device(self):
        # This is important that this is a property and not an attribute as the
        # device may change anytime if the user do ``net.to(newdevice)``.
        return next(self.parameters()).device

    def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def criterion(self, y, d):
        pass


class StatsManager(object):
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, x=None, y=None, d=None):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (float): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
        """
        self.running_loss += loss
        self.number_update += 1

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        return self.running_loss / self.number_update


class Experiment(object):
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        batch_size (integer, optional): the size of the mini batches.
            (default: 16)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """

    def __init__(self, net, train_set, val_set, optimizer, criterion, stats_manager,
                 output_dir=None, batch_size=8, perform_validation_during_training=False):

        # Define data loaders
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        train_loader = td.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                     drop_last=True, pin_memory=True)
        val_loader = td.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True)

        # Initialize history
        history = []

        # Define checkpoint paths
        if output_dir is None:
            output_dir = 'experiment_{}'.format(time.time())
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history)

    def setting(self):
        """Returns the setting of the experiment."""
        return {'Net': self.net,
                'TrainSet': self.train_set,
                'ValSet': self.val_set,
                'Optimizer': self.optimizer,
                'StatsManager': self.stats_manager,
                'BatchSize': self.batch_size,
                'PerformValidationDuringTraining': self.perform_validation_during_training}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Net': self.net.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'History': self.history}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.net.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path, map_location='cuda0')
        self.load_state_dict(checkpoint)
        del checkpoint
        
    def run(self, num_epochs, use_gpu, plot=None):
        learning_rate = 0.001
        num_iter = 0
        best_test_loss = np.inf
        self.stats_manager.init()
        start_epoch = self.epoch
        
        #vis = Visualizer(use_incoming_socket=False)
        if plot is not None:
            plot(self)
        for epoch in range(start_epoch, num_epochs):
            
            self.net.train()
            # if epoch == 1:
            #     learning_rate = 0.0005
            # if epoch == 2:
            #     learning_rate = 0.00075
            # if epoch == 3:
            #     learning_rate = 0.001
            
            if epoch >= 30:
                learning_rate=0.0001
            if epoch >= 40:
                learning_rate=0.00001
            # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
            print('Learning Rate for this epoch: {}'.format(learning_rate))
            
            total_loss = 0.
            
            for i,(images,target) in enumerate(self.train_loader):
                images = Variable(images)
                target = Variable(target)
                
                if use_gpu:
                    images,target = images.cuda(),target.cuda()
                
                pred = self.net(images)
                loss = self.criterion(pred,target)
                total_loss += loss.data
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    self.stats_manager.accumulate(loss.item(), images, pred, target)
                
                if (i+1) % 5 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, len(self.train_loader), loss.data, total_loss / (i+1)))
                    num_iter += 1
                    #vis.plot_train_val(loss_train=total_loss/(i+1))
                    
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
            else:
                self.history.append((self.stats_manager.summarize(), self.evaluate(use_gpu=True)))
                
            if best_test_loss > self.history[epoch][1]['loss']:
                best_test_loss = self.history[epoch][1]['loss']
                print('get best test loss %.5f' % best_test_loss)
                torch.save(self.net.state_dict(),'best_vgg16_bn.pth')
            logfile.writelines(str(epoch) + '\t' + str(self.history[epoch][1]['loss']) + '\n')  
            logfile.flush()      
            torch.save(self.net.state_dict(),'vgg16_bn.pth')
            
            
#             #validation
#             validation_loss = 0.0
#             net.eval()
#             for i,(images,target) in enumerate(test_loader):
#                 with torch.no_grad():
#                     images = Variable(images)
#                     target = Variable(target)
#                 if use_gpu:
#                     images,target = images.cuda(),target.cuda()
                
#                 pred = net(images)
#                 loss = criterion(pred,target)
#                 validation_loss += loss.data
#             validation_loss /= len(test_loader)
#             #vis.plot_train_val(loss_val=validation_loss)
            
#             if best_test_loss > validation_loss:
#                 best_test_loss = validation_loss
#                 print('get best test loss %.5f' % best_test_loss)
#                 torch.save(net.state_dict(),'best.pth')
#             logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
#             logfile.flush()      
#             torch.save(net.state_dict(),'yolo.pth')
            self.save()
            if plot is not None:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))


    def evaluate(self, use_gpu):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        self.stats_manager.init()
        self.net.eval()
        with torch.no_grad():
            for images, target in self.val_loader:
                images = Variable(images)
                target = Variable(target)
                if use_gpu:
                    images,target = images.cuda(),target.cuda()
                y = self.net(images)
                loss = self.criterion(y, target)
                self.stats_manager.accumulate(loss.item(), images, y, target)
        self.net.train()
        return self.stats_manager.summarize()

