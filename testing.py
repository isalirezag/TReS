

import os
import argparse
import random
import json
import numpy as np
import torch
from args import Configs
import logging
import data_loader



from models import TReS, Net


print('torch version: {}'.format(torch.__version__))


def main(config,device): 
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpunum
    
    folder_path = {
        'live':     config.datapath,
        'csiq':     config.datapath,
        'tid2013':  config.datapath,
        'kadid10k': config.datapath,
        'clive':    config.datapath,
        'koniq':    config.datapath,
        'fblive':   config.datapath,
        }

    img_num = {
        'live':     list(range(0, 29)),
        'csiq':     list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013':  list(range(0, 25)),
        'clive':    list(range(0, 1162)),
        'koniq':    list(range(0, 10073)),
        'fblive':   list(range(0, 39810)),
        }
    

    print('Testing on {} dataset...'.format(config.dataset))
    


    
    SavePath = config.svpath
    svPath = SavePath+ config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/'+'sv'
    os.makedirs(svPath, exist_ok=True)
        
    
    
     # fix the seed if needed for reproducibility
    if config.seed == 0:
        pass
    else:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)


    
    pretrained_path = config.svpath + config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/'+'sv/'
    print('path: {}'.format(pretrained_path))
    path = pretrained_path + 'test_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'
    path2 = pretrained_path + 'train_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'

    with open(path) as json_file:
	    test_index = json.load(json_file)
    with open(path2) as json_file:
	    train_index =json.load(json_file)



   
    test_loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset],
                                             test_index, config.patch_size,
                                             config.test_patch_num, istrain=False)
    test_data = test_loader.get_data()


    solver = TReS(config,device, svPath, folder_path[config.dataset], train_index, test_index,Net)
    version_test_save = 1000
    srcc_computed, plcc_computed = solver.test(test_data,version_test_save,svPath,config.seed,pretrained=1)
    print('srcc_computed {}, plcc_computed {}'.format(srcc_computed, plcc_computed))

        



if __name__ == '__main__':
    
    config = Configs()
    print(config)

    if torch.cuda.is_available():
            if len(config.gpunum)==1:
                device = torch.device("cuda", index=int(config.gpunum))
            else:
                device = torch.device("cpu")
        
    main(config,device)
    