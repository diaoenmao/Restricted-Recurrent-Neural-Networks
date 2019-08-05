
def init():
    global PARAM
    PARAM = {
        'data_name': {'train':'BITS','test':'BITS'},
        'model_name': 'smartcode_cnn',
        'control_name': 'awgn_-1_100_3_3',
        'special_TAG': '',
        'optimizer_name': 'Adam',
        'scheduler_name': 'ReduceLROnPlateau',
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'milestones': [150,250],
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'factor': 0.1,
        'normalize': False,
        'batch_size': {'train':1000,'test':1000},
        'num_workers': 0,
        'data_size': {'train':0,'test':0},
        'device': 'cpu',
        'num_epochs': 100,
        'save_mode': 0,
        'world_size': 1,
        'metric_names': {'train':['ber'],'test':['ber']},
        'topk': 1,
        'init_seed': 0,
        'num_Experiments': 1,
        'tuning_param': {'channel': 1},
        'loss_mode': {'channel':'ce'},
        'normalization': 'bn',
        'activation': 'relu',
        'resume_mode': 0
    }