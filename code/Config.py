import json
import os
import ast
import datetime

class Config(object):
    def __init__(self, filename=None, kwargs=None):
        # GPU id
        self.gpu = '0'

        # Dataset and model setup
        self.dataset = "glue/sst2"
        self.datapath = "processed_data/glue/sst2"
        self.pretrained_weight = "bert-base-uncased"
        self.is_classification = True
        self.seed = 0

        # SSL setup
        self.n_labeled_per_class = 10
        self.unlabeled_per_class = 0

        self.max_seq_length = 256

        # UDA hyperparameters
        self.sharp_temperature = 1
        self.lambda_u = 1

        # Training parameters
        self.batch_size = 4
        self.batch_size_u = 8
        self.test_batch_size = 64
        self.epochs = 10
        self.val_iteration = 100
        self.grad_accumulation_factor = 1

        # Hyperparameters
        self.lr = 1e-5

        self.exp_dir = "None"

        # Augmentation hyperparameters
        self.transform_type = None
        self.transform_times = 1

        self.emb_aug = None

        if filename:
            self.__dict__.update(json.load(open(filename)))
        if kwargs:
            self.update_kwargs(kwargs)

        if filename or kwargs:
            self.update_exp_config()

    def update_kwargs(self, kwargs):
        for (k, v) in kwargs.items():
            try:
                v = ast.literal_eval(v)
            except ValueError:
                v = v
            setattr(self, k, v)

    def update_exp_config(self):
        '''
        Updates the config default values based on parameters passed in from config file
        '''

        base_dir = os.path.join("exp_out", self.dataset, "%d_lbl_%d_unlbl" % (self.n_labeled_per_class, self.unlabeled_per_class), "%s" % self.transform_type)

        if self.emb_aug is not None:
            base_dir = os.path.join(base_dir, self.emb_aug)

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if self.n_labeled_per_class == -1:
            if not os.path.exists(self.exp_dir):
                now = datetime.datetime.now()
                ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                                        now.second)
                self.exp_dir = os.path.join(base_dir, ts)
                if not os.path.exists(self.exp_dir):
                    os.makedirs(self.exp_dir)

        else:
            self.exp_dir = os.path.join(base_dir, "seed_%d" % self.seed)


        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))

        if self.exp_dir is not None:
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores_"+str(self.seed)+".json")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores_"+str(self.seed)+".json")
            self.best_model_file = os.path.join(self.exp_dir, "best_model.pt")

    def to_json(self):
        '''
        Converts parameter values in config to json
        :return: json
        '''
        return json.dumps(self.__dict__, indent=4, sort_keys=True)

    def save_config(self, filename, should_save_gcp=True):
        '''
        Saves the config
        '''
        with open(filename, 'w+') as fout:
            fout.write(self.to_json())
            fout.write('\n')
