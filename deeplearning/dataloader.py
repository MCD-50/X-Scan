"""A Generic Dataset module to build a dataset from a csv file."""
import configargparse
import glob
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchnet as tnt
from skimage import io
import random
from torchvision.transforms import Compose
from image_transforms import transforms as tsfms
import re
import hashlib
import pandas
import torch
import numpy as np


parser = configargparse.get_argument_parser()
parser.add('--train_size', required=True, type=int, default=-1,
           help='size of dataset on which train is to be run.')
parser.add('--val_size', required=True, type=int, default=-1,
           help='size of dataset on which val is to be run.')
parser.add('--train_batch_size', required=True, type=int, default=12,
           help='size of dataset on which train is to be run.')
parser.add('--val_batch_size', required=True, type=int, default=12,
           help='size of dataset on which val is to be run.')
parser.add('--sanity_check', required=True, type=int, default=-1,
           help='sanity check.')
parser.add('--csvfilepath', required=True, help='path to csv file to be used.')
parser.add('--pngpath', required=True,
           help='path to a folder containing all pngs')
parser.add('--maskpath', required=True,
           help='path to a folder containing all masks.')
parser.add('--im_size', required=True, help='image size to be used',
           type=int)
parser.add('--defaultclass', default='others',
           help='default class when none of the classes in buckets are present'
           ' for the image.')
parser.add('--buckets', required=True, help='buckets for the classification. '
           '`,` to separate classes. `:` to separate buckets.')
parser.add('--samplemode', required=False, help='if sampling is to be done '
           'then based on what - classes or buckets.',
           default='class')
parser.add('--sampling', required=True, help='sampling size for each class/'
           'bucket (based on samplemode), -1 if using original data.',
           default=-1)
parser.add('--multilabel', required=True, type=int, default=0,
           help='whether to use multilabel based target or not, default is 0')
parser.add('--multilabelmode', required=False, type=int, default=0,
           help='multilabelmode to be used, 0 - Model '
           'outputs num_class outputs. 1 - Model outputs '
           '2 * num_class outputs.')
parser.add('--cvparam', required=True, type=int,
           help='cvparam to be used to select dataset. -1 if complete dataset'
           ' to be used. 5 fold validation assumed.', default=-1)
parser.add('--num_workers', required=True, type=int, default=8,
           help='Number of workers to be used.')
parser.add('--zeromask_class', required=True, type=int,
           help='Class which will have zero mask.')


def get_train_val_iterators(args):
    """Return train_val iterators along with natural dist of val."""
    print('setting up dataloaders')
    dataset = fpCsvDataSet(args)

    def train_dataloader():
        """Return dataloader for training."""
        dataset.switchmode('train')
        if args.sanity_check != 1:
            dataset.resample()
        return DataLoader(dataset, args.train_batch_size, shuffle=True,
                          num_workers=args.num_workers)

    def val_dataloader():
        """Return dataloader for validation."""
        dataset.switchmode('val')
        return DataLoader(dataset, args.val_batch_size, shuffle=True,
                          num_workers=args.num_workers)

    return {'train': train_dataloader, 'val': val_dataloader}


class fpCsvDataSet(data.Dataset):
    """Filepath CSV dataset based on path input and csv having their description.

    Arguments
    ----------------------------
    csvfilepath(string)
        Path to a csvfile with rows in format: '<filename> tag1|tag2|tag3'.
        tags are '|' separated.
    pngpath(string)
        The path that will be globbed to get a list of all the relevant files
        in the dataset.
    classlist(string)
        A comma separated string that indicates all the classes that should be
        part of the dataset.
        Note:- This classlist is in order of preference, classes that occur
        later are assigned in case of conflict. For Example:- If classlist is
        'normal,abnormal,peffusion,cardiomegaly', then cardiomegaly
        is assigned to a case that has both peffusion and cardiomegaly.
    defaultclass(string)
        class that is to be assigned if a particular filepath exists in csv
        file but doesn't correspond to any of the tags in classlist. Default
        value is 'others'. This needs to be part of classlist.
    buckets(string)
        A string in this format - tag1,tag2:tag3,tag4. A split by ':' should
        return csv strings as tokens. These strings should constitute some
        partition of the class list. No spaces anywhere.
    samplelist(string)
        This argument is used to indicate the relative sampling of classes.
        A split by comma should either return a list of length equal to that of
        the number of buckets or the number of different classes in the
        bucket string. The split can also return a single token.
        Each token should be castable into an integer.
        Each class/bucket will be under/oversampled depending on the
        corresponding token and the number of actual samples of that
        class/bucket in the dataset.
        A single positive token results in all classes/buckets in the bucket
        string being sampled to this number depending on the samplemode.
        If this single token is 0, natural distribution is used.
        If -1, all classes/buckets are sampled to class/bucket average.
        This applies only to the train dataset.
        There shouldn't be any spaces in this string.
    samplemode(string)
        Mentions on what basis sampling must be done, namely, class or bucket.
    cvparam(integer)
        Paths in pngpath are split into 5 parts based on the  md5 hash of
        the file name. An Integer from 0-4 indicates which of these 5 parts
        should be used to draw the validation samples. Defaults to 4.
    multilabel(bool)
        Whether to use multilabel based target or not, default is False.

    """

    def __init__(self, args):
        """Initialize the arguments, assign default values if not passed."""
        self.args = args
        self.csvfilepath = getattr(args, 'csvfilepath',
                                   '/data_nas2/processed/ChestXRays/'
                                   'dataset/NvA/csv/testing.csv')
        self.pngpath = getattr(
            args, 'pngpath',
            '/fast_data2/processed/ChestXRays/all_chest_224/')
        self.maskpath = args.maskpath
        self.defaultclass = getattr(args, 'defaultclass', 'others')
        self.samplelist = getattr(args, 'sampling', '1000,1000,1000')

        self.bucket = getattr(args, 'buckets', 'normal,others:cardiomegaly')
        self.classlist = re.split('[,:]', self.bucket)
        self.bucketlist = [x.split(',') for x in self.bucket.split(':')]
        self.zeromask_class = args.zeromask_class

        self.cvparam = getattr(args, 'cvparam', -1)

        self.samplemode = getattr(args, 'samplemode', 'class')
        assert self.samplemode in ['class', 'bucket'], \
            "Sample mode must be either of class or bucket."

        self.multilabel = getattr(args, 'multilabel', 0)
        self.multilabelmode = getattr(args, 'multilabelmode', 0)

        self.data_mean = 0.5182097657604547
        self.data_std = 0.08537056890509876
        self.fileexts = ['png']
        self.filepath2classmapping = {}
        self.transform = getattr(args, 'transform', None)
        self._setup()

    def __len__(self):
        """Return length of the dataset."""
        if self.mode == 'train':
            return len(self.dataset_tr)
        else:
            return len(self.dataset_val)

    def __getitem__(self, index):
        """Return item at index position."""
        if self.mode == 'train':
            return self.train_transform(self.dataset_tr[index])
        else:
            return self.val_transform(self.dataset_val[index])

    def _setup_transforms(self):
        self.train_transform = Compose([
            tsfms.Resize(int(self.args.im_size*8/7),
                         params={'mode': 'constant'}),
            # tsfms.CenterCrop(self.args.im_size),
            tsfms.RandomCrop(self.args.im_size),
            tsfms.RandomIntensityJitter(0.3, 0.3, 0.3),
            tsfms.Normalize((self.data_mean, self.data_std), mode='meanstd'),
            tsfms.ToTensor(),
            ])
        self.val_transform = Compose([
            tsfms.Resize(self.args.im_size, params={'mode': 'constant'}),
            tsfms.CenterCrop(self.args.im_size),
            tsfms.Normalize((self.data_mean, self.data_std), mode='meanstd'),
            tsfms.ToTensor(),
            ])

    def switchmode(self, mode):
        """Switch between train and val modes."""
        self.mode = mode

    def _setup(self):
        """Call preprocessing functions and _build_dataset."""
        self._preprocess_buckets()
        self._populate_classmapping()
        self._preprocess_sampling()
        self._setup_transforms()
        self._build_dataset()

    def _populate_classmapping(self):
        """Process the csv fiFle and get class mapping for all fileID's in dataset.

        Assigns defaultclass2 for fileID's absent in csvfile and for
        files that exist in csvfile, classmapping is done according to
        the csv. Builds id2filepathsdict.
        id2filepathsdict['cardiomegaly']['train'] is a list of all
        cardiomegaly imagefilepaths in train dataset. Here id is decided based
        upon samplemode.

        Steps:
        1. Read csv file and load as dataframe.
        2. Checks of defaultclass1, defaultclass2 and all classes in bucket.
        3. Drop classes from dataframe which are not being used.
        4. Glob for all the files in the path.
        5. Remove indices from dataframe for which image does not exist.
        6. For rows in dataframe where all columns are 0, set defaultclass1
        as 1
        7. Add images not in csv but exists in pngpath, to the dataframe
        with only defaultclass2
        as 1 and if defaultclass2 not in allclasses then remove files belonging
        to defaultclass2.
        8. Call split dataset with final dataframe and filename2path dict as
        arguments.
        """
        print('processing csv file')
        datadf = pandas.read_csv(self.csvfilepath, header=0, index_col=0)

        all_available_classes = list(datadf)
        assert self.defaultclass not in all_available_classes, \
            "defaultclass already exists in csv."

        all_available_classes += [self.defaultclass]
        allclasses = self.classlist
        for class_name in allclasses:
            assert class_name in all_available_classes, \
                "class {} not in all available classes.".format(class_name)

        print("dropping columns which are not required.")
        datadf.drop([
                col for col in all_available_classes if col in datadf and
                col not in allclasses and col
            ], axis=1, inplace=True)

        all_indices = list(datadf.index)
        all_indices_set = set(all_indices)

        print("globbing {}".format(self.pngpath))
        glob_filepath_list = []
        for ext in self.fileexts:
            glob_filepath_list += glob.glob(
                os.path.join(self.pngpath, '*.'+ext))

        filename2path = {}
        filename_list = []
        for i in glob_filepath_list:
            filename = self._get_filename(i)
            if filename not in filename2path:
                filename2path[filename] = []
            filename2path[filename].append(i)
            filename_list.append(filename)
        filename_set = set(filename_list)

        print("removing unavailable files.")
        unavailable_files = all_indices_set - filename_set
        datadf.drop(unavailable_files, inplace=True)

        if self.defaultclass in allclasses:
            print("handling case for defaultclass - {}.".format(
                self.defaultclass))
            # Tag for images in CSV with all parameters as 0.
            datadf[self.defaultclass] = 0
            datadf_sum = datadf.sum(axis=1)
            zeros_index = np.where(datadf_sum == 0)[0].tolist()
            column_index = datadf.columns.get_loc(self.defaultclass)
            datadf.iloc[zeros_index, column_index] = 1

        print("removing unusable files from filepath list.")
        stale_files_set = filename_set - all_indices_set
        stale_files_list = list(stale_files_set)
        for filename in stale_files_list:
            if filename in filename2path:
                del filename2path[filename]

        self.filename2masks = {}
        for fname in os.listdir(self.maskpath):
            self.filename2masks[os.path.splitext(fname)[0]] = \
                os.path.join(self.maskpath, fname)
        self._split_dataset(datadf, filename2path)

    def _split_dataset(self, datadf, filename2path):
        """Populate id2filepathsdict[classid][trainorval] based on sha1mod5."""
        print("selecting dataset based on sha1 and cvparam.")
        self.id2filepathsdict = {}
        self.filepath2maskpath = {}
        self.multilabeltargets = {}

        for id_name in self.id_name_samplemode:
            self.id2filepathsdict[id_name] = {'train': [], 'val': []}

        datadf = datadf.to_dict()
        for filename, filepath_list in filename2path.items():
            mode = 'val' if self._get_sha1mod5(filename) == self.cvparam else 'train'
            multilabeltarget = None
            if self.multilabel:
                multilabeltarget = [0] * len(self.bucketlist)

            # add file in dataset for all id it exists in.
            for i, id_name in enumerate(self.id_name_samplemode):
                id_list = self.id_list_samplemode[i]
                for class_name in id_list:
                    if datadf[class_name][filename] == 1:
                        for filepath in filepath_list:
                            self.id2filepathsdict[id_name][mode].append(
                                filepath)
                            maskname = '.'.join([filename, class_name])
                            if maskname in self.filename2masks:
                                self.filepath2maskpath[filepath] = \
                                    self.filename2masks[maskname]
                        if self.multilabel:
                            multilabeltarget[self.id_name2target[id_name]] = 1
                        break
            if self.multilabel:
                for filepath in filepath_list:
                    if self.multilabelmode == 1:
                        self.multilabeltargets[filepath] = torch.LongTensor(
                            multilabeltarget)
                    else:
                        self.multilabeltargets[filepath] = torch.FloatTensor(
                            multilabeltarget)

        self.default_image_id_name = self.id_name_samplemode[0]
        self.default_image_fp = self.id2filepathsdict[
            self.default_image_id_name]['train'][0]

    def _preprocess_buckets(self):
        """Create id_name_list and id_name2target based on samplemode."""
        id_name_samplemode = None
        id_list_samplemode = None
        id_name2target = {}
        if self.samplemode == 'class':
            id_name_samplemode = self.classlist
            id_list_samplemode = [[classid] for classid in self.classlist]
            for c in id_name_samplemode:
                for i, bucket in enumerate(self.bucketlist):
                    if c in bucket:
                        id_name2target[c] = i
                        break
        else:
            id_name_samplemode = [','.join(bucket)
                                  for bucket in self.bucketlist]
            id_list_samplemode = self.bucketlist
            for i, id_name in enumerate(id_name_samplemode):
                id_name2target[id_name] = i

        self.id_name_samplemode = id_name_samplemode
        self.id_list_samplemode = id_list_samplemode
        self.id_name2target = id_name2target

        print('id names to target:\n {}'.format(self.id_name2target))

    def _preprocess_sampling(self):
        """Create a list of integers that are to be used by resample method.

        It populates self.sampledist based on self.samplemode, and with a
        list of integers which are per class or per bucket depending on
        the samplemode.
        This should be called before _create_classdatasets and _resample.
        3 cases - sampling classes, sampling buckets, no sampling.
        """
        print('preprocessing sampling parameters')
        sampleweights = self.samplelist.split(',')
        sampleweights = [int(x) for x in sampleweights]

        naturaldist = self._get_natural_dist('train')
        naturalvaldist = self._get_natural_dist('val')

        allids = self.id_name_samplemode

        print('natural distribution of {} :\n {} -> {}\n'.format(
            self.samplemode, allids, naturaldist))
        print('natural distribution in val:\n {} -> {}\n'.format(
            allids, naturalvaldist))

        self.sampledist = []
        if len(sampleweights) == 1:
            if sampleweights[0] == -1:
                # sample classes/buckets to avg
                print('sampling {} to avg across all.'.format(self.samplemode))
                avg = int(sum(naturaldist)/len(naturaldist))
                self.sampledist = [avg for x in naturaldist]
            elif sampleweights[0] == 0:
                # no under/oversampling
                print('using natural distribution')
                self.sampledist = naturaldist
            elif sampleweights[0] > 0:
                # sampling buckets to given token
                print('sampling each {} to {}'.format(
                    self.samplemode, sampleweights[0]))
                self.sampledist = [sampleweights[0] for x in naturaldist]
        elif len(sampleweights) == len(naturaldist):
            print('using {} for sampling {} in {}'.format(
                sampleweights, self.samplemode, self.bucket))
            self.sampledist = sampleweights
        else:
            raise RuntimeError(
                'please recheck sampling params, there should not be any '
                'spaces and number of weights must be 1 or equal to number of '
                'samplemode categories.')

    def _build_dataset(self):
        """Build train and val datasets using list datasets from _create_listdatasets.

        Internally calls resample to build train dataset for the first time.
        Builds val dataset by concatenating the list datasets in
        listdatasets_val.
        """
        # print(self.sampledist)
        listdatasets_tr, listdatasets_val = self._create_listdatasets()
        print(self.sampledist)
        self.shuffledataset_list_tr = [
            tnt.dataset.ShuffleDataset(dataset, weight, True)
            for dataset, weight in zip(listdatasets_tr, self.sampledist)]
        self.dataset_tr = tnt.dataset.ConcatDataset(
            self.shuffledataset_list_tr)
        self.dataset_val = tnt.dataset.ConcatDataset(listdatasets_val)
        if self.args.train_size != -1:
            self.dataset_tr = tnt.dataset.ShuffleDataset(
                self.dataset_tr, self.args.train_size)
        if self.args.val_size != -1:
            self.dataset_val = tnt.dataset.ShuffleDataset(
                self.dataset_val, self.args.val_size)

    def _create_listdatasets(self):
        """Create a list dataset for each class/token in bucket string.

        Concat classes into buckets if sampling mode is 'bucket'.
        populates self.trainlistdatasets and self.vallistdatasets
        that are used by resample method in conjunction with sampledist and
        samplemode set by preprocess_sampling to create the final train and val
        datasets.
        _preprocess_sampling should be called before this method.
        """
        listdatasets_tr = []
        listdatasets_val = []
        for id_name in self.id_name_samplemode:
            train_fp_list = self.id2filepathsdict[id_name]['train']
            listdatasets_tr.append(self._build_listdataset(
                train_fp_list, id_name))
            val_fp_list = self.id2filepathsdict[id_name]['val']
            listdatasets_val.append(self._build_listdataset(
                val_fp_list, id_name))

        return listdatasets_tr, listdatasets_val

    def resample(self):
        """Resample the train dataset.

        To be called at the beginning of every iteration of train dataset
        unless when performing sanity checks.
        """
        for dataset in self.shuffledataset_list_tr:
            dataset.resample()
        self.dataset_tr = tnt.dataset.ConcatDataset(
            self.shuffledataset_list_tr)

        if self.args.train_size != -1:
            self.dataset_tr = tnt.dataset.ShuffleDataset(self.dataset_tr,
                                                         self.args.train_size)

    def _build_listdataset(self, lst, id_name):
        """Build a list dataset given a list and a id_name."""
        def load(path_bucketidtuple):
            path, bucketid = path_bucketidtuple
            try:
                im = io.imread(path)
            except (IOError, OSError):
                print("unable to read: {}".format(path))
                path, bucketid = self.defaultimage_fp, self.defaultimage_bucket
                im = io.imread(path)
                pass
            # because some images are (224,224,4) but rest are (224,224)
            if len(im.shape) == 3:
                im = im[:, :, :2].mean(2)
            mask = np.ones(im.shape) * -1
            if path in self.filepath2maskpath:
                mask = io.imread(self.filepath2maskpath[path]) / 255
            # print(bucketid, self.zeromask_class)
            if bucketid == self.zeromask_class and random.random() < 0.003:
                mask = np.zeros(im.shape)
            return {'input': im, 'target': bucketid, 'filepath': path,
                    'mask_target': mask}

        pathclassidtuplelist = [(x, self.id_name2target[id_name])
                                for x in lst]
        if self.multilabel:
            pathclassidtuplelist = [(x, self.multilabeltargets[x])
                                    for x in lst]

        return tnt.dataset.ListDataset(pathclassidtuplelist, load)

    def _get_filename(self, path):
        """Return a filename given a unix path.

        Splits by underscore, specific to cxr medall case and returns
        first token.
        """
        return os.path.splitext(os.path.basename(path))[0].split('_')[0]

    def _get_sha1mod5(self, s):
        """Return the md5 hash of the string s modulo x.

        x needs to be an integer less than 10
        """
        return int(hashlib.sha1(s.encode()).hexdigest()[-1], 16) % 5

    def _get_natural_dist(self, mode):
        """Return the natural distribution based on mode."""
        return [len(self.id2filepathsdict[id_name][mode])
                for id_name in self.id_name_samplemode]
