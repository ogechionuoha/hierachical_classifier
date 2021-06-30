import os
import numpy as np
import torch
from torch import nn


class HeirarchicalLabelMap:
    def __init__(self, root_folder, level_names=None):
        self.pathmap = self.get_pathmap(root_folder)
        self.classes_to_ix =  self.get_classes_to_ix(root_folder)
        self.ix_to_classes = {self.classes_to_ix[k]: k for k in self.classes_to_ix}
        self.classes = [k for k in self.classes_to_ix]
        self.levels = self.get_levels(root_folder)
        self.n_classes = sum(self.levels)
        self.child_of_family_ix = self.build_label_tree(root_folder)
        self.family = self.get_family(root_folder)
        
        self.level_names = level_names
        if self.level_names is None:
            self.level_names= [str(i) for i in range(len(self.levels))]

    def get_pathmap(self, path, pathmap = {}, index=0):
        subs = os.walk(path).next()[1]

        if len(subs) > 0:
            pathmap[index] = pathmap.get(index, []) + subs
        
        for sub in subs:
            self.get_pathmap(os.path.join(path,sub), pathmap, index+1)

        return pathmap

    def get_levels(self, path, pathmap = {}, index=0):
        subs = os.walk(path).next()[1]
        if len(subs) > 0:
            pathmap[index] = pathmap.get(index, 0) + len(subs)
            for sub in subs:
                self.get_levels(os.path.join(path,sub), pathmap, index+1)
        return [v for k,v in pathmap.items()]

    def get_family(self, path, pathmap={}, start_index=0):
        pm = self.pathmap
        family = []
        for k, arr in pm.items():
            count = 0
            res = {}
            for v in arr:
                res[v] = count
                count += 1
            family.append(res)
        return family

    def get_classes_to_ix(self, path, pathmap={}, start_index=0):
        pm = self.pathmap
        count = 0
        res = {}
        for k, arr in pm.items():
            for v in arr:
                res[v] = count
                count += 1
        return res
        
    def build_label_tree(self, path, d = {}):
        if os.walk(path).next()[1] == []:
            return os.path.basename(path)
        else:
            return {os.path.basename(path) : [self.build_label_tree(os.path.join(path, x)) for x in os.walk(path).next()[1]]}

    def labels_one_hot(self, class_name):
        indices = self.get_level_labels(class_name)
        levels = self.levels[:-1]
        levels.insert(0,0)
        levels = np.array(levels)
        np.cumsum(levels, axis=0, out=levels)
        indices = indices+ np.array(levels)
        retval = np.zeros(self.n_classes)
        retval[indices] = 1
        return retval

    def get_level_labels(self, class_name):
        path = self.find_path(class_name)
        labels = []
        if path:
            for i,level in enumerate(path[0]):
                labels.append(self.family[i][level])
        return np.array(labels)

    def find_path(self, class_name):
        path = []
        def trail(dct, value, path=()):
            if isinstance(dct, dict):
                for key, val in dct.items():
                    if value in val:
                        yield path + (key, )
            
                for key, lst in dct.items():
                    if isinstance(lst, list):
                        for item in lst:
                            for pth in trail(item, value, path + (key, )):
                                yield pth

        for item in trail(self.child_of_family_ix, class_name):
            path.append(item[1:]+(class_name,))
            break

        return path



class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, labelmap, input_size, level_weights=None):
        torch.nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labelmap = labelmap

        self.level_stop, self.level_start = [], []
        for level_id, level_len in enumerate(self.labelmap.levels):
            if level_id == 0:
                self.level_start.append(0)
                self.level_stop.append(level_len)
            else:
                self.level_start.append(self.level_stop[level_id - 1])
                self.level_stop.append(self.level_stop[level_id - 1] + level_len)

        self.module_dict = {}
        for level_id, level_name in enumerate(self.labelmap.level_names):
            if level_id == 0:
                self.module_dict[level_name] = nn.Linear(input_size, self.labelmap.levels[0])

            # setup linear layer for current nodes which are children of level_id-1
            else:
                child_of_l_1 = getattr(self.labelmap, 'child_of_{}_ix'.format(self.labelmap.level_names[level_id-1]))
                for parent_id in child_of_l_1:
                    self.module_dict['{}_{}'.format(level_name, parent_id)] = nn.Linear(input_size, len(child_of_l_1[parent_id]))

        self.module_dict = nn.ModuleDict(self.module_dict)
        print(self.module_dict)

    def forward(self, x):
        """
        Takes input from the penultimate layer of the model and uses the HierarchicalSoftmax layer in the end to compute
        the logits.
        :param x: <torch.tensor> output of the penultimate layer
        :return: all_log_probs <torch.tensor>, last level log_probs <torch.tensor>
        """
        all_log_probs = torch.zeros((x.shape[0], self.labelmap.n_classes)).to(self.device)

        for level_id, level_name in enumerate(self.labelmap.level_names):
            # print(all_log_probs)
            if level_id == 0:
                # print(level_name)
                # print("saving log probs for: {}:{}".format(self.level_start[0], self.level_stop[0]))
                all_log_probs[:, self.level_start[0]:self.level_stop[0]] = torch.nn.functional.log_softmax(self.module_dict[level_name](x), dim=1)

            # setup linear layer for current nodes which are children of level_id-1
            else:
                child_of_l_1 = getattr(self.labelmap, 'child_of_{}_ix'.format(self.labelmap.level_names[level_id-1]))
                # print(child_of_l_1)
                for parent_id in child_of_l_1:
                    # print('child_of_{}_ix'.format(self.labelmap.level_names[level_id - 1]),
                    #       '{}_{}'.format(level_name, parent_id))
                    # print("saving log probs for: {1} -> {0}".format(self.level_start[level_id] + torch.tensor(child_of_l_1[parent_id]), torch.tensor(child_of_l_1[parent_id])))
                    log_probs = torch.nn.functional.log_softmax(self.module_dict['{}_{}'.format(level_name, parent_id)](x), dim=1)
                    # print("{0} + {1} = {2}".format(log_probs, all_log_probs[:, self.level_start[level_id-1] + parent_id].unsqueeze(1), log_probs + all_log_probs[:, self.level_start[level_id-1] + parent_id].unsqueeze(1)))
                    all_log_probs[:, self.level_start[level_id] + torch.tensor(child_of_l_1[parent_id]).to(self.device)] = log_probs + all_log_probs[:, self.level_start[level_id-1] + parent_id].unsqueeze(1)

        # return only leaf probs
        # print(all_log_probs)
        return all_log_probs, all_log_probs[:, self.level_start[-1]:self.level_stop[-1]]


class HierarchicalSoftmaxLoss(torch.nn.Module):
    def __init__(self, labelmap, level_weights=None):
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.criterion = torch.nn.NLLLoss()

    def forward(self, outputs, labels, level_labels):
        return self.criterion(outputs, level_labels[:, -1])



if __name__ == '__main__':
    root_folder = './data/fmnist'
    imgfoldermap = HeirarchicalLabelMap(root_folder)
    hsoftmax = HierarchicalSoftmax(labelmap=imgfoldermap, input_size=4, level_weights=None)
    penult_layer = torch.tensor([[1, 2, 1, 2.0], [1, 10, -7, 10], [1, 9, 1, -2]])
    print(hsoftmax(penult_layer))
