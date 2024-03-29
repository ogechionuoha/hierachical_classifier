import os
import numpy as np
import torch
from torch import nn


class HeirarchicalLabelMap:
    def __init__(self, root_folder, level_names=None):
        root_folder = os.path.expanduser(root_folder)
        self.data_folder = root_folder
        self.pathmap = self.get_pathmap(root_folder)
        self.levels = self.get_levels(root_folder)
        self.n_classes = sum(self.levels)
        self.child_of_family_ix = self.build_label_tree(root_folder)
        self.family = self.get_family()
        self.ix_to_family = [{v:k for k,v in group.items()} for group in self.family]
        self.keytrees = self.get_keytrees(root_folder)
        self.level_names = level_names
        self.child_map = self.get_all_children(root_folder)
        self.leaf_class_labels = self.family[len(self.levels)-1]
        if self.level_names is None:
            self.level_names= [str(i) for i in range(len(self.levels))]

    def get_leaf_indices(self, labels=[]):
        indices = [self.leaf_class_labels[i] for i in labels]
        return torch.tensor(indices)

    def get_heirarchies(self, logits):
        assert logits.shape[1] == self.n_classes, 'Labels do not match the total number of classes including hierarchies'
        res_ind = np.zeros((logits.shape[0], len(self.levels))) 
        start = 0
        for i, level in enumerate(self.levels):
            if i == 0:
                l = logits[:,start: level]
            else: l = logits[:,start: level+start]
            arg_max = torch.argmax(l, dim=1)
            res_ind[:, i] = arg_max.cpu().numpy()
            start += level
        return res_ind

    def get_heirarchy_label(self, indices):
        return [self.ix_to_family[i][int(ind)] for i, ind in enumerate(indices)]

    def get_keytrees(self, path, pathmap = {}, index=0):
        subs = next(os.walk(path))[1]
        if len(subs) > 0:
            pathmap[os.path.basename(path)] = len(subs)
            for sub in subs:
                self.get_keytrees(os.path.join(path,sub), pathmap, index+1)
        return pathmap

    def get_pathmap(self, path, pathmap = {}, index=0):
        subs = next(os.walk(path))[1]

        if len(subs) > 0:
            pathmap[index] = pathmap.get(index, []) + subs
        
        for sub in subs:
            self.get_pathmap(os.path.join(path,sub), pathmap, index+1)

        return pathmap

    def get_levels(self, path, pathmap = {}, index=0):
        subs = next(os.walk(path))[1]
        if len(subs) > 0:
            pathmap[index] = pathmap.get(index, 0) + len(subs)
            for sub in subs:
                self.get_levels(os.path.join(path,sub), pathmap, index+1)
        return [v for _,v in pathmap.items()]

    def get_family(self):
        pm = self.pathmap
        family = []
        for _, arr in pm.items():
            count = 0
            res = {}
            for v in arr:
                res[v] = count
                count += 1
            family.append(res)
        return family
        
    def build_label_tree(self, path, d = {}):
        if next(os.walk(path))[1] == []:
            return os.path.basename(path)
        else:
            return {os.path.basename(path) : [self.build_label_tree(os.path.join(path, x)) for x in next(os.walk(path))[1]]}

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

    def get_index(self, class_name):
        family = self.family
        level = 0
        for fam in family:
            level+=1
            if class_name in fam.keys():
                break
            
        ind = family[level-1][class_name]

        return level-1, ind

    
    def get_all_children(self, path, pathmap={}):
        subs = next(os.walk(path))[1]
        if len(subs) > 0:
            pathmap[os.path.basename(path)] = subs
            for sub in subs:
                self.get_all_children(os.path.join(path,sub), pathmap)
        return pathmap
    
    def get_leaf_paths(self):
        return [x[0] for x in os.walk(self.data_folder) if os.path.basename(x[0]) in self.leaf_class_labels]



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
        for level_id in range(len(self.labelmap.levels)):
            if level_id == 0:
                self.module_dict['root'] = nn.Linear(input_size, self.labelmap.levels[0])

            # setup linear layer for current nodes which are children of level_id-1
            else:
                child_of_l_1 = self.labelmap.pathmap[level_id-1]
                for parent in child_of_l_1:
                    nchildren = self.labelmap.keytrees[parent]
                    self.module_dict['{}_{}_{}'.format(self.labelmap.level_names[level_id-1],parent, level_id)] = nn.Linear(input_size, nchildren)

        self.module_dict = nn.ModuleDict(self.module_dict)
        #print(self.module_dict)

    def forward(self, x):
        """
        Takes input from the penultimate layer of the model and uses the HierarchicalSoftmax layer in the end to compute
        the logits.
        :param x: <torch.tensor> output of the penultimate layer
        :return: all_log_probs <torch.tensor>, last level log_probs <torch.tensor>
        """
        
        all_log_probs = torch.zeros((x.shape[0], self.labelmap.n_classes)).to(self.device)

        for level_id in range(len(self.labelmap.levels)):
            # print(all_log_probs)
            if level_id == 0:
                # print(level_name)
                # print("saving log probs for: {}:{}".format(self.level_start[0], self.level_stop[0]))
                all_log_probs[:, self.level_start[0]:self.level_stop[0]] = torch.nn.functional.log_softmax(self.module_dict['root'](x), dim=1)

            # setup linear layer for current nodes which are children of level_id-1
            else:
                child_of_l_1 = self.labelmap.pathmap[level_id-1]
                for parent in child_of_l_1:
                    log_probs = torch.nn.functional.log_softmax(self.module_dict['{}_{}_{}'.format(self.labelmap.level_names[level_id-1], parent, level_id)](x), dim=1)
                    child_indices = [self.labelmap.get_index(child)[1] for child  in self.labelmap.child_map[parent]]
                    #print('{}_{}'.format(parent, level_id))
                    #print('child indices', child_indices)
                    _, par_ind = self.labelmap.get_index(parent)
                    #print(self.level_start[level_id] , torch.tensor([child_indices]).to(self.device), self.level_start[level_id] + torch.tensor([child_indices]).to(self.device))
                    #print('')
                    #print(par_ind)
                    #print(log_probs, all_log_probs[:, self.level_start[level_id-1] + par_ind])
                    all_log_probs[:, self.level_start[level_id] + torch.tensor(child_indices).to(self.device)] = log_probs.to(self.device) + all_log_probs[:, self.level_start[level_id-1] + par_ind] .unsqueeze(1)

        # return only leaf probs
        # print(all_log_probs)
        return all_log_probs, all_log_probs[:, self.level_start[-1]:self.level_stop[-1]]


if __name__ == '__main__':
    root_folder = './data/val'
    imgfoldermap = HeirarchicalLabelMap(root_folder, level_names=['category', 'subcategory', 'item'])    
    hsoftmax = HierarchicalSoftmax(labelmap=imgfoldermap, input_size=4, level_weights=None)
    penult_layer = torch.tensor([[1, 2, 1, 2.0], [1, 10, -7, 10], [1, 9, 1, -2]])
    criterion = torch.nn.NLLLoss()
    labels = imgfoldermap.get_leaf_indices(['Beef-Tomato','Conference', 'Pink-Lady']).to(hsoftmax.device)
    optimizer = torch.optim.Adam(hsoftmax.parameters())
    print('labels', labels)
    for i in range(500):
        res = hsoftmax(penult_layer)
        #res = torch.exp(res[0]), torch.exp(res[1])
        loss = criterion(res[1], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch {i}: loss: {loss.item()}')

    print(imgfoldermap.ix_to_family)
    logits = torch.exp(res[0])
    print(logits)
    h = imgfoldermap.get_heirarchies(logits)
    print(h)
    for l in h:
        print(imgfoldermap.get_heirarchy_label(l))
