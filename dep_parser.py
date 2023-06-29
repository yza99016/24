import copy

class DepInstanceParser():
    def __init__(self, basicDependencies,basicfistDependencies=None,basicsecDependencies=None, tokens=[]):
        self.basicDependencies = basicDependencies
        self.basicfisDependencies = basicfistDependencies
        self.basicsecDependencies = basicsecDependencies
        self.tokens = tokens
        self.words = []
        self.dep_governed_info = []
        self.dep_governed_info_fist = []
        self.dep_parsing()
        self.dep_parsing_fist()

    def dep_parsing(self):
        if len(self.tokens) > 0:
            words = []
            for token in self.tokens:
                words.append(self.change_word(token))
            dep_governed_info = [
                {"word": word}
                for i,word in enumerate(words)
            ]
            self.words = words
        else:
            dep_governed_info = [{}] * len(self.basicDependencies)
        mydep_governed_info=[]

        for dep in self.basicDependencies:
            dependent_index = dep['dependent'] - 1
            governed_index = dep['governor'] - 1
            mydep_governed_info.append({
                "governor": governed_index,
                "dependent":dependent_index,
                "dep": dep['dep']
            })
        self.dep_governed_info = mydep_governed_info

    def dep_parsing_fist(self):
        if len(self.tokens) > 0:
            words = []
            for token in self.tokens:
                words.append(self.change_word(token))
            dep_governed_info = [
                {"word": word}
                for i, word in enumerate(words)
            ]
            self.words = words
        else:
            dep_governed_info = [{}] * len(self.basicfisDependencies)
        mydep_governed_info = []

        for dep in self.basicfisDependencies:
            dependent_index = dep['dependent']-1
            governed_index = dep['governor']-1
            mydep_governed_info.append({
                "dependent": dependent_index,
                "governor": governed_index,
                "dep": dep['dep'],
                "type":dep['type']
            })
        self.dep_governed_info_fist = mydep_governed_info
    def change_word(self, word):
        if "-RRB-" in word:
            return word.replace("-RRB-", ")")
        if "-LRB-" in word:
            return word.replace("-LRB-", "(")
        return word

    def get_init_dep_matrix(self):
        dep_adj_matrix = [[0] * len(self.words) for _ in range(len(self.words))]
        dep_type_matrix = [["none"] * len(self.words) for _ in range(len(self.words))]
        for i in range(len(self.words)):
            dep_adj_matrix[i][i] = 1
            dep_type_matrix[i][i] = "self_loop"
        return dep_adj_matrix, dep_type_matrix
    def get_init_fisr_dep_matrix(self):
        dep_adj_matrix = [[0] * len(self.words) for _ in range(len(self.words))]
        dep_type_matrix = [[["none","none"]] * len(self.words) for _ in range(len(self.words))]
        for i in range(len(self.words)):
            dep_adj_matrix[i][i] = 1
            dep_type_matrix[i][i] = ["self_loop","self_loop"]
        return dep_adj_matrix, dep_type_matrix

    def get_first_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_init_dep_matrix()
        for dep_info in self.dep_governed_info:
            governor = dep_info["governor"]
            dependent = dep_info["dependent"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[dependent][governor] = 1
            dep_adj_matrix[governor][dependent] = 1
            dep_type_matrix[dependent][governor] = dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governor][dependent] = dep_type if direct is False else "{}_out".format(dep_type)

        return dep_adj_matrix, dep_type_matrix

    def get_fist_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_init_fisr_dep_matrix()

        for dep_info in self.dep_governed_info_fist:
            governor = dep_info["governor"]
            dependent=dep_info["dependent"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[dependent][governor] = 1

            if dep_info["type"] == 11and ((dependent!=len(self.words)-1 or governor!=-1)and (dependent!=-1 or governor!=len(self.words)-1)):
              dep_type_matrix[dependent][governor] = dep_type if direct is False else ["{}_out".format(dep_type[0]),
                                                                                       "{}_out".format(dep_type[1])]
            elif dep_info["type"] == 0and ((dependent!=len(self.words)-1 or governor!=-1)and (dependent!=-1 or governor!=len(self.words)-1)):
                dep_type_matrix[dependent][governor] = dep_type if direct is False else ["{}_in".format(dep_type[0]),
                                                                                         "{}_in".format(dep_type[1])]
            elif dep_info["type"] == 1 and ((dependent!=len(self.words)-1 or governor!=-1)and (dependent!=-1 or governor!=len(self.words)-1)):
                dep_type_matrix[dependent][governor] = dep_type if direct is False else ["{}_in".format(dep_type[0]),
                                                                                         "{}_out".format(dep_type[1])]
            elif dep_info["type"] == 10 and ((dependent!=len(self.words)-1 or governor!=-1)and (dependent!=-1 or governor!=len(self.words)-1)):
                dep_type_matrix[dependent][governor] = dep_type if direct is False else ["{}_out".format(dep_type[0]),
                                                                                         "{}_in".format(dep_type[1])]
        return dep_adj_matrix, dep_type_matrix

    def get_next_order(self, dep_adj_matrix, dep_type_matrix):
        new_dep_adj_matrix = copy.deepcopy(dep_adj_matrix)
        new_dep_type_matrix = copy.deepcopy(dep_type_matrix)
        for target_index in range(len(dep_adj_matrix)):
            for first_order_index in range(len(dep_adj_matrix[target_index])):
                if dep_adj_matrix[target_index][first_order_index] == 0:
                    continue
                for second_order_index in range(len(dep_adj_matrix[first_order_index])):
                    if dep_adj_matrix[first_order_index][second_order_index] == 0:
                        continue
                    if second_order_index == target_index:
                        continue
                    if new_dep_adj_matrix[target_index][second_order_index] == 1:
                        continue
                    new_dep_adj_matrix[target_index][second_order_index] = 1
                    new_dep_type_matrix[target_index][second_order_index] = dep_type_matrix[first_order_index][second_order_index]
        return new_dep_adj_matrix, new_dep_type_matrix

    def get_second_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def get_third_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_second_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def search_dep_path(self, start_idx, end_idx, adj_max, dep_path_arr):
        for next_id in range(len(adj_max[start_idx])):
            if next_id in dep_path_arr or adj_max[start_idx][next_id] in ["none"]:
                continue
            if next_id == end_idx:
                return 1, dep_path_arr + [next_id]
            stat, dep_arr = self.search_dep_path(next_id, end_idx, adj_max, dep_path_arr + [next_id])
            if stat == 1:
                return stat, dep_arr
        return 0, []

    def get_dep_path(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[start_index][right_index] = 1
                    dep_path_type_matrix[start_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[end_index][left_index] = 1
                    dep_path_type_matrix[end_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_local_global_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range+end_range:
            for next_id in range(len(first_order_dep_type_matrix[start_index])):
                if first_order_dep_type_matrix[start_index][next_id] in ["none"]:
                    continue
                dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_matrix[start_index][next_id]
                dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_matrix[next_id][start_index]
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[left_index][right_index] = 1
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[right_index][left_index] = 1
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_local_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range+end_range:
            for next_id in range(len(first_order_dep_type_matrix[start_index])):
                if first_order_dep_type_matrix[start_index][next_id] in ["none"]:
                    continue
                dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_matrix[start_index][next_id]
                dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_matrix[next_id][start_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_global_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[left_index][right_index] = 1
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[right_index][left_index] = 1
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix

