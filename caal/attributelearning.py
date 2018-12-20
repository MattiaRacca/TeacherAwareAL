#!/usr/bin/python
from __future__ import division
import numpy as np
import anytree as at
from bidict import bidict
from scipy.stats import beta
from nltk.corpus import wordnet as wn
import random


class TreeNode(at.NodeMixin):
    def __init__(self, wn_synset, parent, tree):
        super(TreeNode, self).__init__()
        self.name = wn_synset.name()
        self.wn_synset = wn_synset
        self.parent = parent
        self.tree = tree

    @property
    def description(self):
        return self.wn_synset.lemma_names()[0].lower()


class Category(TreeNode):
    def __init__(self, wn_synset, parent, tree):
        super(Category, self).__init__(wn_synset, parent, tree)
        self._beta_a = 1
        self._beta_b = 1
        self._mean = beta(self._beta_a, self._beta_b).mean()
        self._entropy = beta(self._beta_a, self._beta_b).entropy()

    @property
    def description(self):
        return super(Category, self).description + ': ' \
            + str(self._beta_a) + '|' + str(self._beta_b)

    @property
    def entropy(self):
    # Computation of entropy is only on demand on the getter
        if self._entropy is None:
            self._entropy = beta(self._beta_a, self._beta_b).entropy()
        return self._entropy

    @property
    def mean(self):
    # Computation of mean is only on demand on the getter
        if self._mean is None:
            self._mean = beta(self._beta_a, self._beta_b).mean()
        return self._mean

    def reset_learning(self):
        self._beta_a = 1
        self._beta_b = 1
        self._mean = None
        self._entropy = None

    def add_positive_example(self):
        self._beta_a += 1
        self._mean = None
        self._entropy = None

    def add_negative_example(self):
        self._beta_b += 1
        self._mean = None
        self._entropy = None


class Entity(TreeNode):
    def __init__(self, wn_synset, parent, tree):
        super(Entity, self).__init__(wn_synset, parent, tree)
        self._theta = 0.5
        self._entropy = None
        self._distances = None

    @property
    def entropy(self):
        """
        # Entropy of the estimate of the node: H(ber(theta))
        # Approximation of a mixture of Bernoulli
        # See Eq. 5
        """
        self._compute_theta()
        self._entropy = entropy_bernoulli(self._theta)
        return self._entropy

    @property
    def theta(self):
        self._compute_theta()
        return self._theta

    @property
    def description(self):
        dist_string = 'dist: '
        for d in self.distances:
            dist_string += "{:.2f}".format(d) + ' | '
        return super(Entity, self).description + ': ' + \
            "{:.2f}".format(self.theta) + ' - ' + dist_string

    @property
    def distances(self):
        if self._distances is None:
            self._compute_distances()
        return self._distances

    def reset_learning(self):
        self._theta = 0.5
        self._entropy = None

    def _compute_distances(self):
        """
        Computes the distance from the Entity to its Categories and stores
        them in a list, from root to immediate parent
        
        Similarity score based on the category tree (lenght of paths)
        'distance' is a bit misleading as these are not distances but
        similarity scores. See Eq. 3 and 6.
        """        
        self._distances = list()
        w = at.Walker()
        paths = [w.walk(self, ancestor) for ancestor in self.ancestors]
        self._distances = [np.exp(-self.tree.similarity_tree_gamma\
            *(len(path[0]) + len(path[2]))) for path in paths]
        self._distances = np.asarray(self.distances)
        self._distances = normalized(self.distances)
        
    def push_information(self, value):
        """
        Push in ancestor line information coming from a positive/negative sample
        """
        for ancestor in self.ancestors:
            if value:
                ancestor.add_positive_example()
            else:
                ancestor.add_negative_example()

    def revert_information_push(self, value):
        """
        Reverts in ancestor line a push of positive/negative information
        """
        for ancestor in self.ancestors:
            if value:
                ancestor._beta_a -= 1
            else:
                ancestor._beta_b -= 1
            # reset ancestor's mean and entropy, forces to recompute them
            ancestor._mean = None
            ancestor._entropy = None

    def _compute_theta(self):
        """
        Go through the ancestors of the leaf, get the current mean and
        do a weighted average of them (based on self.distances)
        See Eq. 3 and 5
        """
        self.estimates = list()
        for ancestor in self.ancestors:
            self.estimates.append(ancestor.mean)
        self.estimates = np.asarray(self.estimates)
        self._theta = np.matmul(self.estimates, self.distances)


def normalized(a, order=0):
    return a / a.sum(order)


def entropy_bernoulli(theta):
    return -theta * np.log2(theta) - (1 - theta) * np.log2(1 - theta)


class CategoryTree:

    def __init__(self, root_wn_name, similarity_tree_gamma=1):
        self.root = Category(wn.synset(root_wn_name), parent=None, tree=self)
        self.node_dictionary = dict()
        self.node_dictionary[self.root.name] = self.root
        self.leaves = None  # list of leaves in the awa order
        self.category_dictionary = dict() # dict of categories
        self.category_dictionary[self.root.name] = self.root
        self.leaves_to_wn = None  # bidict from leaves to wordnet names
        self.similarity_tree_gamma = similarity_tree_gamma

    def add_leaves(self, leaves_list):
        """
        Searches in WN for the leaves and creates a tree that has as branches
        the paths from leaves to the common root (e.g. placental.n.01 for the
        AwA2 dataset). 
        Creates a dictionary of the nodes of tree - key is 
        the WN synset name. Populates a bidict of the leaves, linking
        leaves_list names with WN names.
        
        See Algorithm 1
        """

        root = self.root.wn_synset
        self.leaves_to_wn = bidict()
        self.leaves = leaves_list

        for leaf in leaves_list:
            correct_syn = None
            correct_hyp = None
            """
            Next loop check for all synsets of entity and gives the one that
            has mammal in the hypernym_path.
            Because some lemmas of animal don't refer to the animal itself
            """
            for synset in wn.synsets(leaf):

                hypernym_paths = synset.hypernym_paths()
                hypernym_path_index = -1
                """
                Check all hypernym paths - some synsets have more than one
                """
                for i, hypernym_path in enumerate(hypernym_paths):
                    hyp_contains_root = root in hypernym_path
                    if hyp_contains_root:
                        hypernym_path_index = i
                        correct_syn = synset
                        correct_hyp = hypernym_path
                        break
                """
                If we found a synset that respects our requirements, we stop
                searching
                """
                if correct_syn is not None:
                    break

            if correct_syn is None:
                print('problem with {}: apparently not a {} for any' + \
                    'meaning'.format(leaf, self.root.name))
            else:
                """
                Now that I have the correct hypernym path, I can go through it
                and create the tree
                """
                found_root = False
                previous_node = None
                for hypernym in correct_hyp:
                    if hypernym == root and not found_root:
                        found_root = True
                        previous_node = self.node_dictionary[hypernym.name()]
                        continue
                    if found_root:
                        # Check if the hypernym is already in the tree,
                        # otherwise add it with the correct parent
                        if hypernym.name() not in self.node_dictionary:
                            # if it is a leaf, create an Entity; otherwise a
                            # Category
                            if hypernym == correct_hyp[-1]:
                                node = Entity(hypernym, parent=previous_node,
                                 tree=self)
                            else:
                                node = Category(hypernym, parent=previous_node,
                                 tree=self)
                                self.category_dictionary[node.name] = node
                            self.node_dictionary[node.name] = node
                        previous_node = self.node_dictionary[hypernym.name()]
                # update the bidict
                self.leaves_to_wn[leaf] = previous_node.name
                self.node_dictionary[self.leaves_to_wn[leaf]].\
                    _compute_distances()

    def simplify_tree(self):
        """
        Goes through the tree and gets rid of nodes with only one child.
        Replaced them with their only child.
        Purges not needed Nodes from the node_dictionary
        
        See Algorithm 1 (line 11)
        """
        nodes = [node for node in at.PostOrderIter(self.root)]
        for node in nodes:
            children = node.children
            if len(children) == 1:
                children[0].parent = node.parent
                if node.parent is None:
                    # The current root has only one child - let's replace it
                    # with the only child
                    self.root = children[0]
                node.parent = None
                try:
                    self.node_dictionary.pop(node.name)
                except KeyError:
                    print(node.name)
                try:
                    self.category_dictionary.pop(node.name)
                except KeyError:
                    print(node.name)
        # recompute distances for the entities
        nodes = [node for node in at.PostOrderIter(self.root)]
        for node in nodes:
            if node.is_leaf:
                node._compute_distances()

    def select_greedy_query(self, query_list, standalone=True):
        """
        Gets the available queries (for now only the still available target 
        list) and goes for the one with the lowest entropy.
        query_list contains leaves_name of the available target.
        Returns the AwA index of the selected query
        
        See Learner C
        """
        if standalone: random.shuffle(query_list)
        # compute the entropies
        entropy = [self.node_dictionary[self.leaves_to_wn[entity]].entropy \
                    for entity in query_list]
        # find the highest entropy
        max_index, max_entropy = max(enumerate(entropy), key=lambda p: p[1])

        # find the index in the original entities vector
        awa_index = self.leaves.index(query_list[max_index])

        # remove the question from the list
        if standalone: query_list.remove(query_list[max_index])

        return awa_index, entropy

    def select_closest_query(self, query_list, previous_query, standalone=True):
        """
        Gets the available queries (for now only the still available target 
        list) and goes for the one closest to the PREVIOUS one.
        query_list contains leaves_name of the available target.
        previous_query contains leaves_name of the previous target.
        Returns the AwA index of the selected query.
        
        See Learner M
        """
        if standalone: random.shuffle(query_list)

        if previous_query is None:
            # if there is no previous, just ask the first (shuffled) entry
            index = 0
            distances = [np.exp(-self.similarity_tree_gamma*2) for entity in \
             query_list]
        else:
            ## Compute the similarities
            # Using a walker on the tree structure
            # path[0] is the path from entity to common ancestor
            # path[2] is the path from previous_entity to common ancestor
            w = at.Walker()
            paths = [w.walk(self.node_dictionary[self.leaves_to_wn[entity]], \
                self.node_dictionary[self.leaves_to_wn[self.leaves\
                [previous_query]]]) for entity in query_list]
            distances = [np.exp(-self.similarity_tree_gamma*(len(path[0]) +\
             len(path[2]))) for path in paths]
            index, max_distance = max(enumerate(distances), key=lambda p: p[1])

        # find the index in the original entities vector
        awa_index = self.leaves.index(query_list[index])

        # remove the question from the list
        if standalone: query_list.remove(query_list[index])

        return awa_index, distances
        
    def select_hybrid_query(self, query_list, previous_query, min_D, max_D,\
     phi=0.5):
        """
        Hybrid method: weighted average of the closest and the greedy methods
        
        See Learner H
        """
        
        random.shuffle(query_list)
        # compute the greedy method entropies
        tilde, entropies = self.select_greedy_query(query_list,\
         standalone=False)
        # compute the closest method distances
        tilde, distances = self.select_closest_query(query_list,\
         previous_query, standalone=False)
        
        # feature scaling and inversion for the distances (to use argmax)
        s = np.shape(distances)
        entropies = list(np.array(entropies))       
        
        min_D_exp = np.exp(-self.similarity_tree_gamma*min_D)
        max_D_exp = np.exp(-self.similarity_tree_gamma*max_D)
        distances = list((np.array(distances) - max_D_exp)/\
         (min_D_exp - max_D_exp))
        
        entropies = (phi)*np.array(entropies)
        distances = (1-phi)*np.array(distances)
        
        # compute the scores
        scores = entropies + distances

        # find the highest score
        max_index, max_score = max(enumerate(scores), key=lambda p: p[1])

        # find the index in the original entities vector
        awa_index = self.leaves.index(query_list[max_index])

        # remove the question from the list
        query_list.remove(query_list[max_index])

        return awa_index, scores

    def select_ere_query(self, query_list):
        """
        Gets the available queries (for now only the still available target
        list) and goes for the one with the highest expect reduction in entropy.
        query_list contains leaves_name of the target.
        Returns the AwA index of the selected query.
        
        Not used in the paper. Similar performances of Learner C but slower.
        """
        if standalone: random.shuffle(query_list)
        expected_reduction_entropy = list()
        entropy_true = np.zeros((len(query_list), 1))
        entropy_false = np.zeros((len(query_list), 1))

        for entity in query_list:
            # retrieve current theta
            current_theta = self.node_dictionary[self.leaves_to_wn[entity]].theta

            # simulate for positive answer
            self.node_dictionary[self.leaves_to_wn[entity]].push_information(True)
            # compute the entropies in this case
            for i, key in enumerate(query_list):
                entropy_true[i] = self.node_dictionary[self.leaves_to_wn[key]].entropy
            # revert the info
            self.node_dictionary[self.leaves_to_wn[entity]].\
            revert_information_push(True)

            # simulate for negative answer
            self.node_dictionary[self.leaves_to_wn[entity]].push_information(False)
            # compute the entropies in this case
            for i, key in enumerate(query_list):
                entropy_false[i] = self.node_dictionary[self.leaves_to_wn[key]].entropy
            # revert the info
            self.node_dictionary[self.leaves_to_wn[entity]].\
             revert_information_push(False)

            expected_reduction_entropy.append(current_theta * (np.sum(\
             entropy_true)) + (1 - current_theta) * (np.sum(entropy_false)))

        # find the lowest overall entropy
        min_index, min_entropy = min(enumerate(expected_reduction_entropy), \
                                        key=lambda p: p[1])

        # find the index in the original entities vector
        awa_index = self.leaves.index(query_list[min_index])

        # remove the question from the list
        if standalone: query_list.remove(query_list[min_index])

        return awa_index
        
    def entities_distance(self, ent1, ent2):
        w = at.Walker()
        path = w.walk(self.node_dictionary[self.leaves_to_wn[self.leaves[ent1]]], \
            self.node_dictionary[self.leaves_to_wn[self.leaves[ent2]]])
        distance = len(path[0]) + len(path[2])
        return distance
        
    def reset_learning(self):
        for key in self.node_dictionary:
            self.node_dictionary[key].reset_learning()
