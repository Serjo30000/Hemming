import numpy as np

class HemmingNetwork:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights_layer1 = np.zeros((num_classes, input_size))
        self.weights_layer2 = np.zeros((num_classes, num_classes))

    def train(self, inputs):
        self.weights_layer1 = inputs / 2
        epsilon = 1 / self.num_classes
        self.weights_layer2 = np.identity(self.num_classes) - epsilon
        np.fill_diagonal(self.weights_layer2, 1)

    def activation_function(self, s):
        T = self.input_size / 2
        return np.where(s <= 0, 0, np.where(s <= T, s, T))

    def classify(self, x_star, E_max=0.1):
        s_layer1 = np.dot(self.weights_layer1, x_star)
        output_layer1 = self.activation_function(s_layer1)
        while True:
            s_layer2 = np.dot(self.weights_layer2, output_layer1)
            output_layer2 = self.activation_function(s_layer2)
            if np.linalg.norm(output_layer2 - output_layer1) < E_max:
                break
            output_layer1 = output_layer2
        positive_indices = np.argmax(output_layer2)
        if positive_indices is None:
            return None
        else:
            return positive_indices

    @staticmethod
    def conversionToFloat(inputs):
        category_mapping = {"Micro": 0, "Little": 1, "Middle": 2, "Big": 3, "Huge": 4}

        for row in inputs:
            category_column_index = len(row) - 1
            category = row[category_column_index]
            if category in category_mapping:
                row[category_column_index] = category_mapping[category]
            else:
                row[category_column_index] = len(category_mapping)

        inputs = np.array(inputs, dtype=float)
        return inputs

    @staticmethod
    def normalize(inputs):
        min_vals1 = np.min(inputs, axis=0)
        max_vals1 = np.max(inputs, axis=0)
        normalized_data = 2 * (inputs - min_vals1) / (max_vals1 - min_vals1) - 1
        return normalized_data

    @staticmethod
    def merge_similar_arrays(arrays, threshold = 0.2):
        merged_arrays = []
        for array in arrays:
            added = False
            for merged_array in merged_arrays:
                if np.all(np.abs(np.mean(array) - np.mean(merged_array)) < threshold):
                    merged_array = np.vstack([merged_array, array])
                    merged_array = np.mean(merged_array, axis=0)
                    added = True
                    break
            if not added:
                merged_arrays.append(array)
        return np.array(merged_arrays)