import numpy as np


def calculate_single_recall(retrieved_ids, expected_ids):
    retrieved_ids_set, expected_ids_set = set(retrieved_ids), set(expected_ids)
    found_relevant = len(retrieved_ids_set.intersection(expected_ids_set))
    total_relevant = len(expected_ids_set)
    recall = found_relevant / total_relevant if total_relevant > 0 else 0
    return recall


def calculate_averaged_recall(retrieved_ids_l, expected_ids_l):
    all_recalls = [
        calculate_single_recall(retrieved_ids, expected_ids)
        for retrieved_ids, expected_ids in zip(retrieved_ids_l, expected_ids_l)
    ]
    average_recall = np.mean(all_recalls)
    return average_recall
