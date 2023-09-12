import torch 
from box import ConfigBox

class Metrics:

    def __init__(self) -> None:
        pass
    
    @classmethod
    def precision(cls, y_pred_class, y_true, class_label=1, epsilon=1e-7):
        # Create a binary mask to select examples where the predicted class is equal to the target class
        pred_mask = (y_pred_class == class_label)
        # Calculate the number of examples where both predicted and true class are equal to the target class
        true_positives = ((y_pred_class == y_true) & pred_mask).sum().item()
        # Calculate the total number of examples where the predicted class is equal to the target class
        predicted_positives = pred_mask.sum().item()
        # Calculate precision as the ratio of true positives to predicted positives
        precision = true_positives / (predicted_positives + epsilon)
        return precision


    @classmethod
    def recall(cls, y_pred_class, y_true, class_label=1, epsilon=1e-7):
        # Create a binary mask to select examples where the true class is equal to the target class
        pred_mask = (y_true == class_label)
        # Calculate the number of examples where both predicted and true class are equal to the target class
        true_positives = ((y_pred_class == y_true) & pred_mask).sum().item()
        # Calculate the total number of examples where the true class is equal to the target class
        actual_positives = pred_mask.sum().item()
        # Calculate recall as the ratio of true positives to actual positives
        recall = true_positives / (actual_positives + epsilon)
        return recall
    
    @classmethod
    def f1_score(cls, y_pred_class, y_tru, class_label=1, epsilon=1e-7):
        precision = cls.precision(y_pred_class, y_tru, class_label)
        recall = cls.recall(y_pred_class, y_tru, class_label)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        return f1_score

    @classmethod
    def accuracy(cls, y_pred_class, y_true):
        # Calculate the number of correct predictions
        correct_predictions = (y_pred_class == y_true).sum().item()
        # Calculate the total number of samples
        total_samples = len(y_true)
        # Calculate accuracy as the ratio of correct predictions to total samples
        accuracy = correct_predictions / total_samples
        return accuracy

    @classmethod
    def map(cls, y_pred_class, y_true, class_label, top_k=None):
        # Create a binary mask to select examples where the predicted class or true class is equal to the target class
        relevant_mask = (y_pred_class == class_label) | (y_true == class_label)
        y_pred_scores = torch.zeros_like(y_pred_class, dtype=torch.float32)
        y_pred_scores[relevant_mask] = 1.0
        y_true_binary = (y_true == class_label).float()

        if top_k is not None:
            if top_k > torch.sum(relevant_mask).item():
                top_k = torch.sum(relevant_mask).item()
            y_pred_scores = y_pred_scores[relevant_mask]  # Filter relevant scores
            sorted_indices = torch.argsort(y_pred_scores, descending=True)
            y_true_binary = y_true_binary[sorted_indices]

        num_relevant = torch.sum(y_true_binary).item()
        if num_relevant == 0:
            return 0.0

        precision_at_i = []
        num_retrieved = 0
        for i in range(len(y_true_binary)):
            if y_true_binary[i].item() == 1:
                num_retrieved += 1
                precision_at_i.append(num_retrieved / (i + 1))

        return sum(precision_at_i) / num_relevant

    @classmethod
    def calculate_metrics_macro(cls, y_pred_class, y_true, num_classes):
        metrics = {}

        precision = 0
        recall = 0
        f1_score = 0
        mAP50 = 0
        mAP90 = 0

        accuracy_value = cls.accuracy(y_pred_class, y_true)
        
        for class_label in range(num_classes):
            class_precision = cls.precision(y_pred_class, y_true, class_label)
            class_recall = cls.recall(y_pred_class, y_true, class_label)
            class_f1_score = cls.f1_score(y_pred_class, y_true, class_label)
            class_mAP50 = cls.map(y_pred_class, y_true, class_label, top_k=50)
            class_mAP90 = cls.map(y_pred_class, y_true, class_label, top_k=90)

            precision += class_precision
            recall += class_recall
            f1_score += class_f1_score
            mAP50 += class_mAP50
            mAP90 += class_mAP90

        precision /= num_classes
        recall /= num_classes
        f1_score /= num_classes
        mAP50 /= num_classes
        mAP90 /= num_classes

        metrics['accuracy'] = accuracy_value
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
        metrics['mAP50'] = mAP50
        metrics['mAP90'] = mAP90

        return ConfigBox(metrics)