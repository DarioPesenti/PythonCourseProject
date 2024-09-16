import torch
import numpy as np
import torch.nn.functional as F

def measure_convexity(embeddings, labels, model, device, num_pairs=10, num_interpolated_points=10):
    # LP: I would recommend at least some minimal documentation for the function,
    # clarifying arguments and at least coarse description of the logic! 
    # Is it something like "I walk from one point to another in the latent space, and count how consistent classifications I get"?


    model.eval()  # Ensuring the model is in evaluation mode
    class_labels = range(10)  # Assuming 10 classes  # LP: maybe this could have been an argument
    class_proportions = {}  # Dictionary to store proportions for each class

    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()  # Convert to numpy array

    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()  # Convert to numpy array

    for class_label in class_labels:
        class_embeddings = embeddings[labels == class_label]
        class_counts = 0

        # Print the number of embeddings found for this class
        #print(f"Class {class_label}: {len(class_embeddings)} embeddings found.")

        for _ in range(num_pairs):
            idxs = np.random.choice(len(class_embeddings), 2, replace=False)
            z1, z2 = class_embeddings[idxs[0]], class_embeddings[idxs[1]]
            interpolation_interval=[(i/num_interpolated_points) for i in range(num_interpolated_points+1)]
            for λ in interpolation_interval: # iterate over λ=0, 0.1, 0.2... 1
                z_prime = (λ * z1) + ((1 - λ) * z2)
                z_prime_tensor = torch.from_numpy(z_prime).float().unsqueeze(0).to(device)
                z_prime_fc1 = F.relu(model.fc1(z_prime_tensor))

                predictions = model.fc2(z_prime_fc1)
                predicted_class = torch.argmax(predictions, dim=1)

                if predicted_class.item() == class_label:
                    class_counts += 1

                #print(f"Interpolation with λ: {λ}, Predicted Class: {predicted_class.item()}, Actual Class: {class_label}")

        proportion = class_counts*100 / (len(interpolation_interval)*num_pairs)
        class_proportions[class_label] = proportion

    # LP: I would recommend return values from the function and print them outside, after you've done the function call.
    # Print the proportions for all classes after processing
    for class_label, proportion in class_proportions.items():
        print(f"Percentage of points correctly predicted as class {class_label}: {proportion}%")