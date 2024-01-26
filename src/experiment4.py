# In this experiment, we will check how the robustness of the models changes by changing the number of concepts.
from concept_mnist import MNISTDatasetWithConcepts
from torchvision.transforms import ToTensor
from models import Sequential, Joint, CNN
from trainer import Sequential_Trainer, Joint_Trainer, CNN_Trainer
import torch
from attacker import Attacker
from torch.utils.data import DataLoader
from adversarial_trainer import AdversarialTrainer
import pandas as pd
import os

def main(dataset="concept-MNIST"):
    # hyperparameters
    config = dict(
        max_num_of_new_concepts = 8,
        batch_size = 64,
        n_epochs = 10,
        epsilon = 0.4,
        alpha = 1e-2,
        num_iter = 10,
        y_targ = 2,
        dataset = 'concept-MNIST'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = []
    if dataset == "concept-MNIST":
        for i in range(config['max_num_of_new_concepts']):
            experiment = {
                "num_of_new_concepts": i
            }
            # Load the dataset with i new concepts
            training_data = MNISTDatasetWithConcepts(split = 'train', num_classes = 10, transform=ToTensor(), num_of_new_concepts=i)
            test_data = MNISTDatasetWithConcepts(split = 'test', num_classes = 10, transform=ToTensor(), num_of_new_concepts=i)

            training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=config["batch_size"], shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True)
            
            # Model paths
            normal_model_paths = {
                'sequential': rf"../models/Experiment 4/Trained on normal data/sequential_model_{i}.pt",
                'joint': rf"../models/Experiment 4/Trained on normal data/joint_model_{i}.pt",
                'cnn': rf"../models/Experiment 4/Trained on normal data/cnn_model_{i}.pt"
            }
            adv_model_paths = {
                'sequential': rf"../models/Adversarially trained models/robust_sequential_model_{i}.pt",
                'joint': rf"../models/Adversarially trained models/robust_joint_model_{i}.pt",
                'cnn': rf"../models/Adversarially trained models/robust_cnn_model_{i}.pt"
            }

            # Initialize models
            models = {
                'sequential': Sequential(i).to(device),
                'joint': Joint(i).to(device),
                'cnn': CNN().to(device)
            }
            trainers = {
                'sequential': Sequential_Trainer(models['sequential']),
                'joint': Joint_Trainer(models['joint']),
                'cnn': CNN_Trainer(models['cnn'])
            }
            attackers = {
                'sequential': Attacker(models['sequential'], config, device),
                'joint': Attacker(models['joint'], config, device),
                'cnn': Attacker(models['cnn'], config, device)
            }

            # Train or load normal models
            for model_key, model in models.items():
                if os.path.exists(normal_model_paths[model_key]):
                    print(f"{model_key} model found, loading...")
                    model.load_state_dict(torch.load(normal_model_paths[model_key]))
                else:
                    print(f"Training the {model_key} model")
                    trainers[model_key].train(training_data_loader, config["n_epochs"])
                    torch.save(model.state_dict(), normal_model_paths[model_key])

                # Test normal model
                acc = trainers[model_key].test(test_data_loader)
                experiment[f'{model_key} model accuracy on normal images'] = acc
                print(f"{model_key} model accuracy: {acc}")

            # Train or load adversarially trained models
            for model_key, model in models.items():
                if os.path.exists(adv_model_paths[model_key]):
                    print(f"Robust {model_key} model found, loading...")
                    model.load_state_dict(torch.load(adv_model_paths[model_key]))
                else:
                    print(f"Adversarial training for the {model_key} model")
                    attackers[model_key].train(training_data_loader, config["n_epochs"], epsilon=config["epsilon"], alpha=config["alpha"], num_iter=config["num_iter"], y_targ=config["y_targ"])
                    torch.save(model.state_dict(), adv_model_paths[model_key])

                # Test adversarially trained model
                acc = trainers[model_key].test(test_data_loader)
                experiment[f'{model_key} model accuracy on normal images after adversarial training'] = acc
                print(f"{model_key} model accuracy after adversarial training: {acc}")

            # save the results
            experiments.append(experiment)
    # Define the path to the experiment results file
    results_file_path = rf"../results/Experiment 4/{config['dataset']}.csv"

    # Check if the experiments file exists
    if os.path.exists(results_file_path):
        print(f"Experiments file found, loading existing results...")
        # Load the existing experiments
        existing_df = pd.read_csv(results_file_path)
        # Convert the current experiments to a DataFrame
        new_df = pd.DataFrame(experiments)
        # Append the new experiments to the existing ones
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print("No existing experiments file found, creating new results...")
        # Convert the current experiments to a DataFrame
        updated_df = pd.DataFrame(experiments)

    # Save the updated experiments to the CSV file
    updated_df.to_csv(results_file_path, index=False)
    print(f"Updated results saved to {results_file_path}")
    # saving the config file
    import json
    with open(rf"../results/Experiment 4/{config['dataset']}_config.json", 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()
    