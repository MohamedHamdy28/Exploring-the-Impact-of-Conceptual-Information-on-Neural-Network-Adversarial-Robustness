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
<<<<<<< HEAD
import os
=======
>>>>>>> 92967c04928c8845205e25cdc2c804511f943e11

def main(dataset="concept-MNIST"):
    # hyperparameters
    config = dict(
        max_num_of_new_concepts = 8,
        batch_size = 64,
        n_epochs = 10,
<<<<<<< HEAD
        n_epochs = 10,
        epsilon = 0.4,
        alpha = 1e-2,
        num_iter = 10,
        num_iter = 10,
=======
        epsilon = 0.4,
        alpha = 1e-2,
        num_iter = 10,
>>>>>>> 92967c04928c8845205e25cdc2c804511f943e11
        y_targ = 2,
        dataset = 'concept-MNIST'
    )
    # Define the path to the experiment results file
    results_file_path = rf"../results/Experiment 4/{config['dataset']}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usinf device: {device}")
    experiments = []
    if dataset == "concept-MNIST":
<<<<<<< HEAD
        for i in range(7, config["max_num_of_new_concepts"]+1):
            print(f"Running with {i} concept(s) ")
=======
        for i in range(config['max_num_of_new_concepts']):
>>>>>>> 92967c04928c8845205e25cdc2c804511f943e11
            experiment = {
                "num_of_new_concepts": i
            }
            # Load the dataset with i new concepts
            training_data = MNISTDatasetWithConcepts(split = 'train', num_classes = 10, transform=ToTensor(), num_of_new_concepts=i)
            test_data = MNISTDatasetWithConcepts(split = 'test', num_classes = 10, transform=ToTensor(), num_of_new_concepts=i)

<<<<<<< HEAD
            training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)
=======
            training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=config["batch_size"], shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True)
>>>>>>> 92967c04928c8845205e25cdc2c804511f943e11
            
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

<<<<<<< HEAD
            # Training the models
            print("Training the sequential model")
            sequential_trainer.train(training_data_loader, config['n_epochs'])
            print("Training the joint model")
            joint_trainer.train(training_data_loader, config['n_epochs'])
            print("Training the CNN model")
            cnn_trainer.train(training_data_loader, config['n_epochs'])
=======
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
>>>>>>> 92967c04928c8845205e25cdc2c804511f943e11

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

<<<<<<< HEAD
            # TODO: save the models
            print(f"Saving the models...")
            torch.save(sequential_model.state_dict(), rf"../models/Experiment 4/Trained on normal data/sequential_model_cons{i}.pt")
            torch.save(joint_model.state_dict(), rf"../models/Experiment 4/Trained on normal data/joint_model_cons{i}.pt")
            torch.save(cnn_model.state_dict(), rf"../models/Experiment 4/Trained on normal data/cnn_model_cons{i}.pt")

            # Loading the saved models
            # print(f"Loading the saved models...")
            # sequential_model.load_state_dict(torch.load(rf"../models/Experiment 4/Trained on normal data/sequential_model_{i}.pt"))
            # joint_model.load_state_dict(torch.load(rf"../models/Experiment 4/Trained on normal data/joint_model_{i}.pt"))
            # cnn_model.load_state_dict(torch.load(rf"../models/Experiment 4/Trained on normal data/cnn_model_{i}.pt"))
            # sequential_trainer = Sequential_Trainer(sequential_model)
            # joint_trainer = Joint_Trainer(joint_model)
            # cnn_trainer = CNN_Trainer(cnn_model)
            
            # Attack the models
            attacker = Attacker(config['batch_size'])
            print("Attacking the models")
            delta_sequential = attacker.pgd_linf_targ(sequential_model, test_data_loader, config["epsilon"], config["alpha"], config["num_iter"], config["y_targ"])
            delta_joint = attacker.pgd_linf_targ(joint_model, test_data_loader, config["epsilon"], config["alpha"], config["num_iter"], config["y_targ"])
            delta_cnn = attacker.pgd_linf_targ(cnn_model, test_data_loader, config["epsilon"], config["alpha"], config["num_iter"], config["y_targ"])
 
            # Testing the models on the adversarail data
            print("Testing the models on the adversarial data")
            sequential_adv_acc = attacker.test_addv(sequential_model, test_data_loader, delta_sequential)
            joint_adv_acc = attacker.test_addv(joint_model, test_data_loader, delta_joint)
            cnn_adv_acc = attacker.test_addv(cnn_model, test_data_loader, delta_cnn)
            
            print(f"Sequential model adversarial accuracy: {sequential_adv_acc}")
            print(f"Joint model adversarial accuracy: {joint_adv_acc}")
            print(f"CNN model adversarial accuracy: {cnn_adv_acc}")
            experiment['Sequential model accuracy on adversarial images'] = sequential_adv_acc
            experiment['Joint model accuracy on adversarial images'] = joint_adv_acc
            experiment['CNN model accuracy on adversarial images'] = cnn_adv_acc

            # Adversarial training for each model
            print("Adversarial training for each model")
            print("Adversarial training for the sequential model")
            robust_sequential_model = adversarial_trainer.train_model(sequential_model, config["num_iter"], training_data_loader, test_data_loader)
            print("Adversarial training for the joint model")
            robust_joint_model = adversarial_trainer.train_model(joint_model, config["num_iter"], training_data_loader, test_data_loader)
            print("Adversarial training for the CNN model")
            robust_cnn_model = adversarial_trainer.train_model(cnn_model, config["num_iter"], training_data_loader, test_data_loader)

            # Test the models on the test data
            print("Testing the models on the test data")
            robust_sequential_acc = sequential_trainer.test(test_data_loader)
            robust_joint_acc = joint_trainer.test(test_data_loader)
            robust_cnn_acc = cnn_trainer.test(test_data_loader)


            print(f"Sequential model accuracy: {robust_sequential_acc}")
            print(f"Joint model accuracy: {robust_joint_acc}")
            print(f"CNN model accuracy: {robust_cnn_acc}")
            experiment['Sequential model accuracy on normal images after adversarial training'] = robust_sequential_acc
            experiment['Joint model accuracy on normal images after adversarial training'] = robust_joint_acc
            experiment['CNN model accuracy on normal images after adversarial training'] = robust_cnn_acc

            # Test the models on the adversarial test data
            print("Testing the models on the adversarial test data")
            robust_sequential_adv_acc = attacker.test_addv(robust_sequential_model, test_data_loader, delta_sequential)
            robust_joint_adv_acc = attacker.test_addv(robust_joint_model, test_data_loader, delta_joint)
            robust_cnn_adv_acc = attacker.test_addv(robust_cnn_model, test_data_loader, delta_cnn)

            print(f"Sequential model adversarial accuracy: {robust_sequential_adv_acc}")
            print(f"Joint model adversarial accuracy: {robust_joint_adv_acc}")
            print(f"CNN model adversarial accuracy: {robust_cnn_adv_acc}")
            experiment['Sequential model accuracy on adversarial images after adversarial training'] = robust_sequential_adv_acc
            experiment['Joint model accuracy on adversarial images after adversarial training'] = robust_joint_adv_acc
            experiment['CNN model accuracy on adversarial images after adversarial training'] = robust_cnn_adv_acc

            # save the robust models
            print(f"Saving the robust models...")
            torch.save(robust_sequential_model.state_dict(), rf"../models/Adversarially trained models/robust_sequential_model_cons{i}.pt")
            torch.save(robust_joint_model.state_dict(), rf"../models/Adversarially trained models/robust_joint_model_cons{i}.pt")
            torch.save(robust_cnn_model.state_dict(), rf"../models/Adversarially trained models/robust_cnn_model_cons{i}.pt")

            # save the results
            experiments.append(experiment)
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
    with open(rf"../results/Experiment 4/{config['dataset']}_config.json", 'w') as f:
=======
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
>>>>>>> 92967c04928c8845205e25cdc2c804511f943e11
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()
    