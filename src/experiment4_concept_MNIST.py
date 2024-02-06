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
    # Define the path to the experiment results file
    results_file_path = rf"../results/Experiment 4/{config['dataset']}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usinf device: {device}")
    experiments = []
    if dataset == "concept-MNIST":
        for i in range(0, config["max_num_of_new_concepts"]+1):
            print(f"Running with {i} concept(s) ")
            experiment = {
                "num_of_new_concepts": i
            }
            # Load the dataset with i new concepts
            training_data = MNISTDatasetWithConcepts(split = 'train', num_classes = 10, transform=ToTensor(), num_of_new_concepts=i)
            test_data = MNISTDatasetWithConcepts(split = 'test', num_classes = 10, transform=ToTensor(), num_of_new_concepts=i)

            training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=config['batch_size'], shuffle=True)
            test_data_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)
            
            # Train the models on the training data
            print("Training the models on the orginal training data")
            sequential_model = Sequential(i).to(device)
            joint_model = Joint(i).to(device)
            cnn_model = CNN().to(device)
            sequential_trainer = Sequential_Trainer(sequential_model)
            joint_trainer = Joint_Trainer(joint_model)
            cnn_trainer = CNN_Trainer(cnn_model)
            adversarial_trainer = AdversarialTrainer()

            # Training the models
            print("Training the sequential model")
            sequential_trainer.train(training_data_loader, config['n_epochs'])
            print("Training the joint model")
            joint_trainer.train(training_data_loader, config['n_epochs'])
            print("Training the CNN model")
            cnn_trainer.train(training_data_loader, config['n_epochs'])

            # Test the models on the test data
            print("Testing the models on the test data")
            sequential_acc = sequential_trainer.test(test_data_loader)
            joint_acc = joint_trainer.test(test_data_loader)
            cnn_acc = cnn_trainer.test(test_data_loader)
            experiment['Sequential model accuracy on normal images'] = sequential_acc
            experiment['Joint model accuracy on normal images'] = joint_acc
            experiment['CNN model accuracy on normal images'] = cnn_acc

            print(f"Sequential model accuracy: {sequential_acc}")
            print(f"Joint model accuracy: {joint_acc}")
            print(f"CNN model accuracy: {cnn_acc}")

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
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()
    