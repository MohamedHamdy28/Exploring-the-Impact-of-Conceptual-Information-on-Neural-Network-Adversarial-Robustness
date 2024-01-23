from concept_mnist import MNISTDatasetWithConcepts
from torchvision.transforms import ToTensor

training_data = MNISTDatasetWithConcepts(split = 'train', num_classes = 10, transform=ToTensor())
test_data = MNISTDatasetWithConcepts(split = 'test', num_classes = 10, transform=ToTensor())    
# printing the label of a sample
print(training_data[0][2])
print(len(training_data))
print([1]+[2])