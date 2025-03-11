import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]

# Training data transformations (with data augmentation)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Validation and test transformations (NO augmentation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_dataset = datasets.CIFAR10(root="./data", train=True, download=True)


train_size = int(0.9 * len(full_dataset))
val_size   = len(full_dataset) - train_size

# Shuffle indices and split
all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)
train_indices = all_indices[:train_size]
val_indices   = all_indices[train_size:]

# Build Subsets that apply different transforms
train_subset = Subset(
    datasets.CIFAR10(root="./data", train=True, transform=train_transform),
    train_indices
)
val_subset = Subset(
    datasets.CIFAR10(root="./data", train=True, transform=val_transform),
    val_indices
)

# Build train/val loaders
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=64, shuffle=False)

# Test set
test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class DynamicCNN:
    def __init__(self, layers_config, optimizer_type='sgd', lr=0.01):
        """
        layers_config: list of tuples specifying layers.
            e.g. [("conv", out_channels, kernel_size, stride), ("pool", kernel_size, stride), ...]
        optimizer_type: 'sgd' or 'adam'
        lr: learning rate
        """
        self.layers_config = layers_config  # architecture blueprint
        self.optimizer_type = optimizer_type
        self.lr = lr

        # Build the PyTorch model from the config
        self.model = self._build_model().to(device)
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # A simple LR scheduler (StepLR) for demonstration
        # Decreases LR by a factor of 0.1 every 10 "epochs"
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _build_model(self):
        """Construct a Sequential model from the layers_config."""
        layers = nn.ModuleList()
        in_channels = 3  # CIFAR-10: 3 color channels

        for layer in self.layers_config:
            if layer[0] == "conv":
                _, out_channels, kernel, stride = layer
                # Use padding = kernel//2 to keep dimensions consistent
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                 stride=stride, padding=kernel // 2)
                layers.append(conv)
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels

            elif layer[0] == "pool":
                # e.g., ("pool", 2, 2) for 2x2 with stride=2
                _, kernel, stride = layer
                layers.append(nn.MaxPool2d(kernel_size=kernel, stride=stride))

            elif layer[0] == "dense":
                # A dense layer config ("dense", num_outputs)
                # We'll flatten and create a linear layer
                _, num_outputs = layer
                layers.append(nn.AdaptiveAvgPool2d((1,1)))
                layers.append(nn.Flatten())
                layers.append(nn.Linear(in_channels, num_outputs))

            else:
                raise ValueError(f"Unknown layer type: {layer[0]}")


        if not self.layers_config or self.layers_config[-1][0] != "dense":
            layers.append(nn.AdaptiveAvgPool2d((1,1)))
            layers.append(nn.Flatten())
            layers.append(nn.Linear(in_channels, 10))

        return nn.Sequential(*layers)

    def _build_optimizer(self):
        """Construct the optimizer based on optimizer_type and lr."""
        if self.optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def train_one_epoch(self, data_loader, max_batches=None):
        """
        Train the model for one epoch, or up to max_batches (to reduce training cost).
        """
        self.model.train()
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                # Stop early if we've hit the max batch limit
                break

        # Step the scheduler after an "epoch"
        self.scheduler.step()

    def evaluate(self, data_loader):
        """Evaluate current model on given data_loader, return accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def clone(self):
        """Create a deep copy of this DynamicCNN (including current model weights)."""
        new_clone = DynamicCNN(list(self.layers_config), self.optimizer_type, self.lr)
        
        # Copy state dict from current model
        new_clone.model.load_state_dict(self.model.state_dict())
        
        # Rebuild optimizer for the new clone
        new_clone.optimizer = new_clone._build_optimizer()
        # Copy the scheduler state as well (optional; you might reset it)
        new_clone.scheduler = optim.lr_scheduler.StepLR(new_clone.optimizer, step_size=10, gamma=0.1)
        
        return new_clone


def partial_copy_conv_weights(child_layer, parent_layer):
    """
    A simple demonstration function that copies overlapping channels
    between child_layer and parent_layer (both nn.Conv2d).
    If the child has more channels, we leave the extra random.
    If it has fewer, we only copy a subset from the parent.
    """
    with torch.no_grad():
        # Child parameters
        child_weight = child_layer.weight
        child_bias = child_layer.bias

        # Parent parameters
        parent_weight = parent_layer.weight
        parent_bias = parent_layer.bias

        # Overlapping channels
        out_channels_child, in_channels_child, kh_child, kw_child = child_weight.shape
        out_channels_parent, in_channels_parent, kh_parent, kw_parent = parent_weight.shape

        min_out = min(out_channels_child, out_channels_parent)
        min_in = min(in_channels_child, in_channels_parent)
        min_kh = min(kh_child, kh_parent)
        min_kw = min(kw_child, kw_parent)

        # Copy the overlapping weights
        child_weight[:min_out, :min_in, :min_kh, :min_kw] = \
            parent_weight[:min_out, :min_in, :min_kh, :min_kw]

        # Copy the overlapping bias
        if child_bias is not None and parent_bias is not None:
            child_bias[:min_out] = parent_bias[:min_out]

def copy_partial_weights_from_parent(child, parent):
    """
    Attempt a more careful copy of weights for matching layers.
    If both child & parent have a nn.Conv2d in the same sequential position,
    we do partial channel copying instead of skipping if shapes differ.
    """
    parent_layers = list(parent.model.children())
    child_layers  = list(child.model.children())

    with torch.no_grad():
        # We iterate over matching indices
        for idx, (p_layer, c_layer) in enumerate(zip(parent_layers, child_layers)):
            # If both are Conv2d, do partial copy
            if isinstance(p_layer, nn.Conv2d) and isinstance(c_layer, nn.Conv2d):
                partial_copy_conv_weights(c_layer, p_layer)
            # If shapes match exactly (including linear layers, etc.), do a direct copy
            elif (
                hasattr(c_layer, "weight") and 
                hasattr(p_layer, "weight") and
                c_layer.weight.shape == p_layer.weight.shape
            ):
                c_layer.weight.copy_(p_layer.weight.data)
                if c_layer.bias is not None and p_layer.bias is not None:
                    c_layer.bias.copy_(p_layer.bias.data)


def mutate_model(parent: DynamicCNN):
    """Return a mutated child of the given parent (with partial weight inheritance)."""
    import random

    # Create a clone
    child = parent.clone()

    mutations = ["add_layer", "remove_layer", "modify_layer", "optim_lr"]
    mutation = random.choice(mutations)

    if mutation == "add_layer":
        layer_type = random.choice(["conv", "pool"])
        insert_pos = random.randint(0, len(child.layers_config))

        if layer_type == "conv":
            out_channels = random.choice([16, 32, 64, 128])
            kernel = random.choice([3, 3, 5]) 
            stride = random.choice([1, 1, 2])  
            new_layer = ("conv", out_channels, kernel, stride)
        else:
            # pool
            pool_size = 2
            new_layer = ("pool", pool_size, pool_size)

        child.layers_config.insert(insert_pos, new_layer)
        child.model = child._build_model().to(device)

        # More advanced partial copy
        copy_partial_weights_from_parent(child, parent)

    elif mutation == "remove_layer":
        # Remove a layer if there's more than 1
        if len(child.layers_config) > 1:
            remove_idx = random.randint(0, len(child.layers_config) - 1)
            # Avoid removing final dense if it exists
            if (child.layers_config[remove_idx][0] == "dense" and 
                remove_idx == len(child.layers_config) - 1):
                # fallback: remove a different layer if possible
                if remove_idx > 0:
                    remove_idx -= 1

            _ = child.layers_config.pop(remove_idx)

            child.model = child._build_model().to(device)
            copy_partial_weights_from_parent(child, parent)

    elif mutation == "modify_layer":
        # Modify a random conv layer's out_channels
        conv_indices = [i for i, l in enumerate(child.layers_config) if l[0] == "conv"]
        if conv_indices:
            idx = random.choice(conv_indices)
            layer_type, out_ch, kernel, stride = child.layers_config[idx]

            # We do a small tweak or random pick
            if out_ch < 128 and random.random() < 0.5:
                new_out = out_ch * 2
            elif out_ch > 16 and random.random() < 0.5:
                new_out = out_ch // 2
            else:
                new_out = random.choice([16, 32, 64, 128])

            child.layers_config[idx] = (layer_type, new_out, kernel, stride)
            child.model = child._build_model().to(device)
            copy_partial_weights_from_parent(child, parent)

    else:  # "optim_lr"
        # Either adjust LR or switch optimizer
        if random.random() < 0.5:
            # random LR factor
            new_lr = child.lr * random.choice([0.5, 2.0])
            new_lr = max(1e-5, min(1.0, new_lr))  # clamp
            child.lr = new_lr
        else:
            child.optimizer_type = "adam" if child.optimizer_type == "sgd" else "sgd"

        child.optimizer = child._build_optimizer()
        # Also re-init scheduler
        child.scheduler = optim.lr_scheduler.StepLR(child.optimizer, step_size=10, gamma=0.1)

    return child


os.makedirs("checkpoints", exist_ok=True)

population_size = 10  # keep small for demonstration
population = []
for i in range(population_size):
    model = DynamicCNN(layers_config=[], optimizer_type='sgd', lr=0.1)
    population.append(model)

generations = 1000 
max_batches = 100  

logbook = []  # to record stats each generation

best_overall_acc = 0.0
best_overall_model = None

for gen in range(generations):
    print(f"\n===== Generation {gen+1} =====")

    # Train + evaluate each model
    val_scores = []
    for idx, individual in enumerate(population):
        # Train for partial epoch
        individual.train_one_epoch(train_loader, max_batches=max_batches)
        # Evaluate
        score = individual.evaluate(val_loader)
        val_scores.append(score)
        print(f" Model {idx}: Val Accuracy = {score*100:.2f}%")

    # Sort by validation accuracy (descending)
    ranked_indices = sorted(range(len(population)), key=lambda i: val_scores[i], reverse=True)
    best_idx  = ranked_indices[0]
    worst_idx = ranked_indices[-1]
    best_val_acc = val_scores[best_idx]
    worst_val_acc = val_scores[worst_idx]
    print(f" Best model is {best_idx} with val acc {best_val_acc*100:.2f}%")
    print(f" Worst model is {worst_idx} with val acc {worst_val_acc*100:.2f}%")

    # Keep track of the overall best
    if best_val_acc > best_overall_acc:
        best_overall_acc = best_val_acc
        best_overall_model = population[best_idx].clone()
        # Save a checkpoint
        torch.save(best_overall_model.model.state_dict(), 
                   f"checkpoints/best_model_gen{gen+1}.pth")
        print(f"  -> Updated overall best, saving checkpoint (ValAcc={best_val_acc:.4f})")

    # Replace the worst with a mutated copy of the best
    best_model = population[best_idx]
    mutated_child = mutate_model(best_model)
    population[worst_idx] = mutated_child
    print(f" -> Replaced model {worst_idx} with mutated child of {best_idx}")

    # Log generation info
    log_entry = {
        "generation": gen + 1,
        "best_model_idx": best_idx,
        "best_val_acc": best_val_acc,
        "worst_model_idx": worst_idx,
        "worst_val_acc": worst_val_acc
    }
    logbook.append(log_entry)

# Evaluate final best model on the test set
test_accuracy = best_overall_model.evaluate(test_loader)
print(f"\nFinal best model's Test Accuracy = {test_accuracy*100:.2f}%")

print("\nLogbook:")
for entry in logbook:
    print(entry)
