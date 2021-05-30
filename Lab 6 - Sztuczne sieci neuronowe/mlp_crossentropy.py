from collections import Counter

from dataset import vocabulary, classes, test_set, train_set
import torch
from torch import nn, optim


def vectorize(dataset_seq, vocabulary, classes):
    vocabulary_stoi = {term: index for index, term in enumerate(vocabulary)}
    classes_stoi = {term: index for index, term in enumerate(classes)}

    dataset_bow = []
    for label, msg in dataset_seq:
        bow = Counter(msg)
        feature_vector = torch.zeros(len(vocabulary))
        for elm, freq in bow.items():
            if elm in vocabulary_stoi:
                feature_vector[vocabulary_stoi[elm]] = freq

        dataset_bow.append({'input': feature_vector, 'target': classes_stoi[label]})
    return dataset_bow


class Mlp(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=512):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features)
        )

    def forward(self, x_in):
        return self.model(x_in)


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices)


# Feature engineering
train_set = vectorize(train_set, vocabulary, classes)
test_set = vectorize(test_set, vocabulary, classes)

# Hyperparameters
epochs_nb = 30
batch_size = 128
learning_rate = 0.1

classifier = Mlp(in_features=len(vocabulary), out_features=len(classes))

# Training, Validation split
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_generator = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

### Training ###
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)

for epoch_index in range(epochs_nb):
    # Training
    train_loss = 0.0
    for batch_index, batch_dict in enumerate(train_generator):
        # step 1. zero the gradients
        optimizer.zero_grad()

        # step 2. compute the output
        y_pred = classifier(x_in=batch_dict['input'])

        # step 3. compute the loss
        loss = loss_func(y_pred, batch_dict['target'])
        loss_t = loss.item()
        train_loss += (loss_t - train_loss) / (batch_index + 1)

        # step 4. use loss to produce gradients
        loss.backward()

        # step 5. use optimizer to take gradient step
        optimizer.step()
    print(f'Epoch {epoch_index} train loss: {train_loss}')

    # Validation
    val_loss = 0.0
    for batch_index, batch_dict in enumerate(val_generator):
        # compute the output
        y_pred = classifier(x_in=batch_dict['input'])

        # compute the loss
        loss = loss_func(y_pred, batch_dict['target'])
        loss_t = loss.item()
        val_loss += (loss_t - val_loss) / (batch_index + 1)

    print(f'Epoch {epoch_index} validation loss: {val_loss}')

### Testing ###
test_generator = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
test_loss = 0.0
test_acc = 0.0
for batch_index, batch_dict in enumerate(test_generator):
    # compute the output
    y_pred = classifier(x_in=batch_dict['input'])

    # compute the loss
    loss = loss_func(y_pred, batch_dict['target'])
    loss_t = loss.item()
    test_loss += (loss_t - test_loss) / (batch_index + 1)

    # compute accuracy
    test_acc += (compute_accuracy(y_pred, batch_dict['target']) - test_acc) / (batch_index + 1)

print(f'Test loss: {test_loss}. Test accuracy: {test_acc}')