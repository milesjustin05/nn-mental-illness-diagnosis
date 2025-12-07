import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, OrderedDict
import math

np.random.seed(42)
torch.manual_seed(42)

# put data into dataframe
df = pd.read_csv("data.csv")

# all of the symptoms into a list
symptom_cols = [
    "Anxiety", "Panic", "Sadness", "Lack of Interest", "Irritability",
    "Sleeping Problems", "Energy Problems", "Attention Problems",
    "Impulsivity", "Intrusive Thoughts", "Compulsivity", "Flashbacks",
    "Avoidance", "Mood Swings", "Self-Harm", "Emotional Instability",
    "Fear of abandonment", "Hypervigilance", "Somatic Symptoms", "Dissociation"
]

# numpy float32 data structure
X = df[symptom_cols].values.astype("float32")      

# get diagnosis into dataframe
y_text = df["Diagnosis"]

# map diagnoses into numbers
encoder = LabelEncoder()
y_global = encoder.fit_transform(y_text).astype("int64") 

#get numbers for symptoms and classes
num_features = X.shape[1]
num_classes = len(encoder.classes_)

# dictionary map that shows where every illness has a placement in the dataset
class_to_indices = defaultdict(list)
for i, c in enumerate(y_global):
    class_to_indices[c].append(i)

for c in class_to_indices:
    class_to_indices[c] = np.array(class_to_indices[c])

def sample_classification_task(N_way=2, K_shot=5, Q_query=10):

    # find what classes meet the qualifications and store them in available classes
    available_classes = [c for c, idxs in class_to_indices.items()
                         if len(idxs) >= K_shot + Q_query]

    # randomly picks two of those classes and stores them in chosen classes
    chosen_classes = np.random.choice(available_classes, size=N_way, replace=False)

    # predefining lists to store tasks
    support_X_list, support_y_list = [], []
    query_X_list, query_y_list = [], []

    # actual task sampler
    for local_label, c in enumerate(chosen_classes):
        # stores what data is for the needed illnesses
        idxs = class_to_indices[c]

        # randomize the order and set them up into lists
        chosen = np.random.choice(idxs, size=K_shot + Q_query, replace=False)
        support_idx = chosen[:K_shot]
        query_idx = chosen[K_shot:]

        # gets the chosen data readu to be put into tensors
        support_X_list.append(X[support_idx])
        support_y_list.append(np.full(K_shot, local_label, dtype=np.int64))

        query_X_list.append(X[query_idx])
        query_y_list.append(np.full(Q_query, local_label, dtype=np.int64))

    # only chatgpt understands these 4 lines but they work!
    X_support = torch.tensor(np.vstack(support_X_list), dtype=torch.float32)
    y_support = torch.tensor(np.concatenate(support_y_list), dtype=torch.long)
    X_query   = torch.tensor(np.vstack(query_X_list), dtype=torch.float32)
    y_query   = torch.tensor(np.concatenate(query_y_list), dtype=torch.long)

    return X_support, y_support, X_query, y_query, chosen_classes

# MLP model
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)  # use out_dim (2-way)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = F.linear(x, params['fc1.weight'], params['fc1.bias'])
            x = F.relu(x)
            x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        return x

# hyperparameters
inner_lr = 0.001
outer_lr = 0.00001
num_inner_steps = 5
num_tasks = 16  

# define main model stuff
model = SimpleClassifier(num_features, out_dim=2)    
meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
loss_fn = nn.CrossEntropyLoss()

def get_params_dict(model):
    return OrderedDict((name, p) for (name, p) in model.named_parameters())

for epoch in range(800):
    meta_loss = 0.0

    for _ in range(num_tasks):
        # create task
        X_sup, y_sup, X_q, y_q, chosen_classes = sample_classification_task(N_way=2, K_shot=7, Q_query=3)

        # get the parameters from the base model
        fast_params = get_params_dict(model)

        # train the new model on the task
        for _ in range(num_inner_steps):
            logits_sup = model(X_sup, params=fast_params)
            loss_sup = loss_fn(logits_sup, y_sup)

            # update weights for test model
            grads = torch.autograd.grad(loss_sup, fast_params.values(), create_graph=True)

            # saves the parameters so they can be reused in the loop
            fast_params = OrderedDict((name, p - inner_lr * g) for ( (name, p), g ) in zip(fast_params.items(), grads))

        #do one more run and get the loss for that specific model
        logits_q = model(X_q, params=fast_params)
        loss_q = loss_fn(logits_q, y_q)
        meta_loss += loss_q

    #average all of the losses
    meta_loss /= num_tasks

    #use the meta loss to update the main model
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Meta Loss: {meta_loss.item():.4f}")

# takes the generalized model and trains it on the two illnesses being compared during testing
# same logic as training code
def adapt_to_task(model, X_sup, y_sup, num_steps=5):
    fast_params = get_params_dict(model)
    for _ in range(num_steps):
        logits = model(X_sup, params=fast_params)
        loss = loss_fn(logits, y_sup)
        grads = torch.autograd.grad(loss, fast_params.values())
        fast_params = OrderedDict(
            (name, p - inner_lr * g)
            for ( (name, p), g ) in zip(fast_params.items(), grads)
        )
    return fast_params

# same logic as task sampling, but used for test data
def build_task_for_pair(name_a, name_b, K_shot=7, Q_query=3):
    global_a = encoder.transform([name_a])[0]
    global_b = encoder.transform([name_b])[0]
    chosen_classes = np.array([global_a, global_b])

    support_X_list, support_y_list = [], []
    query_X_list, query_y_list = [], []

    for local_label, c in enumerate(chosen_classes):
        idxs = class_to_indices[c]
        assert len(idxs) >= K_shot + Q_query, f"Not enough samples for class {c}"
        chosen = np.random.choice(idxs, size=K_shot + Q_query, replace=False)
        support_idx = chosen[:K_shot]
        query_idx = chosen[K_shot:]

        support_X_list.append(X[support_idx])
        support_y_list.append(np.full(K_shot, local_label, dtype=np.int64))

        query_X_list.append(X[query_idx])
        query_y_list.append(np.full(Q_query, local_label, dtype=np.int64))

    X_support = torch.tensor(np.vstack(support_X_list), dtype=torch.float32)
    y_support = torch.tensor(np.concatenate(support_y_list), dtype=torch.long)
    X_query   = torch.tensor(np.vstack(query_X_list), dtype=torch.float32)
    y_query   = torch.tensor(np.concatenate(query_y_list), dtype=torch.long)

    return X_support, y_support, X_query, y_query, chosen_classes

# get test data
test_data = pd.read_csv("test_data.csv")

#pre define variables
correct = 0
distance = 0

# for each row in test data, seperate the diagnosis and illnesses info
for index, row in test_data.iterrows():
    test_diagnosis = row["Diagnosis"]
    features = row.values[1:].astype("float32")

    #randomly generate a comparison illness and make sure it is not the same as the one being tested
    random_illness = test_diagnosis
    while random_illness == test_diagnosis:
        random_illness = np.random.choice(encoder.classes_)

    # build tasks to train generalized model on
    X_sup, y_sup, _, _, chosen_classes = build_task_for_pair(test_diagnosis, random_illness, K_shot=7, Q_query=3)

    # train model on new tasks
    fast_params = adapt_to_task(model, X_sup, y_sup, num_steps=5)

    # put data into a tensor to make pytorch happy
    x_new = torch.tensor(features).unsqueeze(0)

    # run the model and get probabilities
    with torch.no_grad():
        logits = model(x_new, params=fast_params)
        probs = torch.softmax(logits, dim=1)

    # chatGPT can explain this, I however can not.
    pred_local = probs.argmax(dim=1).item()          
    pred_global = chosen_classes[pred_local]         
    diagnosis = encoder.inverse_transform([pred_global])[0]
    chosen_names = encoder.inverse_transform(chosen_classes)

    # my own accuracy test bc too lazy to learn to use scikit
    if diagnosis == test_diagnosis:
        correct += 1

    # specific task info, not really important
    print("Task pair (names):", chosen_names)
    print("Predicted diagnosis:", diagnosis)
    print("Probabilities:", probs.numpy())

    # get probabilites into a numpy array then split them into numbers
    probs_numpy = probs.numpy()  
    p0 = probs_numpy[0][0]
    p1 = probs_numpy[0][1]

    # calculate the distance between them
    task_distance = math.fabs(p0 - p1)
    distance += task_distance

# print model accuracy and average distance
print(f'Accuracy: {(correct/50)*100:.2f}%')
print(f"Average distance: {distance/50}")


