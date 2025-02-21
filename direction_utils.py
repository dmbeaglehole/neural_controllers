import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0,'/u/dbeaglehole/recursive_feature_machines')

from rfm import LaplaceRFM
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from torchmetrics.regression import R2Score

from copy import deepcopy
from tqdm import tqdm

# For scaling linear probe beyond ~50k datapoints.
def batch_transpose_multiply(A, B, mb_size=5000):
    n = len(A)
    assert(len(A) == len(B))
    batches = torch.split(torch.arange(n), mb_size)
    sum = 0.
    for b in batches:
        Ab = A[b].cuda()
        Bb = B[b].cuda()
        sum += Ab.T @ Bb

        del Ab, Bb
    return sum

def accuracy_fn(preds, truth):
    assert(len(preds)==len(truth))
    true_shape = truth.shape
    
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).to(truth.device)
    preds = preds.reshape(true_shape)
    
    if preds.shape[1] == 1:
        preds = torch.where(preds >= 0.5, 1, 0)
        truth = torch.where(truth >= 0.5, 1, 0)
    else:
        preds = torch.argmax(preds, dim=1)
        truth = torch.argmax(truth, dim=1)
        
    acc = torch.sum(preds==truth)/len(preds) * 100
    return acc.item()

def pearson_corr(x, y):     
    assert(x.shape == y.shape)
    
    x = x.float() + 0.0
    y = y.float() + 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))

    return numerator / denominator

def split_data(data, labels):
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=0, shuffle=True
    ) 
    return data_train, data_test, labels_train, labels_test

def precision_score(preds, labels):
    true_positives = torch.sum((preds == 1) & (labels == 1))
    predicted_positives = torch.sum(preds == 1)
    return true_positives / (predicted_positives + 1e-8)  # add small epsilon to prevent division by zero

def recall_score(preds, labels):
    true_positives = torch.sum((preds == 1) & (labels == 1))
    actual_positives = torch.sum(labels == 1)
    return true_positives / (actual_positives + 1e-8)  # add small epsilon to prevent division by zero

def f1_score(preds, labels):
    precision = precision_score(preds, labels)
    recall = recall_score(preds, labels)
    return 2 * (precision * recall) / (precision + recall + 1e-8)  # add small epsilon to prevent division by zero

def compute_classification_metrics(preds, labels):
    num_classes = labels.shape[1]
    if num_classes == 1:  # Binary classification
        preds = torch.where(preds >= 0.5, 1, 0)
        labels = torch.where(labels >= 0.5, 1, 0)
        acc = accuracy_fn(preds, labels)
        precision = precision_score(preds, labels).item()
        recall = recall_score(preds, labels).item()
        f1 = f1_score(preds, labels).item()
    else:  # Multiclass classification
        preds_classes = torch.argmax(preds, dim=1)
        label_classes = torch.argmax(labels, dim=1)
        
        # Compute accuracy
        acc = torch.sum(preds_classes == label_classes).float() / len(preds) * 100
        
        # Initialize metrics for averaging
        precision, recall, f1 = 0.0, 0.0, 0.0
        
        # Compute metrics for each class
        for class_idx in range(num_classes):
            class_preds = (preds_classes == class_idx).float()
            class_labels = (label_classes == class_idx).float()
            
            precision += precision_score(class_preds, class_labels).item()
            recall += recall_score(class_preds, class_labels).item()
            f1 += f1_score(class_preds, class_labels).item()
        
        # Average metrics across classes
        precision /= num_classes
        recall /= num_classes
        f1 /= num_classes
        acc = acc.item()

    metrics = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    return metrics

def get_hidden_states(prompts, model, tokenizer, hidden_layers, forward_batch_size, rep_token=-1, all_positions=False):

    try: 
        name = model._get_name()
        seq2seq = (name=='T5ForConditionalGeneration')
    except:
        seq2seq = False

    if seq2seq:
        encoded_inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    else:
        encoded_inputs = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        encoded_inputs['attention_mask'] = encoded_inputs['attention_mask'].half()
    
    dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=forward_batch_size)

    all_hidden_states = {}
    for layer_idx in hidden_layers:
        all_hidden_states[layer_idx] = []

    use_concat = list(hidden_layers)==['concat']
    print("use_concat", use_concat)
    # Loop over batches and accumulate outputs
    print("Getting activations from forward passes")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch

            if seq2seq:
                encoder_outputs = model.encoder(
                    input_ids=input_ids,
                    output_hidden_states=True
                )                
                out_hidden_states = encoder_outputs.hidden_states

                decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).cuda()
                decoder_outputs = model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    output_hidden_states=True
                )
                out_hidden_states = decoder_outputs.hidden_states

                num_layers = len(out_hidden_states)-1 # exclude embedding layer
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                num_layers = len(model.model.layers)
                out_hidden_states = outputs.hidden_states
            
            hidden_states_all_layers = []
            for layer_idx, hidden_state in zip(range(-1, -num_layers, -1), reversed(out_hidden_states)):
                
                if use_concat:
                    hidden_states_all_layers.append(hidden_state[:,rep_token,:].detach().cpu())
                elif all_positions:
                    all_hidden_states[layer_idx].append(hidden_state.detach().cpu())
                else:
                    all_hidden_states[layer_idx].append(hidden_state[:,rep_token,:].detach().cpu())
                    
            if use_concat:
                hidden_states_all_layers = torch.cat(hidden_states_all_layers, dim=1)
                all_hidden_states['concat'].append(hidden_states_all_layers)
                      
    # Concatenate results from all batches
    final_hidden_states = {}
    for layer_idx, hidden_state_list in all_hidden_states.items():
        final_hidden_states[layer_idx] = torch.cat(hidden_state_list, dim=0)
        
    return final_hidden_states

def project_hidden_states(hidden_states, directions, n_components):
    """
    directions:
        {-1 : [beta_{1}, .., beta_{m}],
        ...,
        -31 : [beta_{1}, ..., beta_{m}]
        }
    hidden_states:
        {-1 : [h_{1}, .., h_{d}],
        ...,
        -31 : [h_{1}, ..., h_{d}]
        }
    """
    print("n_components", n_components)
    assert(hidden_states.keys()==directions.keys())
    layers = hidden_states.keys()
    
    projections = {}
    for layer in layers:
        vecs = directions[layer][:n_components].T
        projections[layer] = hidden_states[layer].cuda()@vecs.cuda()
    return projections

def aggregate_projections_on_coefs(projections, detector_coef):
    """
    detector_coefs:
        {-1 : [beta_{1}, bias_{1}],
        ...,
        -31 : [beta_31_{31}, bias_{31},
        'agg_sol': [beta_{agg}, bias_{agg}]]
    projections:
        {-1 : tensor (n, n_components),
        ...,
        -31 : tensor (n, n_components),
        }
    """
        
    layers = projections.keys()
    agg_projections = []
    for layer in layers:
        X = projections[layer].cuda()
        agg_projections.append(X.squeeze(0))
    
    # print("X", X.shape)
    agg_projections = torch.concat(agg_projections, dim=1).squeeze()
    # print("agg_projections", agg_projections.shape)
    # print("detector_coef", detector_coef)
    agg_beta = detector_coef[0]
    agg_bias = detector_coef[1]
    agg_preds = agg_projections@agg_beta + agg_bias
    return agg_preds

def project_onto_direction(tensors, direction, device='cuda'):
    """
    tensors : (n, d)
    direction : (d, )
    output : (n, )
    """
    print("tensors", tensors.shape, "direction", direction.shape)
    assert(len(tensors.shape)==2)
    assert(tensors.shape[1] == direction.shape[0])
    
    return tensors.to(device=device) @ direction.to(device=device, dtype=tensors.dtype)

def fit_pca_model(train_X, train_y, n_components=1):
    """
    Assumes the data are in ordered pairs of pos/neg versions of the same prompts:
    
    e.g. the first four elements of train_X correspond to 
    
    Dishonestly say something about {object x}
    Honestly say something about {object x}
    
    Honestly say something about {object y}
    Dishonestly say something about {object y}
    
    """
    pos_indices = torch.isclose(train_y, torch.ones_like(train_y)).squeeze(1)
    neg_indices = torch.isclose(train_y, torch.zeros_like(train_y)).squeeze(1)
    
    pos_examples = train_X[pos_indices]
    neg_examples = train_X[neg_indices]
    
    dif_vectors = pos_examples - neg_examples
    
    # randomly flip the sign of the vectors
    random_signs = torch.randint(0, 2, (len(dif_vectors),)).float().to(dif_vectors.device) * 2 - 1
    dif_vectors = dif_vectors * random_signs.reshape(-1,1)
    
    # dif_vectors : (n//2, d)
    XtX = dif_vectors.T@dif_vectors
    S, U = torch.linalg.eigh(XtX)

    return torch.flip(U[:,-n_components:].T, dims=(0,))

def append_one(X):
    Xb = torch.concat([X, torch.ones_like(X[:,0]).unsqueeze(1)], dim=1)
    new_shape = X.shape[:1] + (X.shape[1]+1,) 
    assert(Xb.shape == new_shape)
    return Xb

def linear_solve(X, y, use_bias=True):
    """
    projected_inputs : (n, d)
    labels : (n, c) or (n, )
    """
    
    if use_bias:
        inputs = append_one(X)
    else:
        inputs = X
    
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    num_classes = y.shape[1]
    n, d = inputs.shape
    
    if n>d:   
        XtX = inputs.T@inputs
        XtY = inputs.T@y
        beta = torch.linalg.pinv(XtX)@XtY # (d, c)
    else:
        XXt = inputs@inputs.T
        alpha = torch.linalg.pinv(XXt)@y # (n, c)
        beta = inputs.T @ alpha
    
    if use_bias:
        sol = beta[:-1]
        bias = beta[-1]
        if num_classes == 1:
            bias = bias.item()
        return sol, bias
    else:
        return beta
        

def logistic_solve(X, y):
    """
    projected_inputs : (n, d)
    labels : (n, c)
    """

    num_classes = y.shape[1]
    if num_classes == 1:
        y = y.flatten()
        model = LogisticRegression(fit_intercept=True, max_iter=1000) # use bias
    else:
        y = y.argmax(dim=1)
        model = LogisticRegression(fit_intercept=True, max_iter=1000, multi_class='multinomial') # use bias
    model.fit(X.cpu(), y.cpu())
    
    beta = torch.from_numpy(model.coef_).to(X.dtype).to(X.device)
    bias = torch.from_numpy(model.intercept_).to(X.dtype).to(X.device)
    
    return beta.T, bias

def aggregate_layers(layer_outputs, val_y, test_y, use_logistic=False, use_rfm=False):
            
    # solve aggregator on validation set
    val_X = torch.concat(layer_outputs['val'], dim=1) # (n, num_layers*n_components)    
    test_X = torch.concat(layer_outputs['test'], dim=1)
    print("val_X", val_X.shape, "val_y", val_y.shape)

    
    if use_rfm:
        model = LaplaceRFM(bandwidth=100, reg=1e-3, device='cuda')
        model.fit(
            (val_X, val_y), 
            (test_X, test_y), 
            loader=False, 
            iters=4,
            classif=False,
            method='lstsq',
            M_batch_size=2048,
            verbose=False
        )              
        agg_preds = model.predict(test_X)
        metrics = compute_classification_metrics(agg_preds, test_y)
        return metrics, None, None
    elif use_logistic:
        print("Using logistic aggregation")
        agg_beta, agg_bias = logistic_solve(val_X, val_y) # (num_layers*n_components, num_classes)
    else:
        print("Using linear aggregation")
        agg_beta, agg_bias = linear_solve(val_X, val_y) # (num_layers*n_components, num_classes)

    # evaluate aggregated predictor on test set
    agg_preds = test_X@agg_beta + agg_bias
    agg_preds = agg_preds.reshape(test_y.shape)
    metrics = compute_classification_metrics(agg_preds, test_y)
    return metrics, agg_beta, agg_bias

    
def train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, hyperparams, 
                               bws=[0.2, 1, 5, 10, 100, 1000], 
                               regs=[1e-3, 1e-2, 1e-1, 1e0, 1e1]):
    
    best_model = None
    best_loss = float('inf')
    best_acc = None
    for reg in regs:
        for bw in bws:
            model = LaplaceRFM(bandwidth=bw, reg=reg, device='cuda')

            model.fit(
                (train_X, train_y), 
                (val_X, val_y), 
                loader=False, 
                iters=hyperparams['rfm_iters'],
                classification=False,
                method='lstsq',
                M_batch_size=hyperparams['M_batch_size'],
                verbose=False
            )

            preds = []
            mb_size = 512
            batches = torch.split(torch.arange(len(val_X)), mb_size)
            for b in batches:
                Xb = val_X[b].cuda()
                preds.append(model.predict(Xb))
                del Xb
            preds = torch.concat(preds, dim=0)
            val_loss = torch.mean((preds-val_y.cuda())**2)
            
            assert(preds.shape == val_y.shape)
            
            if preds.shape[1] == 1:
                r2score = R2Score().cuda()
                val_r2 = r2score(preds, val_y.cuda()).item()
            else:
                val_r2 = None

            if val_loss < best_loss:
                best_val_r2 = val_r2
                best_loss = val_loss
                best_reg = reg
                best_model = deepcopy(model)
                best_bw = bw    
                best_acc = accuracy_fn(preds, val_y)
        
    print(f'Best RFM loss: {best_loss}, R2: {best_val_r2}, reg: {best_reg}, bw: {best_bw}, acc: {best_acc}')

    return best_model

def train_linear_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False):
    
    if use_bias:
        X = append_one(train_X)
        Xval = append_one(val_X)
    else:
        X = train_X
        Xval = val_X
    
    n, d = X.shape
    num_classes = train_y.shape[1]

    best_loss = float('inf')
    best_beta = None
    for reg in [0., 1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100, 1000, 10000]:

        if n>d:
            XtX = batch_transpose_multiply(X, X)
            XtY = batch_transpose_multiply(X, train_y)
            beta = torch.linalg.pinv(XtX + reg*torch.eye(X.shape[1]).cuda())@XtY
        else:
            X = X.cuda()
            train_y = train_y.cuda()
            X = X.cuda()
            Xval = Xval.cuda()

            XXt = X@X.T
            alpha = torch.linalg.pinv(XXt + reg*torch.eye(X.shape[0]).cuda())@train_y
            beta = X.T@alpha

        preds = Xval.cuda() @ beta

        val_loss = torch.mean((preds-val_y.cuda())**2)

        if preds.shape[1] == 1:
            r2score = R2Score().cuda()
            val_r2 = r2score(preds, val_y.cuda()).item()
        else:
            val_r2 = None

        if val_loss < best_loss:
            best_val_r2 = val_r2
            best_loss = val_loss
            best_reg = reg
            best_beta = deepcopy(beta)
            best_acc = accuracy_fn(preds, val_y)
    
    print(f'Linear probe loss: {best_loss}, R2: {best_val_r2}, reg: {best_reg}, acc: {best_acc}')

    if use_bias:
        line = best_beta[:-1].to(train_X.device)
        if num_classes == 1:
            bias = best_beta[-1].item()
        else:
            bias = best_beta[-1]
    else:
        line = best_beta.to(train_X.device)
        bias = 0
        
    return line, bias
