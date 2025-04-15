import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys


from rfm import LaplaceRFM
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from torchmetrics.regression import R2Score
from sklearn.metrics import roc_auc_score

from copy import deepcopy
from tqdm import tqdm

from utils import preds_to_proba, split_indices

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

def auroc_score(preds, labels):
    return roc_auc_score(labels.cpu(), preds.cpu())

def compute_prediction_metrics(preds, labels):
    num_classes = labels.shape[1]
    auc = auroc_score(preds, labels)
    mse = torch.mean((preds-labels)**2).item()
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

    metrics = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mse': mse}
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
    
    agg_projections = torch.concat(agg_projections, dim=1).squeeze()
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
    assert(len(tensors.shape)==2)
    assert(tensors.shape[1] == direction.shape[0])
    
    return tensors.to(device=device) @ direction.to(device=device, dtype=tensors.dtype)

def fit_pca_model(train_X, train_y, n_components=1, mean_center=True):
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
    if mean_center:
        dif_vectors -= torch.mean(dif_vectors, dim=0, keepdim=True)

    # dif_vectors : (n//2, d)
    XtX = dif_vectors.T@dif_vectors
    # _, U = torch.linalg.eigh(XtX)
    # return torch.flip(U[:,-n_components:].T, dims=(0,))

    _, U = torch.lobpcg(XtX, k=n_components)
    return U.T

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
    else:
        y = y.argmax(dim=1)
    model = LogisticRegression(fit_intercept=True, max_iter=1000) # use bias
    model.fit(X.cpu(), y.cpu())
    
    beta = torch.from_numpy(model.coef_).to(X.dtype).to(X.device)
    bias = torch.from_numpy(model.intercept_).to(X.dtype).to(X.device)
    
    return beta.T, bias

def aggregate_layers(layer_outputs, val_y, test_y, use_logistic=False, use_rfm=False, tuning_metric='auc'):
    
    # solve aggregator on validation set
    val_X = torch.concat(layer_outputs['val'], dim=1) # (n, num_layers*n_components)    
    test_X = torch.concat(layer_outputs['test'], dim=1)

    # split validation set into train and val
    train_indices, val_indices = split_indices(len(val_y))
    train_y = val_y[train_indices]
    val_y = val_y[val_indices]
    train_X = val_X[train_indices]
    val_X = val_X[val_indices]
    print("train_X", train_X.shape, "val_X", val_X.shape, "test_X", test_X.shape)

    if use_rfm:
        model = LaplaceRFM(bandwidth=10, reg=1e-3, device='cuda')
        model.fit(
            (train_X, train_y), 
            (val_X, val_y), 
            loader=False, 
            iters=5,
            classification=False,
            method='lstsq',
            M_batch_size=2048,
            verbose=False,
            return_best_params=True
        )        
        agg_preds = model.predict(test_X)
        metrics = compute_prediction_metrics(agg_preds, test_y)
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
    metrics = compute_prediction_metrics(agg_preds, test_y)
    return metrics, agg_beta, agg_bias

    
def train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, 
                               hyperparams, search_space=None, 
                               tuning_metric='auc'):
    
    if search_space is None:
        search_space = {
            'regs': [1e-3],
            'bws': [1, 10, 100],
            'center_grads': [True, False]
        }
    
    best_model = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'acc'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for reg in search_space['regs']:
        for bw in search_space['bws']:
            for center_grads in search_space['center_grads']:
                try:
                    model = LaplaceRFM(bandwidth=bw, reg=reg, center_grads=center_grads, device='cuda')
                    model.fit(
                        (train_X, train_y), 
                        (val_X, val_y), 
                        loader=False, 
                        iters=hyperparams['rfm_iters'],
                        classification=False,
                        method='lstsq',
                        M_batch_size=2048,
                        verbose=False,
                        return_best_params=True
                    )              
                    val_score = model.score(val_X, val_y, metric='accuracy')

                    if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
                        best_score = val_score
                        best_reg = reg
                        best_model = deepcopy(model)
                        best_bw = bw    
                        best_center_grads = center_grads
                except Exception as e:
                    import traceback
                    print(f'Error fitting RFM: {traceback.format_exc()}')
                    continue
            
    print(f'Best RFM {tuning_metric}: {best_score}, reg: {best_reg}, bw: {best_bw}, center_grads: {best_center_grads}')

    return best_model

def train_linear_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, tuning_metric='auc', device='cuda'):

    reg_search_space = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    
    if use_bias:
        X = append_one(train_X)
        Xval = append_one(val_X)
    else:
        X = train_X
        Xval = val_X
    
    n, d = X.shape
    num_classes = train_y.shape[1]

    best_beta = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'acc'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for reg in reg_search_space:
        try:
            if n>d:
                XtX = batch_transpose_multiply(X, X)
                XtY = batch_transpose_multiply(X, train_y)
                beta = torch.linalg.solve(XtX + reg*torch.eye(X.shape[1]).to(device), XtY)
            else:
                X = X.to(device)
                train_y = train_y.to(device)
                Xval = Xval.to(device)

                XXt = X@X.T
                alpha = torch.linalg.solve(XXt + reg*torch.eye(X.shape[0]).to(device), train_y)
                beta = X.T@alpha

            preds = Xval.to(device) @ beta
            preds_proba = preds_to_proba(preds)
            val_score = compute_prediction_metrics(preds_proba, val_y)[tuning_metric]

            if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
                best_score = val_score
                best_reg = reg
                best_beta = deepcopy(beta)
        except Exception as e:
            import traceback
            print(f'Error fitting linear probe: {traceback.format_exc()}')
            continue
    
    print(f'Linear probe {tuning_metric}: {best_score}, reg: {best_reg}')

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

def train_logistic_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, num_classes=1, tuning_metric='auc'):
    
    C_search_space = [1000, 100, 10, 1, 1e-1, 1e-2]

    val_y = val_y.cpu()
    if num_classes == 1:
        train_y_flat = train_y.squeeze(1).cpu()
    else:
        train_y_flat = train_y.argmax(dim=1).cpu()   

    best_beta = None
    best_bias = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'acc'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for C in C_search_space:
        model = LogisticRegression(fit_intercept=False, max_iter=1000, C=C)
        model.fit(train_X.cpu(), train_y_flat.cpu())
        
        # Get probability predictions
        val_probs = torch.tensor(model.predict_proba(val_X.cpu()))
        if num_classes == 1:
            val_probs = val_probs[:,1].reshape(val_y.shape)
        val_score = compute_prediction_metrics(val_probs, val_y)[tuning_metric]

        if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
            best_score = val_score
            best_beta = torch.from_numpy(model.coef_).T
            if use_bias:
                best_bias = torch.from_numpy(model.intercept_)
            best_C = C

    print(f'Logistic probe {tuning_metric}: {best_score}, C: {best_C}')

    if use_bias:
        line = best_beta.to(train_X.device)
        if num_classes == 1:
            bias = best_bias.item()
        else:
            bias = best_bias
    else:
        line = best_beta.to(train_X.device)
        bias = 0
        
    return line, bias