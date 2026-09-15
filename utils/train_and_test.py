import torch 
import torch.nn.functional as F 
 
 
def train(model, device, train_loader, optimizer, epoch, verbose=False): 
    """ 
    Train the model for one epoch. 
 
    Returns: 
        (average_loss, accuracy_percentage) 
    """ 
    model.train() 
 
    total_loss = 0.0 
    correct = 0 
    total_samples = 0 
 
    for batch_idx, (data, target) in enumerate(train_loader): 
        data = data.to(device, non_blocking=True) 
        target = target.to(device, non_blocking=True) 
 
        optimizer.zero_grad() 
 
        output = model(data) 
        loss = F.nll_loss(output, target) 
 
        loss.backward() 
        optimizer.step() 
 
        batch_size = target.size(0) 
 
        total_loss += loss.item() * batch_size 
 
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq( 
            target.view_as(pred) 
        ).sum().item() 
 
        total_samples += batch_size 
 
        if batch_idx % 20 == 0 and verbose: 
            processed = batch_idx * batch_size 
            percentage = ( 
                100.0 * batch_idx / len(train_loader) 
                if len(train_loader) > 0 
                else 0.0 
            ) 
 
            print( 
                f"Train Epoch: {epoch} " 
                f"[{processed}/{len(train_loader.dataset)} " 
                f"({percentage:.0f}%)]\t" 
                f"Loss: {loss.item():.6f}" 
            ) 
 
    if total_samples == 0: 
        raise RuntimeError( 
            "Training loader contains no samples." 
        ) 
 
    average_loss = total_loss / total_samples 
    accuracy = 100.0 * correct / total_samples 
 
    return average_loss, accuracy 
 
 
def test(model, device, test_loader, verbose=False): 
    """ 
    Evaluate the model without updating its parameters. 
 
    Returns: 
        (average_loss, accuracy_percentage) 
    """ 
    model.eval() 
 
    total_loss = 0.0 
    correct = 0 
    total_samples = 0 
 
    with torch.no_grad(): 
        for data, target in test_loader: 
            data = data.to(device, non_blocking=True) 
            target = target.to(device, non_blocking=True) 
 
            output = model(data) 
            loss = F.nll_loss(output, target) 
 
            batch_size = target.size(0) 
 
            total_loss += loss.item() * batch_size 
 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq( 
                target.view_as(pred) 
            ).sum().item() 
 
            total_samples += batch_size 
 
    if total_samples == 0: 
        raise RuntimeError( 
            "Test loader contains no samples." 
        ) 
 
    average_loss = total_loss / total_samples 
    accuracy = 100.0 * correct / total_samples 
 
    if verbose: 
        print( 
            f"Test set: Average loss: {average_loss:.4f}, " 
            f"Accuracy: {correct}/{total_samples} " 
            f"({accuracy:.2f}%)" 
        ) 
 
    return average_loss, accuracy 
