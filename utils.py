import os 
import torch 
from tqdm import tqdm 
from typing import Dict,List,Tuple

from torchvision import transforms,datasets 
from torch.utils.data import DataLoader 
import torch 
import matplotlib.pyplot as plt
from pathlib import Path 

NUM_WORKERS = os.cpu_count()

def set_seed(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def DataPreprocessing(TrainDir:str,ValDir:str,transeform=None,imageSize:int=224,batch:int=64):
    if not transeform:
        manualTranforms = transforms.Compose([
            transforms.Resize([imageSize,imageSize]),
            transforms.ToTensor()
        ])
    else:
        manualTranforms = transeform

    TrainData = datasets.ImageFolder(TrainDir,transform= manualTranforms)
    ValData = datasets.ImageFolder(ValDir,transform=manualTranforms)

    class_names = TrainData.classes

    TrainLoader = DataLoader(TrainData,batch_size=batch,shuffle=True,num_workers = NUM_WORKERS)
    ValLoader = DataLoader(ValData,batch_size=batch,shuffle =True,num_workers=NUM_WORKERS)

    return TrainLoader,ValLoader,class_names

def train_step(model:torch.nn.Module,
                dataloader:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                optimizer:torch.optim.Optimizer,
                device:torch.device)->Tuple[float,float]:

    model.train()
    train_loss,train_acc = 0,0 
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred,y)
        train_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc +=(y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss,train_acc 

def test_step(model:torch.nn.Module,
                dataloader:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                device:torch.device) -> Tuple[float,float]:

    model.eval()
    test_loss,test_acc = 0,0 
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device),y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits,y)
            test_loss +=loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels==y).sum().item()/len(test_pred_labels))

    test_loss = test_loss /len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc 



def train(model:torch.nn.Module,
            train_dataloader:torch.utils.data.DataLoader,
            test_dataloader:torch.utils.data.DataLoader,
            optimizer : torch.optim.Optimizer,
            loss_fn:torch.nn.Module,
            epochs:int,
            device:torch.device)->Dict[str,List]:

    results = {"train_loss":[],
                "train_acc":[],
                "test_loss":[],
                "test_acc":[]
                }

    model.to(device)
    print("Start Model Trainning") 
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model=model,
                                            dataloader = train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        test_loss,test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn = loss_fn,
                                        device = device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} |"
            f"train_acc: {train_acc:.4f} |"
            f"test_loss: {test_loss:.4f} |"
            f"test_acc: {test_acc:.4f} |"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results 

# Plot loss curves of a model
def plot_loss_curves(results,dir):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.savefig(f"{dir}/result.png")

def save_model(model:torch.nn.Module,
                traget_dir:str,
                model_name:str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)
    assert  model_name.endswith(".pth") or model_name.endswith(".pt"),"model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path/model_name

    print(f"[INFO] Saving model to : {model_save_path}")
    torch.save(obj=model.state_dict(),f = model_save_path)

    memory = Path(model_save_path).stat().st_size // (1024*1024)
    print(f"[INFO] Model Memory Size : {memory} MB")

