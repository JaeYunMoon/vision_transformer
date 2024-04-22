import os 
import glob
import torch 
import torchvision 
from utils import set_seed,DataPreprocessing,train,plot_loss_curves,save_model

from model import ViT

def main(pretrained = True):
    IMAGE_SIZE= 224
    BATCH_SIZE = 64 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE : ",DEVICE)

    TrainDir = "/home/compu/Data/Apple/apple/Train"
    ValDir = "/home/compu/Data/Apple/apple/Val"
    saveDir = "./result/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    set_seed()

    if pretrained:
        print("pretrained Weights")
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(DEVICE)
        # for parameter in vit.parameters():
        #     parameter.requires_grad = False
        pretrained_vit_transforms = pretrained_vit_weights.transforms()
        TrainLoader,ValLoader,class_names=DataPreprocessing(TrainDir,ValDir,batch=BATCH_SIZE,transeform=pretrained_vit_transforms)
    else:
        print("Not pretrained")
        vit = ViT(num_classes=len(class_names))
        TrainLoader,ValLoader,class_names=DataPreprocessing(TrainDir,ValDir,batch=BATCH_SIZE)

    optimizer = torch.optim.Adam(params = vit.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    results = train(model=vit,
                    train_dataloader = TrainLoader,
                    test_dataloader=ValLoader,
                    optimizer=optimizer,
                    loss_fn = loss_fn,
                    epochs=50,
                    device = DEVICE
                    )

    plot_loss_curves(results,dir = saveDir)
    save_model(model = vit,traget_dir=saveDir,model_name="apple_pest.pt")


if __name__ == "__main__":
    main()