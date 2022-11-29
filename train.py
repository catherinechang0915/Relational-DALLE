import torch
import torch.utils.data
from dataset import SortOfClevrDataset

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, TRAIN_CONFIG, WANDB_KEY
from model import RN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

arch_string = """
Insert your architecture description here
"""
arch = "Insert your arch name here"
arch_file = arch+".txt"

def evaluate(model, dataloader):
    model.eval()
    total_val_acc, total_val_loss = 0.0, 0.0
    for _, data in enumerate(dataloader):
        input_img, input_qst, label = data
        input_img, input_qst, label = input_img.to(device), input_qst.to(device), label.to(device)
        with torch.no_grad():
            val_acc, val_loss = model.test_(input_img, input_qst, label)
        total_val_acc += val_acc
        total_val_loss += val_loss
        del input_img, input_qst, label, val_loss
    total_val_acc /= len(dataloader)
    total_val_loss /= len(dataloader)
    return total_val_acc, total_val_loss

def train():
    TRAIN_CONFIG['cuda'] = (device == 'cuda')
    batch_size = TRAIN_CONFIG['batch_size']
    if WANDB_KEY:
        import wandb
        wandb.login(key=WANDB_KEY)
        with open(arch_file, "w") as f:
            f.write(arch_string)

        run = wandb.init(
            name=arch,
            reinit = True, ### Allows reinitalizing runs when you re-run this cell
            project = "RelationalNetwork", ### Project should be created in your wandb account.
            entity = "11-785-deep-learning", 
        )
        wandb.save(arch_file)

    print("Training with config", TRAIN_CONFIG)

    # create dataset and dataloader
    print("==== Data Loading START ====")
    train_dataset = SortOfClevrDataset(TRAIN_DIR)
    val_dataset = SortOfClevrDataset(VAL_DIR)
    test_dataset = SortOfClevrDataset(TEST_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print("Train: {}, val: {}, test: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    print("==== Data Loading END ==== ")

    # create model
    model = RN(TRAIN_CONFIG).to(device)
    print("==== Model Created ====")

    print("==== Training START ====")
    best_acc = 0.0
    # train loop
    for epoch in range(TRAIN_CONFIG['epochs']):
        print("Epoch {}".format(epoch))

        # training
        model.train()
        total_train_acc, total_train_loss = 0.0, 0.0
        for idx, data in enumerate(train_loader):
            input_img, input_qst, label = data
            input_img, input_qst, label = input_img.to(device), input_qst.to(device), label.to(device)
            train_acc, train_loss = model.train_(input_img, input_qst, label)
            total_train_acc += train_acc
            total_train_loss += train_loss
            del input_img, input_qst, label, train_loss

        curr_lr = float(model.optimizer.param_groups[0]['lr'])

        print("Train \t Accuracy: {} \t Loss: {}".format(
            total_train_acc / len(train_loader), 
            total_train_loss / len(train_loader)
        ))

        # validation
        val_acc, val_loss = evaluate(model, val_loader)
        print("Val \t Accuracy: {} \t Loss: {}".format(val_acc, val_loss))

        if WANDB_KEY:
            wandb.log({
                "epoch": epoch,
                "train_accuracy": total_train_acc / len(train_loader),
                "train_loss":  total_train_loss / len(train_loader),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "lr": curr_lr
            })
        
        # save model
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_model(epoch)
        
    print("==== Training END ====")

    if WANDB_KEY:
        run.finish()
    
    # test
    test_acc, test_loss = evaluate(model, test_loader)
    print("Test \t Accuracy: {} \t Loss: {}".format(test_acc, test_loss))


if __name__ == '__main__':
    train()