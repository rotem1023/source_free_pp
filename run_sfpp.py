from loaders.load_data import *
from loaders.load_model import *


def run_sffp(model, data):
    pass
    


if __name__ == "__main__":
    train_loader, validation_loader = load_data(dataset="VISDA", domain="SR", year='2019')
    model = load_model(dataset="VISDA", model_name="shot", src_domain='S', tgt_domain='R', year='2019')
    
    with torch.no_grad():
        iter_test = iter(validation_loader)  # Create an iterator
        for _ in range(len(validation_loader)):
            data = next(iter_test)  # Use Python's built-in next()
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)