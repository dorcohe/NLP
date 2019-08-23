import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,  DataLoader
from parse_data import Parse_data
import torch
from pytorch_transformers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########################################################################## Database

class EmoBankDatabase(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, reader_path, writer_path, raw_information_path, sem2007_paths):
      parse_data_instance = Parse_data(reader_path, writer_path, raw_information_path, sem2007_paths)
      self.par_data = parse_data_instance.get_train_data()

    def __len__(self):
        return len(self.par_data)

    def __getitem__(self, idx):
        return self.par_data[idx]["text"], torch.tensor([self.par_data[idx]["v"], self.par_data[idx]["a"], self.par_data[idx]["d"]])

##########################################################################


########################################################################## Model
class EmobankFineTuningModule(torch.nn.Module):
    def __init__(self, model_class):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(EmobankFineTuningModule, self).__init__()

        self.bert = model_class
        self.linear2 = torch.nn.Linear(768, 3)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = [get_all_hidden_states(instance, self.bert) for instance in x]
        x = [self.linear2(instance.view(-1, 768)) for instance in x]
        x = [torch.mean(instance, dim=0) for instance in x]
        x = torch.stack(x, dim=0, out=None)
        return x


##########################################################################




########################################################################## Methods

def get_all_hidden_states(x, model):
    all_hidden_states, all_attentions = model(x)[-2:]

    return all_hidden_states[0]


def get_all_hidden_states(x, model):
    all_hidden_states, all_attentions = model(x)[-2:]

    return all_hidden_states[0]



##########################################################################



########################################################################## Set  basic parameters

reader_path = "./corpus/reader.csv"
writer_path = "./corpus/writer.csv"
raw_path = "./corpus/raw.csv"
sem2007_paths = "./affectivetext_test.emotions.gold"



MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,      'gpt2'),
          (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024')]

model_class, tokenizer_class, pretrained_weights = MODELS[0]



##########################################################################




########################################################################## Set network training parameters


def main():
    ## Dataset
    emobank_dataset_training = EmoBankDatabase(reader_path, writer_path, raw_path, sem2007_paths)

    dataloader_training = DataLoader(emobank_dataset_training, batch_size=4,
                            shuffle=True, num_workers=4)

    ## Model
    bert_model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=True)
    bert_model.to(device)
    criterion = nn.MSELoss()
    model = EmobankFineTuningModule(bert_model)
    model.to(device)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)




    Train_Net(model, dataloader_training, criterion, tokenizer)



if __name__ == '__main__':
    main()