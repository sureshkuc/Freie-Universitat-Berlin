import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import urllib
from torch.autograd import Variable


text=''
for line in urllib.request.urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
  text+=line.decode("utf-8")
#path_to_file ='/content/data.txt'
#text = open(path_to_file, encoding='utf-8').read()

print(text[:250])



# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))


# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
text_as_int = np.array([char2idx[c] for c in text])

# Create a mapping from indices to characters
idx2char = np.array(vocab)

text_as_int


print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))



def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr) // batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n + seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y



batches = get_batches(text_as_int, 8, 50)
x, y = next(batches)

# printing out the first 10 items in a sequence
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])



train_on_gpu = torch.cuda.is_available()
print ('Training on GPU' if train_on_gpu else 'Training on CPU')

class VanillaCharRNN(nn.Module):
    def __init__(self, vocab, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super(VanillaCharRNN,self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.vocab=vocab
        #self.encoder = nn.Embedding(len(self.vocab), n_hidden)
        #self.gru = nn.GRU(n_hidden, n_hidden, n_layers)
        #self.decoder = nn.Linear(n_hidden, len(self.vocab))
        
        self.hidden_size = n_hidden

        self.i2h = nn.Linear(len(self.vocab), n_hidden)
        self.i2o = nn.Linear(len(self.vocab), len(self.vocab))
        self.softmax = nn.LogSoftmax(dim=1)
        '''TODO: define the layers you need for the model'''


    def forward(self, input, hidden):
        print(input.size(),hidden.size())
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = weight.new(batch_size, seq_length,len(self.vocab)).zero_().cuda()
        else:
            hidden = weight.new(batch_size, seq_length,len(self.vocab)).zero_()
        return hidden

class LSTMCharRNN(nn.Module):
    def __init__(self, vocab, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.vocab=vocab
        '''TODO: define the layers you need for the model'''
        self.lstm = nn.LSTM(len(self.vocab), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.vocab))

    def forward(self, x, hidden):
        '''TODO: Forward pass through the network
        x is the input and `hidden` is the hidden/cell state .'''
        r_output, hidden_t = self.lstm(x, hidden)
        
        # pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        
        ## put x through the fully-connected layer
        out = self.fc(out)
        # return the final output and the hidden state
        return out, hidden_t

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

def train(model, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network

        Arguments
        ---------

        model: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if (train_on_gpu):
        model.cuda()

    counter = 0
    n_vocab = len(model.vocab)
    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)
        
        '''TODO: use the get_batches function to generate sequences of the desired size'''
        dataset = get_batches(data, batch_size, seq_length) 

        for x, y in dataset:
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_vocab)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            #h = tuple([each.data for each in h])
            if type(h) is tuple:
                if train_on_gpu:
                    h = tuple([each.data.cuda() for each in h])
                else:
                    h = tuple([each.data for each in h])
            else:
                if train_on_gpu:
                    h = h.data.cuda()
                else:
                    h = h.data
            # zero accumulated gradients
            model.zero_grad()
            
            '''TODO: feed the current input into the model and generate output'''
            output, h = model(inputs, h) # TODO

            '''TODO: compute the loss!'''
            loss = criterion(output, targets.view(batch_size*seq_length))# TODO
            
            # perform backprop
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_vocab)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if (train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    '''TODO: feed the current input into the model and generate output'''
                    output, val_h = model(inputs, val_h) # TODO
                    
                    '''TODO: compute the validation loss!'''
                    val_loss =criterion(output, targets.view(batch_size*seq_length)) # TODO

                    val_losses.append(val_loss.item())

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
                
                '''TODO: sample from the model to generate texts'''
                input_eval ="wink'st" # TODO: choose a start string
                print(sample(model, 1000, prime=input_eval, top_k=10))
                
                model.train()  # reset to train mode after iterationg through validation data

def predict(model, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[char2idx[char]]])
    x = one_hot_encode(x, len(model.vocab))
    inputs = torch.from_numpy(x)

    if (train_on_gpu):
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    '''TODO: feed the current input into the model and generate output'''
    output, h = model(inputs, h) # TODO

    # get the character probabilities
    p = F.softmax(output, dim=1).data
    if (train_on_gpu):
        p = p.cpu()  # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(model.vocab))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return idx2char[char], h


def sample(model, size, prime='The', top_k=None):
    if (train_on_gpu):
        model.cuda()
    else:
        model.cpu()

    model.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = model.init_hidden(1)
    for ch in prime:
        char, h = predict(model, ch, h, top_k=top_k)

    chars.append(char)

    for ii in range(size):
      '''TODO: pass in the previous character and get a new one'''
      char, h = predict(model, chars[-1], h, top_k=top_k)# TODO
      chars.append(char)

    model.train()
    return ''.join(chars)

''''TODO: Try changing the number of units in the network to see how it affects performance'''
n_hidden =512 # TODO
n_layers =2 # TODO

vanilla_model = VanillaCharRNN(vocab, n_hidden, n_layers)
print(vanilla_model)


lstm_model = LSTMCharRNN(vocab, n_hidden, n_layers)
print(lstm_model)

''''TODO: Try changing the hyperparameters in the network to see how it affects performance'''
batch_size =50 #TODO
seq_length =130 # TODO
n_epochs =20 #TODO  # start smaller if you are just testing initial behavior
#train(vanilla_model, text_as_int, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)

train(lstm_model, text_as_int, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)