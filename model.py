from torch import nn

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_class, dropout_p):
        super(FastText, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_size, sparse=True)
        nn.init.kaiming_normal_(self.embedding.weight)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(embedding_size, num_class, bias=True)
        nn.init.kaiming_normal_(self.linear.weight)

        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.dropout(embedded)
        return self.linear(embedded)