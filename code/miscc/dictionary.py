class Dictionary(object):

    def __init__(self, id2word, word2id): #, lang):
        assert len(id2word) == len(word2id)
        self.id2word = id2word
        self.word2id = word2id
        #self.lang = lang
        self.check_valid()
        self.SOS_TOKEN = None
        self.EOS_TOKEN = None
        self.PAD_TOKEN = None
        self.UNK_TOKEN = None

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare the dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        #return self.lang == y.lang and all(self.id2word[i] == y[i] for i in range(len(y)))
        return True # hack

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert len(self.id2word) == len(self.word2id)
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i

    def index(self, word):
        """
        Returns the index of the specified word.
        """
        return self.word2id[word]
