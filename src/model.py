import os
import re
import numpy as np
from sklearn import linear_model
from scipy import sparse
import collections
import codecs
import random


class HMM(object):
    """
     HMM Model
    """
    def __init__(self, dic, decode_type):
        """
        Initialize the model.
        """

        self.num_words = len(dic['word_to_id'])
        self.num_tags = len(dic['tag_to_id'])

        self.initial_prob = np.ones([self.num_tags])
        self.transition_prob = np.ones([self.num_tags, self.num_tags])
        self.emission_prob = np.ones([self.num_tags, self.num_words])
        self.decode_type = decode_type
        self.q = 0
        # This is dummy code to create uniform probability distributions. Feel free to remove it.
        self.initial_prob /= np.sum(self.initial_prob)

        for i,p in enumerate(self.transition_prob):
            p /= np.sum(p)

        for i,p in enumerate(self.emission_prob):
            p /= np.sum(p)

        return

    def train(self, corpus):
        """
        TODO: Train a bigram HMM model using MLE estimates.
        Update self.initial_prob, self.transition_prob and self.emission_prob appropriately.

        corpus is a list of dictionaries of the form:
        {'str_words': str_words,   ### List of string words
        'words': words,            ### List of word IDs
        'tags': tags}              ### List of tag IDs
        All three lists above have length equal to the sentence length for each instance.

        """
        # BEGIN CODE
        transition_counts = np.zeros([self.num_tags, self.num_tags]) #initialize matrix for matrix counts
        emission_counts = np.zeros([self.num_tags, self.num_words]) #initialize matrix for emission counts
        initial_counts = np.zeros([self.num_tags])
        for sentence in corpus:
            sentence_tags = sentence["tags"]
            sentence_words = sentence["words"]
            idx = 0
            # Loop to count emission and transition
            for t_tags, t_words in zip(sentence_tags, sentence_words):
                emission_counts[t_tags][t_words] +=1 #add emission counts
                if idx == 0:
                    initial_counts[t_tags] += 1
                if idx > 0:
                    transition_counts[sentence_tags[idx - 1]][t_tags] += 1
                idx +=1
        self.initial_prob =  (1/np.sum(initial_counts)) * initial_counts
        emission_sum = np.sum(emission_counts, axis=1)
        transition_sum = np.sum(transition_counts, axis=1)
        for i in range(self.num_tags):
            self.emission_prob[i] = (emission_counts[i]) / (emission_sum[i])
            self.transition_prob[i] = (transition_counts[i]) / (transition_sum[i])

        # END CODE

        return
    def greedy_decode(self, sentence):
        """
        Decode a single sentence in Greedy fashion
        Return a list of tags.
        """
        tags = []

        init_scores = [self.initial_prob[t] * self.emission_prob[t][sentence[0]] for t in range(self.num_tags)]
        tags.append(np.argmax(init_scores))

        for w in sentence[1:]:
            scores = [self.transition_prob[tags[-1]][t] * self.emission_prob[t][w] for t in range(self.num_tags)]
            tags.append(np.argmax(scores))

        assert len(tags) == len(sentence)
        return tags



    def viterbi_decode(self, sentence):
        """
        TODO: Decode a single sentence using the Viterbi algorithm.
        Return a list of tags.
        """
        tags = []

        # BEGIN CODE
        #Initial scores
        init_scores = [self.initial_prob[t] * self.emission_prob[t][sentence[0]] for t in range(self.num_tags)]
        #Initialize array to compute viterbi
        len_sent = len(sentence)
        viterb_arr = np.zeros([self.num_tags,len_sent])
        back_tag_arr = np.zeros([self.num_tags,(len_sent-1)])
        viterb_max_list = np.zeros(len(sentence))
        for idx, w in enumerate(sentence):
            # Initial probabilities
            if idx == 0:
                for t in range(self.num_tags):
                    viterb_arr[t][idx] = init_scores[t] * self.emission_prob[t][w]
            else:
                for t in range(self.num_tags):
                    possible_list = []
                    for p in range(self.num_tags):
                        possible_list.append(viterb_arr[p][idx-1] * self.transition_prob[p][t] * self.emission_prob[t][w])
                    viterb_arr[t][idx] = np.max(possible_list) #Get the maximum value
                    back_tag_arr[t][idx-1] = np.argmax(possible_list)
        tags_revr = []
        for idx in list(range(len_sent))[::-1]:
            if idx == (len_sent - 1):
                max_b = np.argmax(viterb_arr[:,idx])
                tags_revr.append(max_b)
            else:
                max_b = back_tag_arr[int(max_b)][idx]
                tags_revr.append(max_b)
        tags = tags_revr[::-1]

        # END CODE
        assert len(tags) == len(sentence)
        return tags

    def tag(self, sentence):
        """
        Tag a sentence using a trained HMM.
        """
        if self.decode_type == 'viterbi':
            return self.viterbi_decode(sentence)
        else:
            return self.greedy_decode(sentence)



#assuming the context window is 1
class FFN(object):
    """
     Window-based feed forward neural network classifier
    """
    def __init__(self, dic, embedding, hidden_size=15, window=2):
        """
        Initialize the model.
        """

        self.num_words = len(dic['word_to_id'])
        self.num_tags = len(dic['tag_to_id'])

        self.dic=dic
        self.window = window
        self.hidden_size = hidden_size
        self.learning_rate = 0.15
        self.eps = 0.0006

        # This contains a dictionary of word embeddings {str_word -> embedding}
        self.embedding=embedding
        self.embedding_size = list(self.embedding.values())[0].shape[1]

        # TODO: make sure to initialize these appropriately.
        #INITIALZIED BELOW. I use he et al (2015) initalization!
        np.random.seed(117)
        self.w=np.random.randn(((2*self.window)+1) * self.embedding_size, self.hidden_size) * np.sqrt(2/( (((2*self.window)+1) * self.embedding_size) + self.hidden_size))  # weights for hidden layer
        self.b1=np.random.rand(self.hidden_size)  # bias for hidden layer

        self.u = np.random.randn(self.hidden_size, 5) * np.sqrt(2/( self.hidden_size + 5)) # weights for output layer
        self.b2 = np.random.rand(5) # bias for output layer



        return


    def make_windowed_data(self, sentence, tags):
        """
        TODO: Convert a single sentence and corresponding tags into a batch of inputs and outputs to the FFN

        """

        input_vector=np.zeros([len(sentence), (2*self.window+1) * self.embedding_size])
        output_vector=np.zeros([len(sentence), self.num_tags])


        #BEGIN CODE
        len_sent = len(sentence)
        for idx, (w, t) in enumerate(zip(sentence, tags)):
            output_vector[idx][t] = 1
            #INPUT VECTOR
            #Start token
            for i in [1,2,3,4,5]:
                k = i - 3
                if (0 <= idx + k < len_sent):
                    key_string = (str(sentence[idx + k])).lower()
                    if key_string in self.embedding:
                        input_vector[idx][(i-1) * self.embedding_size: (i) * self.embedding_size] = self.embedding[key_string]
                    else:
                        input_vector[idx][(i-1) * self.embedding_size: (i) * self.embedding_size] = np.zeros(self.embedding_size)
                else:
                    input_vector[idx][(i-1) * self.embedding_size: (i) * self.embedding_size] = np.zeros(self.embedding_size)
            #OUTPUT vector

        #END CODE


        return input_vector,output_vector



    def train(self, corpus):
        """
        TODO: Train the FFN with stochastic gradient descent.
        For each sentence in the corpus, convert it to a batch of inputs, compute the log loss and apply stochastic gradient descent on the parameters.
        """
        # Useful functions
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        def sigmoid_derivative(x):
            return sigmoid(x) *(1-sigmoid (x))

        def softmax(A):
            expA = np.exp(A)
            return expA / expA.sum(axis=0, keepdims=True)

        def stablesoftmax(A):
            """Compute the softmax of vector x in a numerically stable way."""
            shiftA = A - np.max(A)
            exps = np.exp(shiftA)
            return exps / np.sum(exps)

        eps = self.eps
        # BEGIN CODE
        step_size = self.learning_rate

        #1. TODO: Initialize any useful variables here.
        i =0
        converge_count = 0
        # FOR EACH EPOCH:
        while i < 35 :
            #FOR EACH sentence in CORPUS:
            if converge_count < 2:
                random.shuffle(corpus) #Randomize ordered
            #if i % 1 ==0:
            #print("ROUND",i)
            i +=1

            for k, sentence in enumerate(corpus):
                str_words = sentence["str_words"]
                sent_tags = sentence["tags"]
                #2. TODO: Make windowed batch data
                in_out_obj = self.make_windowed_data(str_words, sent_tags)
                input_vector = in_out_obj[0]
                output_vector = in_out_obj[1]
                #2A create gradients for the entire sentences
                grad_b2_sum = 0
                grad_b1_sum = 0
                grad_w_sum = 0
                grad_u_sum = 0
                len_in = len(input_vector)
                #3. TODO: Do a forward pass through the network.
                # loop in the sentence itself
                for in_vec, out_vec in zip(input_vector, output_vector):
                    sig_input = np.matmul(in_vec, self.w) + self.b1
                    #print(sig_input.shape)
                    h_t = sigmoid(sig_input)
                    #print(h_t.shape)
                    y_hat = softmax(np.matmul(h_t, self.u) + self.b2)
                    #rint(y_hat.shape)
                    #4. TODO: Do a backward pass through the network to compute required gradients.
                    dJ_dB = y_hat - out_vec
                    dJ_dh = np.matmul(dJ_dB, np.transpose(self.u))
                    dJ_dA = np.multiply(sigmoid_derivative(sig_input), dJ_dh)
                    grad_b1_sum += dJ_dA
                    grad_b2_sum += dJ_dB
                    grad_w_sum += np.outer(np.transpose(in_vec), dJ_dA) ##OBtain outer product
                    grad_u_sum += np.outer(np.transpose(h_t), (dJ_dB)) #outer product of 2 vectors
                    #if np.isnan(np.linalg.norm(grad_u_sum)) == False:
                    #    print("ROUND")
                    #    print(y_hat)
                    #    print(out_vec)
                    #    print(np.linalg.norm(grad_u_sum))
                    #    print(np.linalg.norm(grad_b1_sum))
                    #    print(np.linalg.norm(grad_b2_sum))

                #5. TODO: Update the weights (self.w, self.b1, self.u, self.b2)s
                self.b2 = self.b2 - (step_size * (grad_b2_sum/ len_in))
                self.w = self.w - (step_size  * (grad_w_sum/len_in))
                self.b1 = self.b1 - (step_size * (grad_b1_sum/len_in))
                self.u = self.u - (step_size * (grad_u_sum/len_in))
                if np.all(np.absolute(grad_u_sum/len_in) < eps) and np.all(np.absolute(grad_b2_sum/len_in) < eps) and np.all(np.absolute(grad_w_sum/len_in) < eps) and np.all(np.absolute(grad_b1_sum/len_in) < eps):
                    self.b2 = self.b2 + (step_size * (grad_b2_sum/ len_in))
                    self.w = self.w + (step_size * (grad_w_sum/len_in))
                    self.b1 = self.b1 + (step_size * (grad_b1_sum/len_in))
                    self.u = self.u + (step_size * (grad_u_sum/len_in))
                    #print("CONVERGED!")
                    converge_count +=1
                    eps = eps * 0.9
                if converge_count > 1:
                    break
                else:
                    continue
                break
            else:
                continue
            break
        # END CODE

        return

    def tag(self, sentence):
        """
        TODO: Tag a sentence using a trained FFN model.
        Since this model is not sequential (why?), you do not need to do greedy or viterbi decoding.
        """
        tags = []

        # Helper functions.
        def sigmoid(x):
            return 1/(1+np.exp(-x))

        def sigmoid_derivative(x):
            return sigmoid(x) *(1-sigmoid (x))

        def softmax(A):
            expA = np.exp(A)
            return expA / expA.sum(axis=0, keepdims=True)

        #BEGIN CODE

        #1. Convert sentence into windowed data
        str_words = []
        for id in sentence:
            if id in self.dic['id_to_word']:
                word = self.dic['id_to_word'][id]
            else:
                print("NOT FOUND")
                word = "-1"
            str_words.append(str(word))
        input_vector=np.zeros([len(sentence), (2*self.window+1) * self.embedding_size])
        len_sent = len(sentence)
        for idx, w in enumerate(str_words):
            for i in [1,2,3,4,5]:
                k = i - 3
                if (0 <= idx + k < len_sent):
                    key_string = (str(str_words[idx + k])).lower()
                    if key_string in self.embedding:
                        input_vector[idx][(i-1) * self.embedding_size: (i) * self.embedding_size] = self.embedding[key_string]
                    else:
                        input_vector[idx][(i-1) * self.embedding_size: (i) * self.embedding_size] = np.zeros(self.embedding_size)
                else:
                    input_vector[idx][(i-1) * self.embedding_size: (i) * self.embedding_size] = np.zeros(self.embedding_size)
            #OUTPUT vector

        #2. Do a forward pass to predict entity tags
        for in_vec in input_vector:
            sig_input = np.matmul(in_vec, self.w) + self.b1
            #print(sig_input.shape)
            h_t = sigmoid(sig_input)
            #print(h_t.shape)
            y_hat = softmax(np.matmul(h_t, self.u) + self.b2)
            tags.append(np.argmax(y_hat))

        #END CODE

        assert len(tags) == len(sentence)
        return tags
