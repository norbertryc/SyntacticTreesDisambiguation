import pickle
import theano
from theano import tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams 
import numpy as np

dataType = 'int64'

from collections import OrderedDict

class TreeLSTM(object):  

    def __init__(self, h_dim, nc, w2v_model_path, file_with_rules, 
                 rules_emb_dim, max_phrase_length, emb_dropout_rate, h_dropout_rate, l, srng,
                load_params=None): 

        '''

        - dropout stanu ukrytego (LSTM_1)
        - dropout embeddinga (LSTM_1)
        - regularyzacja l2 (LSTM_1)
        - indywidualna obsluga lisci - struktura taka sama, macierze te same, ale uczymy: h_aggregated_0, hidden_state_0, cell_state_0, zamiast brac w te miejsca 0


        nh :: dimension of hidden state
        nc :: number of classes
        '''

        self.max_phrase_length = max_phrase_length

        w2vecs = pickle.load(open(w2v_model_path,"rb"))
        self.emb = theano.shared(w2vecs["vectors"].astype(theano.config.floatX))
        self.words2ids = w2vecs["words2ids"]

        emb_dim = w2vecs["vectors"].shape[1]
        del w2vecs

        
        r = open(file_with_rules,"r")
        rules = [x.split() for x in r.readlines()]
        r.close()
        unique_rules = set()
        for i in range(len(rules)):
            for j in range(len(rules[i])):
                unique_rules.add(rules[i][j])
                
        number_of_uniue_rules = len(unique_rules)
 
        r = 0.05

        self.rules2ids = dict(zip(unique_rules,range(number_of_uniue_rules)))
        self.emb_rules = theano.shared(r * np.random.uniform(-1,1,(number_of_uniue_rules, rules_emb_dim)).astype(theano.config.floatX))
        
   

        self.W_i = theano.shared(r * np.random.uniform(-1.0, 1.0, (emb_dim+rules_emb_dim, h_dim) ).astype(theano.config.floatX))
        self.U_i = theano.shared(r * np.random.uniform(-1.0, 1.0, (h_dim, h_dim) ).astype(theano.config.floatX))
        self.b_i = theano.shared(r * np.random.uniform(-1.0, 1.0, h_dim ).astype(theano.config.floatX))

        self.W_f = theano.shared(r * np.random.uniform(-1.0, 1.0, (emb_dim+rules_emb_dim, h_dim) ).astype(theano.config.floatX))
        self.U_f = theano.shared(r * np.random.uniform(-1.0, 1.0, (h_dim, h_dim) ).astype(theano.config.floatX))
        self.b_f = theano.shared(r * np.random.uniform(-1.0, 1.0, h_dim ).astype(theano.config.floatX))

        self.W_o = theano.shared(r * np.random.uniform(-1.0, 1.0, (emb_dim+rules_emb_dim, h_dim) ).astype(theano.config.floatX))
        self.U_o = theano.shared(r * np.random.uniform(-1.0, 1.0, (h_dim, h_dim) ).astype(theano.config.floatX))
        self.b_o = theano.shared(r * np.random.uniform(-1.0, 1.0, h_dim ).astype(theano.config.floatX))

        self.W_u = theano.shared(r * np.random.uniform(-1.0, 1.0, (emb_dim+rules_emb_dim, h_dim) ).astype(theano.config.floatX))
        self.U_u = theano.shared(r * np.random.uniform(-1.0, 1.0, (h_dim, h_dim) ).astype(theano.config.floatX))
        self.b_u = theano.shared(r * np.random.uniform(-1.0, 1.0, h_dim ).astype(theano.config.floatX))

        self.W_y   = theano.shared(r * np.random.uniform(-1.0, 1.0, (h_dim, nc)).astype(theano.config.floatX))
        self.b_y   = theano.shared(r * np.random.uniform(-1.0, 1.0, nc).astype(theano.config.floatX))




        self.h_aggregated_0 = theano.shared(r * np.random.uniform(-1.0, 1.0, h_dim ).astype(theano.config.floatX))
        self.cell_state_0 = theano.shared(r * np.random.uniform(-1.0, 1.0, h_dim ).astype(theano.config.floatX))
        self.hidden_state_0 = theano.shared(r * np.random.uniform(-1.0, 1.0, h_dim ).astype(theano.config.floatX))



        self.srng = srng
        self.h_dropout_rate = h_dropout_rate
        self.emb_dropout_rate = emb_dropout_rate
        self.l = l


        if load_params:
            load_params = dict(pickle.load(open(load_params,"rb")))
            for key in load_params.keys():
                if key not in ['emb', 'emb_rules', 'W_i', 'U_i', 'b_i', 'W_f', 'U_f', 'b_f', 'W_o', 'U_o', 'b_o', 'W_u', 'U_u', 'b_u', 'W_y', 'b_y', 'h_aggregated_0', 'cell_state_0', 'hidden_state_0']:
                    setattr(self, key, load_params[key])
                else:
                    setattr(self, key, theano.shared(load_params[key]))
        
        

        def one_step(word_id, rule_id, word_children_positions, y_true, k, hidden_states, cell_states, learning_rate):

            x = T.concatenate( [self.emb[word_id], self.emb_rules[rule_id] ])

            #dropout:
            mask1 = self.srng.binomial(n=1, p=1-self.emb_dropout_rate, size=(emb_dim+rules_emb_dim,), dtype='floatX')
            x = x * mask1


            tmp = word_children_positions>=0.0
            number_of_children = tmp.sum(dtype = theano.config.floatX) 
            idx_tmp = tmp.nonzero()                                                                   # indeksy realne dzieci - czyli te, gdzie nie ma -1        

            h_aggregated = ifelse(T.gt(number_of_children, 0.0), hidden_states[word_children_positions[idx_tmp]].sum(axis=0), self.h_aggregated_0)


            i = T.nnet.sigmoid(	T.dot(x, self.W_i) + T.dot(h_aggregated, self.U_i) + self.b_i)             

            o = T.nnet.sigmoid(	T.dot(x, self.W_o) + T.dot(h_aggregated, self.U_o) + self.b_o)             

            u = T.tanh(	T.dot(x, self.W_u) + T.dot(h_aggregated, self.U_u) + self.b_u)             

            f_c = ifelse(T.gt(number_of_children, 0.0), 
                (T.nnet.sigmoid( T.dot(x, self.W_f ) + T.dot(hidden_states[word_children_positions[idx_tmp]], self.U_f)  + self.b_f )*cell_states[word_children_positions[idx_tmp]]).sum(axis=0),
                T.nnet.sigmoid( T.dot(x, self.W_f ) + T.dot(self.hidden_state_0, self.U_f)  + self.b_f ) * self.cell_state_0
            )

            c = i*u + f_c

            h = o * T.tanh(c)
            #dropout:
            mask = self.srng.binomial(n=1, p=1-self.h_dropout_rate, size=(h_dim,), dtype='floatX')
            h = h * mask


            current_cell_state = cell_states[k]
            cell_states_new = T.set_subtensor(current_cell_state, c)

            current_hidden_state = hidden_states[k]
            hidden_states_new = T.set_subtensor(current_hidden_state, h)


            y_prob = T.nnet.softmax(T.dot(h,self.W_y) + self.b_y)[0]

            cross_entropy = -T.log(y_prob[y_true])						      

            return cross_entropy, hidden_states_new, cell_states_new  


        y = T.vector('y',dtype=dataType)
        learning_rate = T.scalar('lr',dtype=theano.config.floatX)
        words = T.vector(dtype=dataType)
        rules = T.vector(dtype=dataType)
        children_positions = T.matrix(dtype=dataType)
        words_indexes = T.vector(dtype=dataType)

        [cross_entropy_vector, _, _] , _ = theano.scan(fn=one_step, \
                                 sequences = [words, rules, children_positions,y,words_indexes],
                                 outputs_info = [None, 
                                     theano.shared(np.zeros((self.max_phrase_length+1,h_dim), dtype = theano.config.floatX)),
                                     theano.shared(np.zeros((self.max_phrase_length+1,h_dim), dtype = theano.config.floatX))],
                                 non_sequences = learning_rate,
                                 n_steps = words.shape[0])

        cost = T.mean(cross_entropy_vector) + self.l * (self.emb_rules**2).sum() #*0.5 * self.l * ((self.W_i**2).sum()+(self.W_f**2).sum()+(self.W_o**2).sum()+(self.W_u**2).sum()+(self.W_y**2).sum()+(self.U_i**2).sum()+(self.U_f**2).sum()+(self.U_o**2).sum()+(self.U_u**2).sum())

        updates = OrderedDict([
            (self.W_i, self.W_i-learning_rate*T.grad(cost, self.W_i)),
            (self.W_f, self.W_f-learning_rate*T.grad(cost, self.W_f)),
            (self.W_o, self.W_o-learning_rate*T.grad(cost, self.W_o)),
            (self.W_u, self.W_u-learning_rate*T.grad(cost, self.W_u)),
            (self.W_y, self.W_y-learning_rate*T.grad(cost, self.W_y)),

            (self.U_i, self.U_i-learning_rate*T.grad(cost, self.U_i)),
            (self.U_f, self.U_f-learning_rate*T.grad(cost, self.U_f)),
            (self.U_o, self.U_o-learning_rate*T.grad(cost, self.U_o)),
            (self.U_u, self.U_u-learning_rate*T.grad(cost, self.U_u)),

            #(self.emb, self.emb-learning_rate*T.grad(cost, self.emb)), #SPROBOWAC TU 0.1 ZAMIAST LR, A DLA POLSKICH BEZ AKTUALIZACJI EMB
            (self.emb_rules, self.emb_rules-learning_rate*T.grad(cost, self.emb_rules)),
            (self.b_i, self.b_i-learning_rate*T.grad(cost,self.b_i)),
                        (self.b_f, self.b_f-learning_rate*T.grad(cost,self.b_f)),
                        (self.b_o, self.b_o-learning_rate*T.grad(cost,self.b_o)),
                        (self.b_u, self.b_u-learning_rate*T.grad(cost,self.b_u)),
                        (self.b_y, self.b_y-learning_rate*T.grad(cost,self.b_y)),

            (self.h_aggregated_0, self.h_aggregated_0-learning_rate*T.grad(cost,self.h_aggregated_0)),
            (self.cell_state_0, self.cell_state_0-learning_rate*T.grad(cost,self.cell_state_0)),
            (self.hidden_state_0, self.hidden_state_0-learning_rate*T.grad(cost,self.hidden_state_0))

            ])

        self.train = theano.function( inputs  = [words, rules, children_positions, y, words_indexes, learning_rate],
                                      outputs = [],
                                      updates = updates,
                                      allow_input_downcast=True,
                                      mode='FAST_RUN'
                                      )


        def one_step_classify(word_id, rule_id, word_children_positions, k, hidden_states, cell_states):

            x = T.concatenate( [self.emb[word_id], self.emb_rules[rule_id] ])

            x = (1-self.emb_dropout_rate) * x

            tmp = word_children_positions>=0.0
            number_of_children = tmp.sum(dtype = theano.config.floatX) 
            idx_tmp = tmp.nonzero()                                                                   # indeksy realne dzieci - czyli te, gdzie nie ma -1        

            h_aggregated = ifelse(T.gt(number_of_children, 0.0), hidden_states[word_children_positions[idx_tmp]].sum(axis=0), self.h_aggregated_0)

            i = T.nnet.sigmoid(	T.dot(x, self.W_i) + T.dot(h_aggregated, self.U_i) + self.b_i)             

            o = T.nnet.sigmoid(	T.dot(x, self.W_o) + T.dot(h_aggregated, self.U_o) + self.b_o)             

            u = T.tanh(	T.dot(x, self.W_u) + T.dot(h_aggregated, self.U_u) + self.b_u)             

            f_c = ifelse(T.gt(number_of_children, 0.0), 
                (T.nnet.sigmoid( T.dot(x, self.W_f ) + T.dot(hidden_states[word_children_positions[idx_tmp]], self.U_f)  + self.b_f )*cell_states[word_children_positions[idx_tmp]]).sum(axis=0),
                T.nnet.sigmoid( T.dot(x, self.W_f ) + T.dot(self.hidden_state_0, self.U_f)  + self.b_f ) * self.cell_state_0
            )

            c = i*u + f_c

            h = o * T.tanh(c)
            # podczas uczenia zerowalismy 1-dropout_rate procent wspolrzednych, wiec trzeba to 
            h = h * (1-self.h_dropout_rate)

            current_cell_state = cell_states[k]
            cell_states_new = T.set_subtensor(current_cell_state, c)

            current_hidden_state = hidden_states[k]
            hidden_states_new = T.set_subtensor(current_hidden_state, h)


            y_prob = T.nnet.softmax(T.dot(h,self.W_y) + self.b_y)[0]             

            return  y_prob, hidden_states_new, cell_states_new


        [y_probs_classify, _ , _ ], _ = theano.scan(
                 fn=one_step_classify, 
                                 sequences = [words, rules, children_positions, words_indexes],
                 outputs_info = [None,
                         theano.shared(np.zeros((self.max_phrase_length+1,h_dim), dtype = theano.config.floatX)),
                         theano.shared(np.zeros((self.max_phrase_length+1,h_dim), dtype = theano.config.floatX))])

        predictions, _ = theano.scan(lambda i: T.argmax(y_probs_classify[i]), 
                                     sequences = [words_indexes])
        
        probs, _ = theano.scan(lambda i: y_probs_classify[i], 
                                     sequences = [words_indexes])

        self.classify = theano.function(inputs=[words, rules, children_positions,words_indexes], 
                                     outputs=predictions,
                                     allow_input_downcast=True,
                                     mode='FAST_RUN' 
                                     )

        self.predict_proba = theano.function(inputs=[words, rules, children_positions,words_indexes], 
                             outputs=probs,
                             allow_input_downcast=True,
                             mode='FAST_RUN' 
                             )

        self.calculate_loss = theano.function(inputs=[words, rules, children_positions, y, words_indexes, learning_rate], 
                     outputs=cost,
                     allow_input_downcast=True,
                     mode='FAST_RUN' 
                     )
        
    def save_model(path):
        params = [ (k, v.get_value())  if type(v)==theano.tensor.sharedvar.TensorSharedVariable else (k,v) for k, v in list(self.__dict__.items())]
        params = dict(params)
        pickle.dump(params,open(path,"wb"))

        
        

