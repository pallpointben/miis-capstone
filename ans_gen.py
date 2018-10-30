import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import pdb    
import copy
import torch.optim.lr_scheduler as lr_scheduler
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()



def loadGloveModel(gloveFile):
    f = open(gloveFile,'rb')
    model = {}
    for line in f:
        line=line.decode()
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    return model
    
embedding = loadGloveModel('glove.840B.300d.txt')


class AnsNN(nn.Module):
    def __init__(self, Vocab_size,hops=3, dropout=0.2):
        super(AnsNN, self).__init__()
        self.Vocab_size=Vocab_size
        self.hops = hops
        embd_size = 150
        init_rng = 0.1
        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.ModuleList([nn.Embedding(self.Vocab_size, embd_size) for _ in range(hops+1)])
        for i in range(len(self.A)):
            self.A[i].weight.data.normal_(0, init_rng)
            self.A[i].weight.data[0] = 0 # for padding index
        self.B = self.A[0] # query encoder
        self.C = self.A[-1]

    def forward(self, x, query):
        # x (bs, story_len, s_sent_len)
        # q (bs, q_sent_len)
        
        '''
        candidates are answer candidates for AQ
                   are KB facts for knowledge verification
                   are question template fields for question verification
        '''
        
        bs = x.size(0)
        story_len = x.size(1)
        s_sent_len = x.size(2)
        
        x = x.view(bs*story_len, -1) # (bs*s_sent_len, s_sent_len)
        
        u = self.dropout(self.B(query)) # (bs, q_sent_len, embd_size)
        u = torch.sum(u, 1) # (bs, embd_size)
        
        # Adjacent weight tying
        for k in range(self.hops):
            m = self.dropout(self.A[k](x))            # (bs*story_len, s_sent_len, embd_size)
            m = m.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
           
            m = torch.sum(m, 2) # (bs, story_len, embd_size)
            
            c = self.dropout(self.A[k+1](x))           # (bs*story_len, s_sent_len, embd_size)
            c = c.view(bs, story_len, s_sent_len, -1)  # (bs, story_len, s_sent_len, embd_size)
            c = torch.sum(c, 2)                        # (bs, story_len, embd_size)

            p = torch.bmm(m, u.unsqueeze(2)).squeeze(-1) # (bs, story_len)
            p = F.softmax(p, -1).unsqueeze(1)          # (bs, 1, story_len)
            o = torch.bmm(p, c).squeeze(1)             # use m as c, (bs, embd_size)
            u = o + u # (bs, embd_size)
        
        W = torch.t(self.A[-1].weight) # (embd_size, vocab_size)
        output = F.softmax(torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs, 1, 1)).squeeze() )# (bs, ans_size)
        
        return output


class Net:
    def __init__(self,model,vocab,frequency):
        self.dict=vocab
        self.frequency=frequency

        self.QA_model= model
      
        self.opt1=optim.SGD(self.QA_model.parameters(),lr=0.01)


    def train(self):

        np.random.shuffle(self.train_data)
        train_data=self.train_data
        self.QA_model.train()

        print_loss=0
        num_batches=0
        for batch_input in self.get_batch(train_data):
            batch_tensor=self.get_tensor(batch_input)
            
            story_tensor=batch_tensor[0].cuda()
            if(len(story_tensor)==0):
                    continue
            query_tensor=batch_tensor[1].cuda()
            answers=batch_tensor[2].cuda()
            rewards=batch_tensor[3].cuda()
            candidates=batch_tensor[4]
            
            
            answer_preds=self.QA_model.forward(story_tensor,query_tensor)
            
            loss=0
            num_batches=0
            for index,candidate in enumerate(candidates): 
                try:
                    loss+=F.cross_entropy(F.softmax(answer_preds[index][candidate]).unsqueeze(0),answers[index].unsqueeze(0))*rewards[index]
                except:
                    loss+=F.cross_entropy(F.softmax(answer_preds[candidate]).unsqueeze(0),answers[index].unsqueeze(0))*rewards[index]
                num_batches+=1
                    
            loss/=num_batches
            self.opt1.zero_grad()
            
            #torch.nn.utils.clip_grad_norm_(self.AQ_question_verification.parameters(), 100)
            
            loss.backward()
        
            
            
            self.opt1.step()
            
            #print(loss.item())
            print_loss+=loss.item()
            num_batches+=1
        print_loss/=num_batches
        
        return print_loss


        




    def get_batch(self,data,batch_size=10):
        num_iter = (len(data) + batch_size - 1) // batch_size
        for i in range(num_iter):
            start_idx = i * batch_size
            batch_data = self.Lines2ExsMovieQA(data[start_idx:(start_idx + batch_size)])
            #batch_input=self.batchify(batch_data)
            #batch_input = [x.cuda(async=True) for x in batch_input]
            
            yield batch_data

            
    def get_tensor(self,instances):
        max_num_query_words=0
        max_num_story_words=0
        max_num_story_lines=0
        stories=[]
        queries=[]
        rewards=[]
        answers=[]
        Candidates=[]
        for instance in instances:
            candidates=instance['AnswerCandidate']
            candidates=[self.dict[candidate] for candidate in candidates]
            
            story=instance['hist_x']
            num_story_lines=len(story)
            max_num_story_lines=max(max_num_story_lines,num_story_lines)
            
            story_vec=[]       
            for j,line in story.items():
                story_vec.append(self.String2Vector(line))
                max_num_story_words=max(max_num_story_words,len(line.split()))

            query_vec=self.String2Vector(instance['question'])
            max_num_query_words=max(max_num_query_words,len(instance['question'].split()))

            try:
                answers.append(candidates.index(self.dict[instance['answer']]))
            except:
                continue
            
            stories.append(story_vec)
            queries.append(query_vec)
            rewards.append(instance['reward'])
            Candidates.append(torch.LongTensor(candidates))

        batch_size=len(queries)

        Answers=torch.LongTensor(answers)
        Query = torch.LongTensor(batch_size, max_num_query_words).fill_(0)
        Rewards = torch.FloatTensor(rewards)
        Story = torch.LongTensor(batch_size, max_num_story_lines, max_num_story_words).fill_(0)
        
        
        for i,story_vec in enumerate(stories):
            for j,ti in enumerate(story_vec):
                Story[i,j,:len(ti)].copy_(ti)
            
        
        for i,query in enumerate(queries):
            Query[i,:len(query)].copy_(query)
        
        
        
        return Story,Query,Answers,Rewards,Candidates

    def String2Vector(self,string):
        words=string.split()
        vector=torch.ones(len(words))
        for i in range(len(words)):
            if words[i] not in self.dict:
                vector[i]=1
            else:
                vector[i]=self.dict[words[i]];
        return vector
    
    def GetAnswerCandidates(self,ex):
        AnswerCandidate=[]
        memory=ex['hist_x']
        
        for j in range(len(memory)):
            mem=memory[j].split()

            for k in range(len(mem)):
                candidate=mem[k]
                if(candidate in self.dict and self.frequency[candidate]<10000):
                    AnswerCandidate.append(candidate)

        np.random.shuffle(AnswerCandidate)
        return AnswerCandidate
        
    def Lines2instanceMovieQA(self,lines):
        instance={}
        kb_x={}
        text_x={}
        get_reward=False
        for i in range(len(lines)): 
            line=lines[i]
            if line!="":
                if(get_reward):
                    reward=int(line.split('\t')[-1])
                    if(reward==1):
                        instance['reward']=1
                        return instance
                    else:
                        instance['reward']=-0.2
                        return instance
                
                if line.find("knowledgebase")!=-1:
                    kb_x[len(kb_x)]=' '.join(line.split('\t')[0].split()[2:])
                else:
                      
                    user_input=' '.join(line.split('\t')[0].split()[1:])
                    response=line.split('\t')[1]

                    if(i==len(lines)-2):
                        instance['hist_x']={}
                        instance['kb_x']={}
                        for j,v in kb_x.items():
                            instance['kb_x'][j]=v
                        for j,v in text_x.items():
                            instance['hist_x'][j]=v
                        for j,v in kb_x.items():
                            instance['hist_x'][j]=v
                            
                        instance['question']=user_input
                        instance['answer']=response
                        instance['AnswerCandidate']=self.GetAnswerCandidates(instance)
                        get_reward=True
                    else:
                        text_x[len(text_x)]=user_input
                        if(response!=''):
                             text_x[len(text_x)]=response

    
    def Lines2ExsMovieQA(self,instance_lines):
        dataset=[]
            
        for i in range(len(instance_lines)):
            instance=self.Lines2instanceMovieQA(instance_lines[i])
            dataset.append(instance)
            
        return dataset
        

    def load_lines(self,fname):
        f = open(fname,encoding='utf-8')
        Lines = []
        lines=[]
        while True:
            line = f.readline()
            if line == '':
                break
            if(int(line.split()[0])==1 and lines!=[]):
                Lines.append(lines)
                lines=[]
            
            line = line.translate(str.maketrans('','','!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'))
            line=line.lower()
            line=line[:-1]
            
            lines.append(line)
        
        f.close()
        
        return Lines
    

        
        
    def process_data(self):
        self.train_data=self.load_lines("data/AQ_supervised_data/Task2_AQ_train.txt")  
        self.val_data=self.load_lines("data/AQ_supervised_data/Task2_AQ_dev.txt")  
        self.test_data=self.load_lines("data/AQ_supervised_data/Task2_AQ_test.txt")  


def load_dict(fname):
    f = open(fname,'rb')
    cnt = 0
    vocab = {}
    #rev_dict = {}
    frequency={}
    while True:
        string = str(f.readline())
        string=string[2:-2]
        if(string==''):
            break
        s1=string.split('\\t')
        
        #self.dict[cnt] = s1[0]
        vocab[s1[0]] = cnt
        #rev_dict[cnt] = s1[0]
        cnt = cnt + 1
        frequency[s1[0]]=int(s1[1][:-1])
    f.close()
    return vocab,frequency
        
n_epochs=50
epoch=0
index=0
Accuracy=0
iterations=0
done=False
best_dev_acc=0

vocab,frequency=load_dict("data/movieQA_kb/movieQA.dict")

model=AnsNN(len(vocab)).cuda()

network=Net(model,vocab,frequency)

network.process_data()
best_val_loss=float('inf')

for i in range(n_epochs):
        print('Epoch %d...' % i)
        

        loss=network.train()     
        print('Train Loss: %f' % (loss))
        torch.save(model.state_dict(), 'AQ_answer_generation_supervised_negative_reward.pt')
        '''
        val_loss=network.evaluate()
        print('Val Loss: %f' % (val_loss))

        if(val_loss<best_val_loss):
            torch.save(model.state_dict(), 'AQ_answer_generation_supervised.pt')
            best_val_loss=val_loss
            
        outputs,inputs=network.test()
        print('Query: '+ inputs)
        print('Matched_template: '+ outputs)
            
        '''
        
        
        
        
        