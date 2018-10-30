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
        
class QuestNN(nn.Module):
    def __init__(self, Vocab_size,hops=3, dropout=0.2):
        super(QuestNN, self).__init__()
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
                
    def forward(self, x, query,candidates,indices):
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
        
        
        out=u
        try:
            p=F.softmax(torch.sum(torch.matmul(self.C(candidates),out.transpose(0,1)),dim=1).squeeze(1).transpose(0,1),dim=-1)
        except:
            p=F.softmax(torch.sum(torch.matmul(self.C(candidates),out.transpose(0,1)),dim=1).transpose(0,1),dim=-1)

        out = torch.zeros((bs,14)).cuda()
        
        out.index_add_(-1, indices, p)
        
        return out
 
        
class Net:
    def __init__(self,model,vocab,frequency):
        self.dict=vocab
        self.frequency=frequency
        self.question_templates,self.question_fields,self.story_tensor,self.story_indices_dict,self.story_indices,self.story_templates = self.read_question_templates('question_templates.txt')

        self.AQ_question_verification= model
      
        self.opt1=optim.SGD(self.AQ_question_verification.parameters(),lr=0.01)


    def read_question_templates(self,filename):
        f=open(filename,'r')
        questions = f.readlines()
        Question_templates={}
        question_fields={}
        story_templates=[]
        story_indices={}
        story_indices_list=[]
        for question in questions:
            if(question=='\n'):
                continue
            question=question[:-1]
            if(question[:6]=='FIELD:'):
                field=question[6:].lower()
                if field not in question_fields:
                    question_fields[field]=len(question_fields)
                Question_templates[question_fields[field]]=[]
                story_indices[question_fields[field]]=[]
                continue
            Question_templates[question_fields[field]].append(question)
            story_templates.append(question)
            story_indices[question_fields[field]].append(len(story_templates)-1)
            story_indices_list.append(question_fields[field])
        
        story_tensor=self.get_story_tensor(story_templates).cuda()
        story_indices_list=torch.LongTensor(story_indices_list).cuda()
        
        return Question_templates,question_fields,story_tensor,story_indices,story_indices_list,story_templates
            

    def train(self):

        np.random.shuffle(self.train_data)
        train_data=self.train_data
        self.AQ_question_verification.train()

        print_loss=0
        num_batches=0
        for batch_input in self.get_batch(train_data):
            batch_tensor=self.get_tensor(batch_input)
        
            story_tensor=batch_tensor[0].cuda()
            if(len(story_tensor)==0):
                    continue
            query_tensor=batch_tensor[1].cuda()
            sup_question_choice=batch_tensor[2].cuda()
            rewards=batch_tensor[-1].cuda()
            
            
            question_verification_output_pred=self.AQ_question_verification.forward(story_tensor,query_tensor,self.story_tensor,self.story_indices)

            
            
            loss=F.cross_entropy(question_verification_output_pred,sup_question_choice,reduce=False,size_average=False)
            
            loss=torch.mean(torch.mul(loss,rewards))
            self.opt1.zero_grad()
            
            #torch.nn.utils.clip_grad_norm_(self.AQ_question_verification.parameters(), 100)
            
            loss.backward()
        
            
            
            self.opt1.step()
            
            #print(loss.item())
            print_loss+=loss.item()
            num_batches+=1
        print_loss/=num_batches
        
        return print_loss

    def evaluate(self):

        self.AQ_question_verification.eval()

        
        print_loss=0
        num_batches=0
        with torch.no_grad():
            for batch_input in self.get_batch(self.val_data):
                batch_tensor=self.get_tensor(batch_input)
                
                story_tensor=batch_tensor[0].cuda()
                if(len(story_tensor)==0):
                    continue
                query_tensor=batch_tensor[1].cuda()
                sup_question_choice=batch_tensor[2].cuda()
                rewards=batch_tensor[-1].cuda()
     
                
                
                question_verification_output_pred=self.AQ_question_verification.forward(story_tensor,query_tensor,self.story_tensor,self.story_indices)
    
                
                loss=F.cross_entropy(question_verification_output_pred,sup_question_choice)
                loss=torch.mean(torch.mul(loss,rewards))
                
                #print(loss.item())
                print_loss+=loss.item()
                num_batches+=1
            print_loss/=num_batches
        
        return print_loss

        
    def test(self):
        self.AQ_question_verification.eval()
        np.random.shuffle(self.test_data)
        with torch.no_grad():
            for batch_input in self.get_batch(self.test_data):
                batch_tensor=self.get_tensor(batch_input)
            
                story_tensor=batch_tensor[0].cuda()
                query_tensor=batch_tensor[1].cuda()
                test_queries=batch_tensor[3]

                question_verification_output_pred=self.AQ_question_verification.forward(story_tensor,query_tensor,self.story_tensor,self.story_indices)
    
                #1 random example
                best_cluster=int(torch.argmax(question_verification_output_pred,dim=-1)[0])
                choice_index=np.random.choice(self.story_indices_dict[best_cluster])
                
                outputs=self.story_templates[choice_index]
                inputs = test_queries[index]
                break

        return outputs,inputs



    def get_batch(self,data,batch_size=200):
        num_iter = (len(data) + batch_size - 1) // batch_size
        for i in range(num_iter):
            start_idx = i * batch_size
            batch_data = self.Lines2ExsMovieQA(data[start_idx:(start_idx + batch_size)])
            #batch_input=self.batchify(batch_data)
            #batch_input = [x.cuda(async=True) for x in batch_input]
            
            yield batch_data

    def get_story_tensor(self,story):     
        max_num_story_words=0    
        story_vec=[]
        for j,line in enumerate(story):
            story_vec.append(self.String2Vector(line))
            max_num_story_words=max(max_num_story_words,len(line.split()))

        story_tensor=torch.LongTensor(len(story_vec), max_num_story_words).fill_(0)
        for i,story in enumerate(story_vec):
            
            story_tensor[i,:len(story)].copy_(story)
            
        return story_tensor
            
    def get_tensor(self,instances):
        max_num_query_words=0
        max_num_story_words=0
        max_num_story_lines=0
        string_queries=[]
        stories=[]
        queries=[]
        rewards=[]
        sup_question_choices=[]
        for instance in instances:
            story=instance['hist_x']
            num_story_lines=len(story)
            max_num_story_lines=max(max_num_story_lines,num_story_lines)
            
            story_vec=[]       
            for j,line in story.items():
                story_vec.append(self.String2Vector(line))
                max_num_story_words=max(max_num_story_words,len(line.split()))
            
            string_query = instance['question']
            query=self.String2Vector(instance['question'])
            max_num_query_words=max(max_num_query_words,len(instance['question'].split()))

            question_response=instance['answer'].split()
            ent=[word for word in instance['question'].split() if '_' in word][0]
            ent_index=question_response.index(ent)
            question_response[ent_index]='X'

            found=False
            for field,templates in self.question_templates.items():
                if ' '.join(question_response[3:]) in templates:
                    sup_question_choice=field
                    found=True
            if not found:
                
                continue
            
            
            string_queries.append(string_query)
            stories.append(story_vec)
            queries.append(query)
            sup_question_choices.append(sup_question_choice)
            rewards.append(instance['reward'])

        batch_size=len(queries)


        Query = torch.LongTensor(batch_size, max_num_query_words).fill_(0)
        
        Story = torch.LongTensor(batch_size, max_num_story_lines, max_num_story_words).fill_(0)
        
        
        for i,story_vec in enumerate(stories):
            for j,ti in enumerate(story_vec):
                Story[i,j,:len(ti)].copy_(ti)
        
            
        
        for i,query in enumerate(queries):
            Query[i,:len(query)].copy_(query)
        
        return Story,Query,torch.LongTensor(sup_question_choices),string_queries,torch.FloatTensor(rewards)

    
    
        
    def String2Vector(self,string):
        words=string.split()
        vector=torch.ones(len(words))
        for i in range(len(words)):
            if words[i] not in self.dict:
                vector[i]=1
            else:
                vector[i]=self.dict[words[i]];
        return vector
    

    def Lines2instanceMovieQA(self,lines):
        instance1={}
        kb_x={}
        text_x={}
        get_reward=False
        for i in range(len(lines)): 
            line=lines[i]
            if line!="":
                
                if line.find("knowledgebase")!=-1:
                    kb_x[len(kb_x)]=' '.join(line.split('\t')[0].split()[2:])
                else:
                    
                    user_input=' '.join(line.split('\t')[0].split()[1:])
                    response=line.split('\t')[1]

                    ask_question=' '.join(response.split()[:3]) =='do you mean'

                    if(ask_question==True):

                        instance1['hist_x']={}
                        instance1['kb_x']={}
                        for j,v in kb_x.items():
                            instance1['kb_x'][j]=v
                        for j,v in text_x.items():
                            instance1['hist_x'][j]=v
                        for j,v in kb_x.items():
                            instance1['hist_x'][j]=v
                            
                        instance1['question']=user_input
                        instance1['answer']=response
                        get_reward=True
                        continue
                        
                    if(get_reward==True):
                        reward=int(line.split('\t')[-1])
                        if(reward==1):
                            instance1['reward']=1
                            return instance1
                        else:
                            instance1['reward']=-0.2
                            return instance1
                    else:
                        text_x[len(text_x)]=user_input
                        if(response!=''):
                             text_x[len(text_x)]=response


    def Lines2ExsMovieQA(self,instance_lines):
        dataset=[]
            
        for i in range(len(instance_lines)):
            instance=self.Lines2instanceMovieQA(instance_lines[i])
            if(instance!=-1):
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

model=QuestNN(len(vocab)).cuda()
model.load_state_dict(torch.load("AQ_question_verification_supervised_neg_reward.pt"))

network=Net(model,vocab,frequency)

network.process_data()
best_val_loss=float('inf')

for i in range(n_epochs):
        print('Epoch %d...' % i)

        loss=network.train()     
        print('Train Loss: %f' % (loss))
        
        val_loss=network.evaluate()
        print('Val Loss: %f' % (val_loss))

        if(val_loss<best_val_loss):
            torch.save(model.state_dict(), 'AQ_question_verification_supervised_neg_reward.pt')
            best_val_loss=val_loss
            
        outputs,inputs=network.test()
        print('Query: '+ inputs)
        print('Matched_template: '+ outputs)
        
        
            
        
        
        
        
        
        