import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import json
import gc
from tqdm import tqdm
from sklearn.cluster import KMeans
from encode import BERTMLMSentenceEncoderPrompt
from dataprocess import data_sampler_bert_prompt_deal_first_task_sckd
from model import proto_softmax_layer_bertmlm_prompt
from dataprocess import get_data_loader_bert_prompt
from util import set_seed
import wandb
wandb.login(
    anonymous = 'allow',
    relogin = True,
    key = '9e33535aa8c9fcaa7fc1dfa97a70d9de5107ad37'
)

def eval_model(config, basemodel, test_set, mem_relations,seen_relations_ids):
    # logging.info("One eval")
    # logging.info("test data num is:\t",len(test_set))
    basemodel.eval()

    test_dataloader = get_data_loader_bert_prompt(config, test_set, shuffle=False, batch_size=30)
    allnum= 0.0
    correctnum = 0.0
    with torch.no_grad():
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
                typelabels, masks, mask_pos) in enumerate(test_dataloader):

            sentences = sentences.to(config['device'])
            masks = masks.to(config['device'])
            mask_pos = mask_pos.to(config['device'])
            logits, rep , _ = basemodel(sentences, masks, mask_pos)

            distances = basemodel.get_mem_feature(rep)
            short_logits = distances

            
            for index, logit in enumerate(logits):
                score = short_logits[index]  # logits[index] + short_logits[index] + long_logits[index]
                allnum += 1.0

                golden_score = score[labels[index]]
                max_neg_score = -2147483647.0
                for i in seen_relations_ids :
                    if (i != labels[index]) and score[i] > max_neg_score:
                        max_neg_score = score[i]
                if golden_score >= max_neg_score:
                    correctnum += 1

    acc = correctnum / allnum
    basemodel.train()
    return acc

def get_memory(config, model, proto_set):
    memset = []
    resset = []
    rangeset= [0]
    for i in proto_set:
        memset += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_data_loader_bert_prompt(config, memset, False, False)
    features = []
    for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
               typelabels, masks, mask_pos) in enumerate(data_loader):
        sentences = sentences.to(config['device'])
        masks = masks.to(config['device'])
        mask_pos = mask_pos.to(config['device'])
        feature = model.get_feature(sentences, masks, mask_pos)
        features.append(feature)
    features = np.concatenate(features)

    protos = []
    for i in range(len(proto_set)):
        protos.append(torch.tensor(features[rangeset[i]:rangeset[i+1],:].mean(0, keepdims = True)))
    protos = torch.cat(protos, 0)
    return protos

def select_data(mem_set, proto_memory, config, model, divide_train_set, num_sel_data, current_relations, selecttype):
    ####select data according to selecttype
    #selecttype is 0: cluster for every rel
    #selecttype is 1: use ave embedding
    rela_num = len(current_relations)
    for i in range(0, rela_num):
        thisrel = current_relations[i]
        if thisrel in mem_set.keys():
            #print("have set mem before")
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
            proto_memory[thisrel] = []
        else:
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
        thisdataset = divide_train_set[thisrel]
        data_loader = get_data_loader_bert_prompt(config, thisdataset, False, False)
        features = []
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
                typelabels, masks, mask_pos) in enumerate(data_loader):
            sentences = sentences.to(config['device'])
            masks = masks.to(config['device'])
            mask_pos = mask_pos.to(config['device'])
            feature = model.get_feature(sentences, masks, mask_pos)
            features.append(feature)
        features = np.concatenate(features)
        num_clusters = min(num_sel_data, len(thisdataset))
        if selecttype == 0:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            distances = kmeans.fit_transform(features)
            for i in range(num_clusters):
                sel_index = np.argmin(distances[:, i])
                instance = thisdataset[sel_index]
                ###change tylelabel
                instance[11] = 3
                ###add to mem data
                mem_set[thisrel]['0'].append(instance)  ####positive sample
                cluster_center = kmeans.cluster_centers_[i]
                proto_memory[thisrel].append(instance)
        elif selecttype == 1:
            #print("use average embedding")
            samplenum = features.shape[0]
            veclength = features.shape[1]
            sumvec = np.zeros(veclength)
            for j in range(samplenum):
                sumvec += features[j]
            sumvec /= samplenum

            ###find nearest sample
            mindist = 100000000
            minindex = -100
            for j in range(samplenum):
                dist = np.sqrt(np.sum(np.square(features[j] - sumvec)))
                if dist < mindist:
                    minindex = j
                    mindist = dist
            #print(minindex)
            instance = thisdataset[j]
            ###change tylelabel
            instance[11] = 3
            mem_set[thisrel]['0'].append(instance)
            proto_memory[thisrel].append(instance)
        else:
            print("error select type")
    #####to get negative sample  mem_set[thisrel]['1']
    if rela_num > 1:
        ####we need to sample negative samples
        allnegres = {}
        for i in range(rela_num):
            thisnegres = {'h':[],'t':[]}
            currel = current_relations[i]
            thisrelposnum = len(mem_set[currel]['0'])
            #assert thisrelposnum == num_sel_data
            #allnum = list(range(thisrelposnum))
            for j in range(thisrelposnum):
                thisnegres['h'].append(mem_set[currel]['0'][j][3])
                thisnegres['t'].append(mem_set[currel]['0'][j][5])
            allnegres[currel] = thisnegres
        ####get neg sample
        for i in range(rela_num):
            togetnegindex = (i + 1) % rela_num
            togetnegrelname = current_relations[togetnegindex]
            mem_set[current_relations[i]]['1']['h'].extend(allnegres[togetnegrelname]['h'])
            mem_set[current_relations[i]]['1']['t'].extend(allnegres[togetnegrelname]['t'])
    return mem_set

def select_data_all(mem_set, proto_memory, config, model, divide_train_set, num_sel_data, current_relations, selecttype):
    ####select data according to selecttype
    #selecttype is 0: cluster for every rel
    #selecttype is 1: use ave embedding
    rela_num = len(current_relations)
    for i in range(0, rela_num):
        thisrel = current_relations[i]
        if thisrel in mem_set.keys():
            #print("have set mem before")
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
            proto_memory[thisrel].pop()
        else:
            mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}}
        thisdataset = divide_train_set[thisrel]
        # print(len(thisdataset))
        for i in range(len(thisdataset)):
            instance = thisdataset[i]
            ###change tylelabel
            instance[11] = 3
            ###add to mem data
            mem_set[thisrel]['0'].append(instance)
            proto_memory[thisrel].append(instance)
    if rela_num > 1:
        ####we need to sample negative samples
        allnegres = {}
        for i in range(rela_num):
            thisnegres = {'h':[],'t':[]}
            currel = current_relations[i]
            thisrelposnum = len(mem_set[currel]['0'])
            #assert thisrelposnum == num_sel_data
            #allnum = list(range(thisrelposnum))
            for j in range(thisrelposnum):
                thisnegres['h'].append(mem_set[currel]['0'][j][3])
                thisnegres['t'].append(mem_set[currel]['0'][j][5])
            allnegres[currel] = thisnegres
        ####get neg sample
        for i in range(rela_num):
            togetnegindex = (i + 1) % rela_num
            togetnegrelname = current_relations[togetnegindex]
            mem_set[current_relations[i]]['1']['h'].extend(allnegres[togetnegrelname]['h'])
            mem_set[current_relations[i]]['1']['t'].extend(allnegres[togetnegrelname]['t'])
    return mem_set

def train_model_with_hard_neg(config, model, mem_set, traindata, epochs, current_proto, seen_relation_ids, tokenizer, ifnegtive=0, threshold=0.2, use_loss5=True, only_mem=False):
    print('training data num: ' + str(len(traindata)))
    mem_data = []
    if len(mem_set) != 0:
        for key in mem_set.keys():
            mem_data.extend(mem_set[key]['0'])
    print('memory data num: '+ str(len(mem_data)))
    if only_mem==True:
        train_set = mem_data
    else:
        train_set = traindata + mem_data
    print('all train data: ' + str(len(train_set)))
    data_loader = get_data_loader_bert_prompt(config, train_set, batch_size=config['batch_size_per_step'])
    model.train()
    criterion = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    softmax = nn.Softmax(dim=0)
    lossfn = nn.MultiMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        model.set_memorized_prototypes_midproto(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []
        losses5 = []
        losses6 = []

        lossesfactor1 = 0.0
        lossesfactor2 = 1.0
        lossesfactor3 = 1.0
        lossesfactor4 = 0.0
        if use_loss5 == True:
            lossesfactor5 = 1.0
        else:
            lossesfactor5 = 0.0
        lossesfactor6 = 0.0
        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext, lengths,
            typelabels, masks, mask_pos) in enumerate(data_loader):
            model.zero_grad()
            labels = labels.to(config['device'])
            typelabels = typelabels.to(config['device'])  ####0:rel  1:pos(new train data)  2:neg  3:mem
            numofmem = 0
            numofnewtrain = 0
            allnum = 0
            memindex = []
            for index,onetype in enumerate(typelabels):
                if onetype == 1:
                    numofnewtrain += 1
                if onetype == 3:
                    numofmem += 1
                    memindex.append(index)
                allnum += 1

            sentences = sentences.to(config['device'])
            masks = masks.to(config['device'])
            mask_pos = mask_pos.to(config['device'])
            logits, rep , lmhead_output = model(sentences, masks, mask_pos)
            logits_proto = model.mem_forward(rep)

            loss1 = criterion(logits, labels)
            loss2 = criterion(logits_proto, labels)
            loss4 = lossfn(logits_proto, labels)
            loss3 = torch.tensor(0.0).to(config['device'])
            for index, logit in enumerate(logits):
                score = logits_proto[index]
                preindex = labels[index]
                maxscore = score[preindex]
                size = score.shape[0]
                maxsecondmax = [maxscore]
                secondmax = -100000
                for j in range(size):
                    if j != preindex and score[j] > secondmax:
                        secondmax = score[j]
                maxsecondmax.append(secondmax)
                for j in range(size):
                    if j != preindex and maxscore - score[j] < threshold:
                        maxsecondmax.append(score[j])
                maxsecond = torch.stack(maxsecondmax, 0)
                maxsecond = torch.unsqueeze(maxsecond, 0)
                la = torch.tensor([0]).to(config['device'])
                loss3 += criterion(maxsecond, la)
            loss3 /= logits.shape[0]
            
            loss5 = torch.tensor(0.0).to(config['device'])
            allusenum5 = 0
            # # --- test---
            # try:
            #     print("Prototype : ")
            #     print(model.prototypes.shape)
            # except:
            #     print("no model.prototypes")
            # # --- test---
                
            # --- add info_nce loss ---
            prototypes = model.prototypes.clone()
            infoNCE_loss = 0
            try:
                for i in range(rep.shape[0]):
                    neg_prototypes = [prototypes[rel_id] for rel_id in seen_relations_ids if rel_id != labels[i].item()]
                    neg_prototypes = torch.stack(neg_prototypes)
                    neg_prototypes.requires_grad_ = False
                    neg_prototypes = neg_prototypes.squeeze() # [num_neg_prototypes, dim]

                    f_pos = model.sentence_encoder.infoNCE_f(lmhead_output[i],rep[i])
                    f_neg = model.sentence_encoder.infoNCE_f(lmhead_output[i],neg_prototypes)
                    infoNCE_loss += -torch.log(f_pos / (f_pos + f_neg)).mean()
            except Exception as e:
                print(e.with_traceback())
                print("no infoNCE_loss")
            infoNCE_loss /= rep.shape[0]

            # --- add info_nce loss ---

            # --- add mlm loss ---
            mlm_labels = labels + 30522 - 1
            mlm_labels = mlm_labels.to(config['device'])
            mlm_labels.requires_grad = False
            mlm_loss = criterion(lmhead_output,mlm_labels)
            # --- add mlm loss ---
            for index in memindex:
                preindex = labels[index]
                if preindex in model.haveseenrelations:
                    loss5 += mseloss(softmax(rep[index]), softmax(model.prototypes[preindex]))
                allusenum5 += 1
            
            loss6 = torch.tensor(0.0).to(config['device'])
            allusenum6 = 0
            for index in memindex:
                preindex = labels[index]
                if preindex in model.haveseenrelations:
                    best_distrbution = model.mem_forward_update(rep[index].view(1, -1), model.bestproto)
                    current_distrbution = model.mem_forward_update(model.prototypes[preindex].view(1, -1), model.bestproto)
                    loss6 += mseloss(best_distrbution, current_distrbution)
                allusenum6 += 1
            
            if len(memindex) == 0:
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4
            else:
                loss5 = loss5 / allusenum5
                loss6 = loss6 / allusenum6
                loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4 + loss5 * lossesfactor5 + loss6 * lossesfactor6   + infoNCE_loss * config['infonce_lossfactor'] + mlm_loss * config['mlm_lossfactor']###with loss5
            loss.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            losses5.append(loss5.item())
            losses6.append(loss6.item())
            # print(f" InfoNCE_loss: {infoNCE_loss.item()} , mlm_loss: {mlm_loss.item()}")
            wandb.log({"Loss1": loss1.item(), "Loss2": loss2.item(), "Loss3": loss3.item(), "Loss4": loss4.item(), "Loss5": loss5.item(), "Loss6": loss6.item(), "InfoNCE_loss": infoNCE_loss.item(), "mlm_loss": mlm_loss.item()})

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])#cxd
            optimizer.step()
        return model

def train_memory(config, model, mem_set, train_set, epochs, current_proto, original_vocab_size,seen_relation_ids, ifusemem=True, threshold=0.2):
    train_set = []
    if ifusemem:
        mem_data = []
        if len(mem_set)!=0:
            for key in mem_set.keys():
                mem_data.extend(mem_set[key]['0'])
        train_set.extend(mem_data)
    data_loader = get_data_loader_bert_prompt(config, train_set, batch_size = config['batch_size_per_step'])
    model.train()
    criterion = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()
    softmax = nn.Softmax(dim=0)
    lossfn = nn.MultiMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])#cxd
    for epoch_i in range(epochs):
        model.set_memorized_prototypes_midproto(current_proto)
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []
        losses5 = []
        losses6 = []

        lossesfactor1 = 0.0
        lossesfactor2 = 1.0
        lossesfactor3 = 1.0
        lossesfactor4 = 0.0
        lossesfactor5 = 1.0
        lossesfactor6 = 1.0

        for step, (labels, neg_labels, sentences, firstent, firstentindex, secondent, secondentindex, headid, tailid, rawtext,
                   lengths, typelabels, masks, mask_pos) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            sentences = sentences.to(config['device'])
            masks = masks.to(config['device'])
            mask_pos = mask_pos.to(config['device'])
            logits, rep , lmhead_output = model(sentences, masks, mask_pos)
            logits_proto = model.mem_forward(rep)

            labels = labels.to(config['device'])
            loss1 = criterion(logits, labels)
            loss2 = criterion(logits_proto, labels)
            loss4 = lossfn(logits_proto, labels)
            loss3 = torch.tensor(0.0).to(config['device'])
            ###add triple loss
            for index, logit in enumerate(logits):
                score = logits_proto[index]
                preindex = labels[index]
                maxscore = score[preindex]
                size = score.shape[0]
                maxsecondmax = [maxscore]
                secondmax = -100000
                for j in range(size):
                    if j != preindex and score[j] > secondmax:
                        secondmax = score[j]
                maxsecondmax.append(secondmax)
                for j in range(size):
                    if j != preindex and maxscore - score[j] < threshold:
                        maxsecondmax.append(score[j])
                maxsecond = torch.stack(maxsecondmax, 0)
                maxsecond = torch.unsqueeze(maxsecond, 0)
                la = torch.tensor([0]).to(config['device'])
                loss3 += criterion(maxsecond, la)
            loss3 /= logits.shape[0]

            loss5 = torch.tensor(0.0).to(config['device'])

            for index, logit in enumerate(logits):
                preindex = labels[index]
                if preindex in model.haveseenrelations:
                    loss5 += mseloss(softmax(rep[index]), softmax(model.prototypes[preindex]))
            loss5 /= logits.shape[0] 

            loss6 = torch.tensor(0.0).to(config['device'])
            for index, logit in enumerate(logits):
                preindex = labels[index]
                if preindex in model.haveseenrelations:
                    best_distrbution = model.mem_forward_update(rep[index].view(1, -1), model.bestproto)
                    current_distrbution = model.mem_forward_update(model.prototypes[preindex].view(1, -1), model.bestproto)
                    loss6 += mseloss(best_distrbution, current_distrbution)
            loss6 /= logits.shape[0]
            
             # --- add info_nce loss ---
            prototypes = model.prototypes.clone()
            infoNCE_loss = 0
            try:
                for i in range(rep.shape[0]):
                    neg_prototypes = [prototypes[rel_id] for rel_id in seen_relations_ids if rel_id != labels[i].item()]
                    neg_prototypes = torch.stack(neg_prototypes)
                    neg_prototypes.requires_grad_ = False
                    neg_prototypes = neg_prototypes.squeeze() # [num_neg_prototypes, dim]

                    f_pos = model.sentence_encoder.infoNCE_f(lmhead_output[i],rep[i])
                    f_neg = model.sentence_encoder.infoNCE_f(lmhead_output[i],neg_prototypes)
                    infoNCE_loss += -torch.log(f_pos / (f_pos + f_neg)).mean()
            except Exception as e:
                print(e.with_traceback())
                print("no infoNCE_loss")
            infoNCE_loss /= rep.shape[0]

            # --- add info_nce loss ---

            # --- add mlm loss ---
            mlm_labels = labels + 30522 - 1
            mlm_labels = mlm_labels.to(config['device'])
            mlm_loss = criterion(lmhead_output,mlm_labels)
            # --- add mlm loss ---



            loss = loss1 * lossesfactor1 + loss2 * lossesfactor2 + loss3 * lossesfactor3 + loss4 * lossesfactor4  + loss5 * lossesfactor5 + loss6 * lossesfactor6 + infoNCE_loss * config['infonce_lossfactor'] + mlm_loss * config['mlm_lossfactor']
            loss.backward()
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            losses4.append(loss4.item())
            losses5.append(loss5.item())
            losses6.append(loss6.item())
            wandb.log({"Loss1": loss1.item(), "Loss2": loss2.item(), "Loss3": loss3.item(), "Loss4": loss4.item(), "Loss5": loss5.item(), "Loss6": loss6.item(), "InfoNCE_loss": infoNCE_loss.item(), "mlm_loss": mlm_loss.item()})
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])#cxd
            optimizer.step()
    return model


if __name__ == '__main__':
    f = open("config/config_fewrel_5and10.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    config['neg_sampling'] = False

    donum = 1
    epochs = 1
    threshold=0.1

    wandb.init(
    project = 'DATN',

    config = {
        'name': "ConPL",
        "task" : "FewRel",
        "shot" : "5",
        "infonce_lossfactor" : config['infonce_lossfactor'],
        "mlm_lossfactor" : config['mlm_lossfactor']
    }
)

    for m in range(donum):
        print(m)
        config["rel_cluster_label"] = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_" + str(m) + ".npy"
        config['training_file'] = "data/fewrel/CFRLdata_10_100_10_5/train_" + str(m) + ".txt"
        config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_5/valid_" + str(m) + ".txt"
        config['test_file'] = "data/fewrel/CFRLdata_10_100_10_5/test_" + str(m) + ".txt"
        config['relation_file'] = "data/fewrel/relation_name.txt"
        config["rel_index"] = "data/fewrel/rel_index.npy"
        config['rel_des_file'] = "data/fewre/relation_description.txt"
        config['num_of_relation'] = 80

        config['first_task_k-way'] = 10
        config['k-shot'] = 5
        encoderforbase = BERTMLMSentenceEncoderPrompt(config)

        original_vocab_size = len(list(encoderforbase.tokenizer.get_vocab()))
        print('Vocab size: %d'%original_vocab_size)
        if config["prompt"] == "hard-complex":
            template = 'the relation between e1 and e2 is mask . '
            print('Template: %s'%template)
        elif config["prompt"] == "hard-simple":
            template = 'e1 mask e2 . '
            print('Template: %s'%template)
        else:
            template = None
            print("no use soft prompt.")
        
        sampler = data_sampler_bert_prompt_deal_first_task_sckd(config, encoderforbase.tokenizer, template)
        modelforbase = proto_softmax_layer_bertmlm_prompt(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel, drop=0, config=config)
        modelforbase = modelforbase.to(config["device"])

        sequence_results = []
        sequence_results_average = []
        result_whole_test = []
        result_whole_test_average = []
        all_allresults_array = []

        fr_all = []
        distored_all = []
        for rou in range(6): #6 times different seeds to get average results

            num_class = len(sampler.id2rel)
            print('random_seed: ' + str(config['random_seed'] + 100 * rou))
            set_seed(config, config['random_seed'] + 100 * rou)
            sampler.set_seed(config['random_seed'] + 100 * rou)

            #cxd
            proto_acc = [[] for i in range(num_class)]
            proto_embedding = [[] for i in range(num_class)]

            mem_set = {} ####  mem_set = {rel_id:{'0':[positive samples],'1':[negative samples]}} 换5个head 换5个tail
            mem_relations = []   ###not include relation of current task

            past_relations = []

            savetest_all_data = None
            saveseen_relations = []

            proto_memory = []

            for i in range(len(sampler.id2rel)):
                proto_memory.append([sampler.id2rel_pattern[i]])
            # print('proto_memory', proto_memory)
            oneseqres = []
            whole_acc = []
            allresults_list = []
            ##################################
            whichdataselecct = 1
            ifnorm = True
            ##################################
            id2rel = sampler.id2rel
            rel2id = sampler.rel2id
            seen_test_data_by_task = []
            for steps, (training_data, valid_data, test_data,test_all_data, seen_relations,current_relations) in enumerate(sampler):
                print('current training data num: ' + str(len(training_data)))
                seen_relations_ids = [rel2id[relation] + 1 for relation in seen_relations] # seen relation (list of int) (include relation of current task)
                current_relations_ids = [rel2id[relation] + 1 for relation in current_relations] # current relation (list of int)
                
                seen_test_data_by_task.append(test_data)
                savetest_all_data = [] # test data of all tasks (array of shape 8000x16)
                for tmp in test_all_data:
                    savetest_all_data.extend(tmp)
                saveseen_relations = seen_relations

                currentnumber = len(current_relations)
                print('current relations num: '+ str(currentnumber))
                divide_train_set = {}
                for relation in current_relations_ids:
                    divide_train_set[relation] = []  ##int
                for data in training_data:
                    divide_train_set[data[0]].append(data)
                print('current divide num: '+ str(len(divide_train_set)))


                current_proto = get_memory(config, modelforbase, proto_memory) #这时候的current_proto是根据81个关系的名称输入模型之中得到的81个fake embedding：[81, 200]
                select_data_all(mem_set, proto_memory, config, modelforbase, divide_train_set,
                            config['rel_memory_size'], current_relations_ids, 0)  ##config['rel_memory_size'] == 1 
                            #proto_memory中的样本根据divide_train_set(training_data划分对应类)来增加每个类对应K个样本，mem_set[thisrel] = {'0': [], '1': {'h': [], 't': []}} 0放正样例，1放负样例，datatype=3

                ###add to mem data
                mem_set_length = {}
                proto_memory_length = []
                for i in range(len(proto_memory)):
                    proto_memory_length.append(len(proto_memory[i]))
                for key in mem_set.keys():
                    mem_set_length[key] = len(mem_set[key]['0'])
                print("mem_set_length", mem_set_length)
                print("proto_memory_length", proto_memory_length)

                for j in range(1):
                    current_proto = get_memory(config, modelforbase, proto_memory)
                    modelforbase = train_model_with_hard_neg(config, modelforbase, mem_set, training_data, epochs,
                                                                current_proto, tokenizer = encoderforbase.tokenizer,seen_relation_ids=seen_relations_ids, ifnegtive=0,threshold=threshold, use_loss5=False)
                
                select_data(mem_set, proto_memory, config, modelforbase, divide_train_set,
                            config['rel_memory_size'], current_relations_ids, 0)  ##config['rel_memory_size'] == 1 
                
                mem_set_length = {}
                proto_memory_length = []
                for i in range(len(proto_memory)):
                    proto_memory_length.append(len(proto_memory[i]))
                for key in mem_set.keys():
                    mem_set_length[key] = len(mem_set[key]['0'])
                print("mem_set_length", mem_set_length)
                print("proto_memory_length", proto_memory_length)
                for j in range(1):
                    current_proto = get_memory(config, modelforbase, proto_memory)
                    modelforbase = train_model_with_hard_neg(config, modelforbase, mem_set, training_data, epochs,
                                                                current_proto, tokenizer = encoderforbase.tokenizer,seen_relation_ids=seen_relations_ids, ifnegtive=0,threshold=threshold)
                
                #add train memory
                current_proto = get_memory(config, modelforbase, proto_memory)
                modelforbase = train_memory(config, modelforbase, mem_set, training_data, epochs*3, current_proto, original_vocab_size,ifusemem= True,seen_relation_ids=seen_relations_ids, threshold=threshold)

                
                current_proto = get_memory(config, modelforbase, proto_memory)
                modelforbase.set_memorized_prototypes_midproto(current_proto)
                modelforbase.save_bestproto(current_relations_ids)#save bestproto
                mem_relations.extend(current_relations_ids)

                currentalltest = []
                for mm in range(len(test_data)):
                    currentalltest.extend(test_data[mm])

                #compute mean accuarcy
                results = [eval_model(config, modelforbase, item, mem_relations,seen_relations_ids) for item in seen_test_data_by_task] # results of all previous task + this task after training on current task
                allresults_list.append(results)
                results_average = np.array(results).mean() # average accuracy of all tasks after training on current task
                wandb.log({f"Round {rou} Average Accuracy": results_average})
                whole_acc.append(results_average)

                #compute whole accuarcy
                seen_test_set = []
                for seen_relation in seen_relations_ids:
                    seen_test_set.extend(test_all_data[seen_relation - 1]) # test_all_data is a list of test data of all relation (test_all_data[0] is test data of relation 1])
                thisstepres = eval_model(config, modelforbase, seen_test_set, mem_relations,seen_relations_ids) # combine all test data of all tasks and evaluate
                oneseqres.append(thisstepres)

            sequence_results.append(np.array(oneseqres)) # combine all test data of all tasks and evaluate
            sequence_results_average.append(np.array(whole_acc)) # evaluate each task and average

            allres = eval_model(config, modelforbase, savetest_all_data, saveseen_relations,seen_relations_ids) # eval on all test data of all tasks
            result_whole_test.append(allres)

            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("after one epoch allres whole:\t" + str(allres))
            print(result_whole_test)

            allresults = [eval_model(config, modelforbase, item, num_class,seen_relations_ids) for item in seen_test_data_by_task]
            allresults_average = np.array(allresults).mean()
            result_whole_test_average.append(allresults_average)
            print("after one epoch allres average:\t" + str(allresults))
            print(result_whole_test_average)


            modelforbase = modelforbase.to('cpu')
            del modelforbase
            gc.collect()
            if config['device'] == 'cuda':
                torch.cuda.empty_cache()
            encoderforbase = BERTMLMSentenceEncoderPrompt(config)
            modelforbase = proto_softmax_layer_bertmlm_prompt(encoderforbase, num_class=len(sampler.id2rel), id2rel=sampler.id2rel, drop=0, config=config)
            modelforbase = modelforbase.to(config["device"])
        print("Final result: whole!")
        print(result_whole_test)
        for one in sequence_results:
            for item in one:
                sys.stdout.write('%.4f, ' % item)
            print('')
        avg_result_all_test = np.average(sequence_results, 0)
        for one in avg_result_all_test:
            sys.stdout.write('%.4f, ' % one)
        print('')
        print("Final result: average!")
        print(result_whole_test_average)
        for one in sequence_results_average:
            for item in one:
                sys.stdout.write('%.4f, ' % item)
            print('')
        avg_result_all_test_average = np.average(sequence_results_average, 0)
        for one in avg_result_all_test_average:
            sys.stdout.write('%.4f, ' % one)
        print('')
