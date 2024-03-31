from dataprocess import data_sampler_bert_prompt_deal_first_task_sckd, get_data_loader_bert_prompt
import torch
import argparse
import json
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task" , default="tacred" , type=str)
    parser.add_argument("--shot" , default=5 , type=int)
    args = parser.parse_args()

    if args.task == "tacred":
        f = open("config/config_tacred.json", "r")
    elif args.task == "fewrel":
        f = open("config/config_fewrel_5and10.json", "r")
    else:
        raise ValueError("task must be tacred or fewrel")

    config = json.loads(f.read())
    f.close()

    if args.task == "fewrel":
        config['relation_file'] = "data/fewrel/relation_name.txt"
        config['rel_index'] = "data/fewrel/rel_index.npy"
        config['rel_feature'] = "data/fewrel/rel_feature.npy"
        config['rel_des_file'] = "data/fewrel/relation_description.txt"
        config['num_of_relation'] = 80
        if args.shot == 5:
            print('fewrel 5 shot')
            config['rel_cluster_label'] = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config['training_file'] = "data/fewrel/CFRLdata_10_100_10_5/train_0.txt"
            config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_5/valid_0.txt"
            config['test_file'] = "data/fewrel/CFRLdata_10_100_10_5/test_0.txt"
        elif args.shot == 10:
            config['rel_cluster_label'] = "data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config['training_file'] = "data/fewrel/CFRLdata_10_100_10_10/train_0.txt"
            config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_10/valid_0.txt"
            config['test_file'] = "data/fewrel/CFRLdata_10_100_10_10/test_0.txt"
        else:
            print('fewrel 2 shot')
            config['rel_cluster_label'] = "data/fewrel/CFRLdata_10_100_10_2/rel_cluster_label_0.npy"
            config['training_file'] = "data/fewrel/CFRLdata_10_100_10_2/train_0.txt"
            config['valid_file'] = "data/fewrel/CFRLdata_10_100_10_2/valid_0.txt"
            config['test_file'] = "data/fewrel/CFRLdata_10_100_10_2/test_0.txt"
    else:
        config['relation_file'] = "data/tacred/relation_name.txt"
        config['rel_index'] = "data/tacred/rel_index.npy"
        config['rel_feature'] = "data/tacred/rel_feature.npy"
        config['num_of_relation'] = 41
        if args.shot == 5:
            config['rel_cluster_label'] = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config['training_file'] = "data/tacred/CFRLdata_10_100_10_5/train_0.txt"
            config['valid_file'] = "data/tacred/CFRLdata_10_100_10_5/valid_0.txt"
            config['test_file'] = "data/tacred/CFRLdata_10_100_10_5/test_0.txt"
        else:
            config['rel_cluster_label'] = "data/tacred/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config['training_file'] = "data/tacred/CFRLdata_10_100_10_10/train_0.txt"
            config['valid_file'] = "data/tacred/CFRLdata_10_100_10_10/valid_0.txt"
            config['test_file'] = "data/tacred/CFRLdata_10_100_10_10/test_0.txt"

        
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    config['neg_sampling'] = False

    config['first_task_k-way'] = 10
    config['k-shot'] = 5
    donum = 1
    epochs = 1
    threshold=0.1

    if config["prompt"] == "hard-complex":
        template = 'the relation between e1 and e2 is mask . '
        print('Template: %s'%template)
    elif config["prompt"] == "hard-simple":
        template = 'e1 mask e2 . '
        print('Template: %s'%template)
    else:
        template = None
        print("no use soft prompt.")
    sampler = data_sampler_bert_prompt_deal_first_task_sckd(config, 'sss', template)
    for training_data, valid_data, test_data,test_all_data, seen_relations,current_relations in sampler:
        data_loader = get_data_loader_bert_prompt(config, training_data, batch_size=1)
        print('\nTrain set-------------------------------------------')
        for x in tqdm(data_loader):
            pass
        valid_loader = get_data_loader_bert_prompt(config, valid_data, batch_size=1)
        print('\nValid set-------------------------------------------')
        for x in tqdm(valid_loader):
            pass
        test_loader = get_data_loader_bert_prompt(config, test_data, batch_size=1)
        print('\nTest set-------------------------------------------')
        for x in tqdm(test_loader):
            pass
        test_all_loader = get_data_loader_bert_prompt(config, test_all_data, batch_size=1)
        print('\nTest all set-------------------------------------------')
        for x in tqdm(test_all_loader):
            pass