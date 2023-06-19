import torch
from loss import cal_loss
from metrics import cal_statistic
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np


def eval_raw(valid_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=100, desc='- (Validation)  ', leave=False):
            sig2, sig, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text':
                pred = model(sig)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct

            if args.modality == 'eeg':
                pred = model(sig2)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct

    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels


def eval_fusion(valid_loader1, device, model, total_num, args):
    model.eval()

    all_labels = []
    all_res = []
    all_pred = []
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
      
     
      for batch in tqdm(valid_loader1, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)

        if args.model == 'fusion':

            out, _, _ = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, out = out )
            total_loss += loss.item()
            total_correct += (n_correct1)

        if (args.model == 'CCA fusion') or (args.model == 'WD fusion'):

            out, pred, pred2 = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, pred=pred, pred2 = pred2, out = out )
            total_loss += loss.item()
            total_correct += (n_correct1)


    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num 
    valid_acc = total_correct /total_num 
    return valid_loss, valid_acc, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels

def eval_alignment_ds(valid_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=100, desc='- (Validation)  ', leave=False):
            sig2, sig1, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text':
                pred, pred2 = model(sig1, sig2)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.detach().cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct

            if args.modality == 'eeg':
                pred, pred2 = model(sig1, sig2)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred2.max(1)[1].cpu().numpy())
                all_pred.extend(pred2.detach().cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct

    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    return valid_loss, valid_acc, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels





def test_raw(test_loader, device, model, total_num):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

            sig2, sig, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text': 
                pred = model(sig)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct

            if args.modality == 'eeg': 
                pred = model(sig2)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred = pred)

                total_loss += loss.item()
                total_correct += n_correct


    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt',all_pred)
    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))

def test_fusion(test_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
     
      for batch in tqdm(test_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label1, = map(lambda x: x.to(device), batch)

        if args.model == 'fusion':
            out, _, _ = model(sig1, sig2)  
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().numpy())
            loss, n_correct1 = cal_loss(label1,device, args, out = out)
            total_loss += loss.item()
            total_correct += (n_correct1)
    
        if (args.model == 'CCA fusion') or (args.model == 'WD fusion'):
            out, pred, pred2 = model(sig1, sig2)  
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().numpy())
            loss, n_correct1 = cal_loss(label1,device, args, pred=pred, pred2=pred2, out = out)
            total_loss += loss.item()
            total_correct += (n_correct1)

    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt',all_pred)
    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num 
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))


def test_alignment_ds(test_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

            sig2, sig1, label, = map(lambda x: x.to(device), batch)

            if args.modality == 'text':

                pred, pred2 = model(sig1, sig2)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct
            
            if args.modality == 'eeg':
                pred, pred2 = model(sig1, sig2)  
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred2.max(1)[1].cpu().numpy())
                all_pred.extend(pred2.cpu().numpy())
                loss, n_correct = cal_loss(label, device, args, pred=pred, pred2=pred2)

                total_loss += loss.item()
                total_correct += n_correct


    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_pred.txt',all_pred)
    np.savetxt(f'baselines/{args.model}_{args.modality}/{args.level}_all_label.txt', all_labels)
    all_pred = np.array(all_pred)
    cm = confusion_matrix(all_labels, all_res)
    print("test_cm:", cm)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i=rec_i))
    print('F1_i is : {F1_i}'.format(F1_i=F1_i))
    test_acc = total_correct / total_num
    print('test_acc is : {test_acc}'.format(test_acc=test_acc))