import torch
from loss import cal_loss
from metrics import cal_statistic
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from config import num_heads, num_layers

def eval(valid_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader, mininterval=100, desc='- (Validation)  ', leave=False):
            
            if args.modality == 'text':
                text, label = batch['sentence'].to(device), batch['label'].to(device)
            elif args.modality == 'eeg':
                eeg, label = batch['seq'].to(device), batch['label'].to(device)
            else:
                text, eeg, label = batch['sentence'].to(device), batch['seq'].to(device), batch['label'].to(device)
            
            if args.modality == 'text':
                pred_text = model(text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_text.max(1)[1].cpu().numpy())
                all_pred.extend(pred_text.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_text)

                total_loss += loss.item()
                total_correct += n_correct

            elif args.modality == 'eeg':
                pred_eeg = model(eeg_src_seq = eeg)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_eeg.max(1)[1].cpu().numpy())
                all_pred.extend(pred_eeg.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_eeg)

                total_loss += loss.item()
                total_correct += n_correct
            else:
                pred = model(eeg_src_seq = eeg, text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred)
                all_pred.extend(pred.cpu().detach().numpy())

                total_loss += loss.item()
                total_correct += n_correct
                
    cm = confusion_matrix(all_labels, all_res)
    acc_SP, pre_i, rec_i, F1_i = cal_statistic(cm)
    print('acc_SP is : {acc_SP}'.format(acc_SP=acc_SP))
    print('pre_i is : {pre_i}'.format(pre_i=calculate_average(pre_i)))
    print('rec_i is : {rec_i}'.format(rec_i=calculate_average(rec_i)))
    print('F1_i is : {F1_i}'.format(F1_i=calculate_average(F1_i)))
    valid_loss = total_loss / total_num
    valid_acc = total_correct / total_num
    print(f'Validation Loss: {valid_loss}')
    print(f'Validation Accuracy: {valid_acc}')
    return valid_loss, valid_acc, cm, sum(rec_i[1:]) * 0.6 + sum(pre_i[1:]) * 0.4, all_pred, all_labels


def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Return 0 if the list is empty to avoid division by zero error
    
    total = sum(numbers)
    average = total / len(numbers)
    return average

def inference(test_loader, device, model, total_num, args):
    all_labels = []
    all_res = []
    all_pred = []
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=0.5, desc='- (Validation)  ', leave=False):

            if args.modality == 'text':
                text, label = batch['sentence'].to(device), batch['label'].to(device)
            elif args.modality == 'eeg':
                eeg, label = batch['seq'].to(device), batch['label'].to(device)
            else:
                text, eeg, label = batch['sentence'].to(device), batch['seq'].to(device), batch['label'].to(device)
            
            if args.modality == 'text':
                pred_text = model(text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_text.max(1)[1].cpu().numpy())
                all_pred.extend(pred_text.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_text)

                total_loss += loss.item()
                total_correct += n_correct

            elif args.modality == 'eeg':
                pred_eeg = model(eeg_src_seq = eeg)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred_eeg.max(1)[1].cpu().numpy())
                all_pred.extend(pred_eeg.cpu().detach().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred_eeg)

                total_loss += loss.item()
                total_correct += n_correct
            else:
                pred, eeg_embed, text_embed = model(eeg_src_seq = eeg, text_src_seq = text)
                all_labels.extend(label.cpu().numpy())
                all_res.extend(pred.max(1)[1].cpu().numpy())
                loss, n_correct = cal_loss(label, args, pred = pred, eeg_embed = eeg_embed, text_embed = text_embed)
                all_pred.extend(pred.cpu().detach().numpy())

                total_loss += loss.item()
                total_correct += n_correct


    np.savetxt(f'pred_labels/{args.model}_{args.modality}_{args.level}_{num_layers}_{num_heads}_{args.batch_size}_all_pred.txt',all_pred)
    np.savetxt(f'pred_labels/{args.model}_{args.modality}_{args.level}_{num_layers}_{num_heads}_{args.batch_size}_all_label.txt', all_labels)
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
