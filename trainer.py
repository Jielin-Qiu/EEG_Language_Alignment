def train_raw(train_loader, device, model, optimizer, total_num, args):
    all_labels = []
    all_res = []
    all_pred =[]
    model.train()
    total_loss = 0
    total_correct = 0    
    
    for batch in tqdm(train_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig, label, = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        if args.modality == 'text':
            pred = model(sig)
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred)
            all_pred.extend(pred.cpu().detach().numpy())
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

        if args.modality == 'eeg':
            pred = model(sig2)
            all_labels.extend(label.cpu().numpy())
            all_res.extend(pred.max(1)[1].cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred)
            all_pred.extend(pred.cpu().detach().numpy())
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cm, all_pred, all_labels



def train_fusion(train_loader, device, model, optimizer, total_num, args):
    model.train()
    all_labels = []
    all_res = []
    all_pred = []
    total_loss = 0
    total_correct = 0

    for batch in tqdm(train_loader, mininterval=100, desc='- (Training)  ', leave=False): 
        
        sig2, sig1, label1, = map(lambda x: x.to(device), batch)
      
      
        optimizer.zero_grad()
      
        if args.model == 'fusion':

            out, _, _ = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, out = out)
            
            
            loss.backward()
            optimizer.step_and_update_lr()
            total_loss += loss.item()
            total_correct += (n_correct1)
            
            cm = confusion_matrix(all_labels, all_res)

        if (args.model == 'CCA_fusion') or (args.model == 'WD_fusion'):

            out, pred, pred2 = model(sig1, sig2)
            all_labels.extend(label1.cpu().numpy())
            all_res.extend(out.max(1)[1].cpu().numpy())
            all_pred.extend(out.cpu().detach().numpy())
            loss, n_correct1 = cal_loss(label1, device, args, pred = pred, pred2 = pred2, out = out)
            
            
            loss.backward()
            optimizer.step_and_update_lr()
            total_loss += loss.item()
            total_correct += (n_correct1)
            
            cm = confusion_matrix(all_labels, all_res)
    

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num

    return train_loss, train_acc, cm, all_pred, all_labels


def train_alignment_ds(train_loader, device, model, optimizer, total_num, args):
    all_labels = []
    all_res = []
    all_pred_train = []
    model.train()
    total_loss = 0
    total_correct = 0    
    
    for batch in tqdm(train_loader, mininterval=100, desc='- (Training)  ', leave=False): 

        sig2, sig1, label, = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        pred, pred2 = model(sig1, sig2)  
        all_labels.extend(label.cpu().numpy())

        if args.modality == 'text':
            all_res.extend(pred.max(1)[1].cpu().numpy())
            all_pred_train.extend(pred.detach().cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred, pred2 = pred2)
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

        if args.modality == 'eeg':
            all_res.extend(pred2.max(1)[1].cpu().numpy())
            all_pred_train.extend(pred2.detach().cpu().numpy())
            loss, n_correct = cal_loss(label, device, args, pred = pred, pred2 = pred2)
            loss.backward()
            optimizer.step_and_update_lr()

            total_loss += loss.item()
            total_correct += n_correct
            cm = confusion_matrix(all_labels, all_res)

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    return train_loss, train_acc, cm, all_pred_train, all_labels