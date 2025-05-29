from config import*
from utils import*
from optimizer import*
class Trainer:
    def __init__(self, model, optimizer, criterion = loss_func, patience = 50, device = DEVICE):
        # self.model = model.to(device)
        self.model = model.to(DEVICE)
        self.num_epochs = NUM_EPOCHS
        # self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.optimizer = optimizer
        self.early_stop_counter = 0
        self.train_losses, self.val_losses = [], []
        self.train_dices, self.val_dices = [], []
        self.train_ious, self.val_ious = [], []
        self.best_model, self.best_dice, self.best_epoch_dice = None, 0.0, 0
        self.best_iou, self.best_epoch_iou = 0.0, 0
        self.dice_list = []
        self.iou_list = []
        self.path_list = []
        self.log_interval = 1  # Số bước để log
         # Khởi tạo CosineAnnealingLR scheduler
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=lr_min)
        # self.scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)
        self.scheduler = ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=5,min_lr=1e-6)  # <- giới hạn nhỏ nhất của learning rate
        
    def save_checkpoint(self, epoch, dice, iou, filename, mode = "pretrained"):
        if mode == "train":
            self.start_epoch = 0
        checkpoint = {
            'start_epoch' : self.start_epoch,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
            'best_dice': dice,
            'best_iou': iou,
            'best_epoch_dice': self.best_epoch_dice,
            'best_epoch_iou': self.best_epoch_iou,
        }
        torch.save(checkpoint, filename)
        print(f"[INFO] Checkpoint saved: {filename}")
    def load_checkpoint(self, path):
        self.checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.start_epoch=self.checkpoint['epoch']
        self.train_losses, self.val_losses = self.checkpoint['train_losses'], self.checkpoint['val_losses']
        self.train_dices, self.val_dices = self.checkpoint['train_dices'], self.checkpoint['val_dices']
        self.train_ious, self.val_ious = self.checkpoint['train_ious'], self.checkpoint['val_ious']
        self.best_dice, self.best_epoch_dice = self.checkpoint['best_dice'], self.checkpoint['best_epoch_dice']
        self.best_iou, self.best_epoch_iou = self.checkpoint['best_iou'], self.checkpoint['best_epoch_iou']

    def train(self, train_loader, val_loader, test_loader):
        print("lr0", lr0)
        print("bach_size", bach_size)
        print("weight_decay", weight_decay)
        print("input_image_width", input_image_width)
        print("input_image_height", input_image_height)
        print("numclass", numclass)
        print("NUM_EPOCHS", NUM_EPOCHS)
        print("augment", augment)
        # print(f"[INFO] Training completed!")
        start_time = time.time()
        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
        # for epoch in range(self.num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_dice = 0.0
            val_dice = 0.0
            train_iou = 0.0
            val_iou = 0.0

            # Training loop with progress bar
            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            train_loader_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
            for i, (images, masks, _) in train_loader_progress:
                images, masks = images.to(self.device), masks.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(images)
                dice = dice_coeff(outputs, masks)
                loss = self.criterion(dice)
                dice = torch.mean(dice)
                iou = iou_core(outputs, masks)
                iou = torch.mean(iou)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_dice += dice.item()
                train_iou += iou.item()

                # Log every 15 steps
                if (i + 1) % self.log_interval == 0:
                    train_loader_progress.set_postfix({'Step': i + 1, 'Loss': loss.item(), 'Dice': dice.item(), 'Iou': iou.item()})
            # self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_loader_progress = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
                for i, (images, masks, _) in val_loader_progress:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    dice = dice_coeff(outputs, masks)
                    loss = self.criterion(dice)
                    dice = torch.mean(dice)
                    iou = iou_core(outputs, masks)
                    iou = torch.mean(iou)
                    
                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_iou += iou.item()
                    if (i + 1) % self.log_interval == 0:
                        val_loader_progress.set_postfix({'Step': i + 1, 'Loss': loss.item(), 'Dice': dice.item(), 'Iou': iou.item()})


            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_dice = train_dice / len(train_loader)
            self.avg_val_dice = val_dice / len(val_loader)
            avg_train_iou = train_iou / len(train_loader)
            avg_val_iou = val_iou / len(val_loader)
            self.scheduler.step(avg_val_loss) #=> scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            print(f"Epoch {epoch+1}: LR {self.scheduler.get_last_lr()[0]}, Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Train Dice {avg_train_dice:.4f}, Val Dice {self.avg_val_dice:.4f}, Train Iou {avg_train_iou:.4f}, Val Iou {avg_val_iou:.4f}")
            # print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Train Dice {avg_train_dice:.4f}, Val Dice {self.avg_val_dice:.4f}, Train Iou {avg_train_iou:.4f}, Val Iou {avg_val_iou:.4f}")           
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_dices.append(avg_train_dice)
            self.val_dices.append(self.avg_val_dice)
            self.train_ious.append(avg_train_iou)
            self.val_ious.append(avg_val_iou)    
             

            self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, f'last_model.pth', mode="train")
            if avg_val_iou > self.best_iou:
                self.best_iou, self.best_epoch_iou = avg_val_iou, epoch + 1
            if self.avg_val_dice > self.best_dice:
                self.best_dice, self.best_epoch_dice = self.avg_val_dice, epoch + 1
                self.save_checkpoint(epoch +1, self.best_dice, self.best_iou, f'best_model.pth', mode="train")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, f'last_model.pth', mode="train")
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break
            torch.cuda.empty_cache()
            gc.collect()

        print(f"[INFO] Training completed in {time.time() - start_time:.2f}s")

    def pretrained(self, train_loader, val_loader, test_loader, checkpoint_path):
        print("lr0",lr0)
        print("bach_size",bach_size)
        print("weight_decay",weight_decay)
        print("input_image_width",input_image_width)
        print("input_image_height",input_image_height)
        print("numclass",numclass)
        print("NUM_EPOCHS",NUM_EPOCHS)
        print("augment", augment)
        print("Đường dẫn dẫn đến file checkpoint", checkpoint_path)
        # Load model from checkpoint
        self.load_checkpoint(checkpoint_path)
        # Continue training from the checkpoint
        print(f"[INFO] Continuing training from epoch {self.start_epoch + 1}")
        start_time = time.time()
        # Tạo lại vòng lặp huấn luyện
        for epoch in tqdm(range(self.start_epoch, self.num_epochs), desc="Training Progress"):
            train_loss = 0.0
            val_loss = 0.0
            train_dice = 0.0
            val_dice = 0.0
            train_iou = 0.0
            val_iou = 0.0

            # Training loop with progress bar
            print(f'\nEpoch {epoch + 1}/{self.num_epochs}')
            train_loader_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
            for i, (images, masks, _) in train_loader_progress:
                images, masks = images.to(self.device), masks.to(self.device)

                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(images)
                dice = dice_coeff(outputs, masks)
                loss = self.criterion(dice)
                dice = torch.mean(dice)
                iou = iou_core(outputs, masks)
                iou = torch.mean(iou)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_dice += dice.item()
                train_iou += iou.item()

                # Log every 15 steps
                if (i + 1) % self.log_interval == 0:
                    train_loader_progress.set_postfix({'Step': i + 1, 'Loss': loss.item(), 'Dice': dice.item(), 'Iou': iou.item()})
            # self.scheduler.step(epoch)
            self.model.eval()
            with torch.no_grad():
                val_loader_progress = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
                for i, (images, masks, _) in val_loader_progress:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    dice = dice_coeff(outputs, masks)
                    loss = self.criterion(dice)
                    dice = torch.mean(dice)
                    iou = iou_core(outputs, masks)
                    iou = torch.mean(iou)
                    
                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_iou += iou.item()
                    if (i + 1) % self.log_interval == 0:
                      val_loader_progress.set_postfix({'Step': i + 1, 'Loss': loss.item(), 'Dice': dice.item(), 'Iou': iou.item()})

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_dice = train_dice / len(train_loader)
            self.avg_val_dice = val_dice / len(val_loader)
            avg_train_iou = train_iou / len(train_loader)
            avg_val_iou = val_iou / len(val_loader)
            self.scheduler.step(avg_val_loss) # => scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            print(f"Epoch {epoch+1}: LR {self.scheduler.get_last_lr()[0]}, Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Train Dice {avg_train_dice:.4f}, Val Dice {self.avg_val_dice:.4f}, Train Iou {avg_train_iou:.4f}, Val Iou {avg_val_iou:.4f}")
            # print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Train Dice {avg_train_dice:.4f}, Val Dice {self.avg_val_dice:.4f}, Train Iou {avg_train_iou:.4f}, Val Iou {avg_val_iou:.4f}")
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_dices.append(avg_train_dice)
            self.val_dices.append(self.avg_val_dice)
            self.train_ious.append(avg_train_iou)
            self.val_ious.append(avg_val_iou)

            self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, f'last_model.pth')
            if avg_val_iou > self.best_iou:
                self.best_iou, self.best_epoch_iou = avg_val_iou, epoch + 1
                
            if self.avg_val_dice > self.best_dice:
                self.best_dice, self.best_epoch_dice = self.avg_val_dice, epoch + 1
                self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, f'best_model.pth')
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                self.save_checkpoint(epoch + 1, self.best_dice, self.best_iou, f'last_model.pth')
                print(f"[INFO] Early stopping at epoch {epoch + 1}")
                break

            torch.cuda.empty_cache()
            gc.collect()

        print(f"[INFO] Training completed in {time.time() - start_time:.2f}s")
    def evaluate(self, train_loader, val_loader, test_loader, checkpoint_path):
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        test_dice_total = 0.0
        test_iou_total = 0.0
        # dice_list = []
        # iou_list = []

        with torch.no_grad():
            test_loader_progress = tqdm(enumerate(val_loader), total=len(val_loader), desc="Testing")
            for i, (images, masks, image_paths) in test_loader_progress:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                # Lặp từng ảnh trong batch để tính riêng biệt
                for j in range(images.size(0)):
                    output = outputs[j].unsqueeze(0)
                    mask = masks[j].unsqueeze(0)
                    path = image_paths[j]

                    dice = dice_coeff(output, mask)
                    iou = iou_core(output, mask)

                    self.dice_list.append(dice.item())
                    self.iou_list.append(iou.item())
                    self.path_list.append(path)

                    test_loader_progress.set_postfix({'Image': i + 1, 'Dice': dice.item(), 'IoU': iou.item()})

                    test_dice_total += dice.item()
                    test_iou_total += iou.item()

        num_samples = len(self.dice_list)
        avg_dice = test_dice_total / num_samples
        avg_iou = test_iou_total / num_samples
        print(f"[VALID] Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}")
        return avg_dice, avg_iou, self.dice_list, self.iou_list, self.path_list


    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'train_ious': self.train_ious,
            'val_ious' : self.val_ious,
            'best_dice': self.best_dice,
            'best_iou': self.best_iou,
            'best_epoch_dice': self.best_epoch_dice,
            'best_epoch_iou': self.best_epoch_iou            
        }

    
