import argparse
# from dataset import*
# import model.Unet
# import model.Unet
def get_args():
    # Tham số bắt buộc nhập
    parser = argparse.ArgumentParser(description="Train, Pretrain hoặc Evaluate một model AI")
    parser.add_argument("--epoch", type=int, help="Số epoch để train")
    # parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến model")
    parser.add_argument("--mode", type=str, choices=["train", "pretrain", "evaluate"], required=True, help="Chế độ: train hoặc pretrain hoặc evaluate")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn đến dataset đã giải nén")
    # Tham số trường hợp
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn đến file checkpoint (chỉ dùng cho chế độ pretrain và evaluate)")
    parser.add_argument("--augment", action='store_true', help="Bật Augmentation cho dữ liệu đầu vào")
    # Tham số mặc định(default)
    parser.add_argument("--saveas", type=str, help="Thư mục lưu checkpoint")
    parser.add_argument("--lr0", type=float, help="learning rate, default = 0.0001")
    parser.add_argument("--batchsize", type=int, help="Batch size, default = 8")

    parser.add_argument("--weight_decay", type=float,  help="weight_decay, default = 1e-6")
    parser.add_argument("--img_size", type=int, nargs=2,  help="Height and width of the image, default = [256, 256]")
    parser.add_argument("--numclass", type=int, help="shape of class, default = 1")
    
    """
    # Với img_size, cách chạy: python script.py --img_size 256 256
    Nếu muốn nhập list dài hơn 3 phần tử, gõ 
    parser.add_argument("--img_size", type=int, nargs='+', default=[256, 256], help="Image dimensions")
    Chạy:
    python script.py --img_size 128 128 3
    """
    parser.add_argument("--loss", type=str, choices=["Dice_loss", "BCEDice_loss"], default="Dice_loss", help="Hàm loss sử dụng, default = Dice_loss")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "SGD"], default="Adam", help="Optimizer sử dụng, default = Adam")

    args = parser.parse_args()
    if args.mode in ["train", "pretrain"] and args.epoch is None:
        parser.error("--epoch là bắt buộc khi mode là 'train' hoặc 'pretrain'")
    return args

def main():  
    import numpy as np
    import torch
    import random
    from trainer import Trainer
    from model import Unet, Swin_unet
    import optimizer
    from result import export, export_evaluate
    from dataset import get_dataloaders
    global trainer
    SEED=42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model1 = Unet.Unet(input_channel = 3)
    model1 = Swin_unet.SwinUnet() 
    # Swin_unet.load_pretrained_encoder(model1)
    Swin_unet.load_pretrained_encoder(model1, "swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth")
    optimizer1 = optimizer.optimizer(model = model1)
    trainer = Trainer(model = model1, optimizer = optimizer1)
    trainLoader, validLoader, testLoader = get_dataloaders(args.augment)
    if args.mode == "train":
        trainer.train(trainLoader, validLoader, testLoader)
        export(trainer)
    elif args.mode == "pretrain":
        if not args.checkpoint:
            raise ValueError("Chế độ pretrain yêu cầu checkpoint!")
        trainer.load_checkpoint(args.checkpoint)
        if args.epoch <= trainer.checkpoint['epoch']:
            raise ValueError(
            f"Epoch bạn nhập ({args.epoch}) phải lớn hơn số epoch hiện tại trong checkpoint ({trainer.start_epoch})."
        )
        trainer.pretrained(train_loader=trainLoader, val_loader=validLoader, test_loader = testLoader, checkpoint_path = args.checkpoint)
        # trainer.pretrained(trainLoader,validLoader,args.checkpoint)
        export(trainer)
    else:
        if not args.checkpoint:
            raise ValueError("Chế độ evaluate yêu cầu checkpoint!")
        trainer.evaluate(train_loader=trainLoader, val_loader=validLoader, test_loader = testLoader, checkpoint_path = args.checkpoint)
        # trainer.pretrained(trainLoader,validLoader,args.checkpoint)
        export_evaluate(trainer)

if __name__ == "__main__":
    args = get_args()
    main()
