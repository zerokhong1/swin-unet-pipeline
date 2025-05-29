from config import*
from train import get_args
# import model
# def optimizer():
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     return optimizer

def optimizer(model):
    args = get_args()
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)
        # optimizer = Adam(
        #     model.parameters(),
        #     lr = INIT_LR,
        #     betas = BETA,
        #     weight_decay = WEIGHT_DECAY,
        #     amsgrad = AMSGRAD  # nếu bạn muốn bật AMSGrad giống như bài báo có đề cập
        # )
        return optimizer
    elif args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=lr0, weight_decay=weight_decay) 
        return optimizer
