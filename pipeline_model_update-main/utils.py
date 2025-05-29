from config import*
from train import get_args

# def dice_coeff(pred, target, smooth=1e-5):
#     intersection = torch.sum(pred * target)
#     return (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)

# def dice_coef_loss(pred, target, smooth=1e-6):
#     """
#     Dice Loss: Thước đo sự chồng lấn giữa output và ground truth.
#     """
#     pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
#     intersection = (pred * target).sum()
#     union = pred.sum() + target.sum()
#     dice_score = (2.0 * intersection + smooth) / (union + smooth)
#     return 1 - dice_score  # Dice loss

# def iou(y_pred, y_true, eps=1e-7):
#     y_true_f = y_true.view(-1)  # flatten
#     y_pred_f = y_pred.view(-1)  # flatten

#     intersection = torch.sum(y_true_f * y_pred_f)
#     union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection

#     return intersection / (union + eps)  # thêm eps để tránh chia 0

# def iou_core(y_pred, y_true, eps=1e-7):
#     y_true_f = y_true.view(-1)  # flatten
#     y_pred_f = y_pred.view(-1)  # flatten

#     intersection = torch.sum(y_true_f * y_pred_f)
#     union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection

#     return intersection / (union + eps)  # thêm eps để tránh chia 0
    
def dice_coeff(pred, target, epsilon=1e-6):
    y_pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
    numerator = 2 * torch.sum(target * y_pred, dim=(1, 2, 3))
    denominator = torch.sum(target + y_pred, dim=(1, 2, 3))
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice
    
def dice_coef_loss(dice):
    loss = 1 - dice
    loss_mean = torch.mean(loss)
    return  loss_mean # Dice loss

def bce_loss(pred, target):
    pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
    bce = nn.BCELoss()
    bce_score = bce(pred, target)
    return bce_score 
    
def bce_dice_loss(pred, target):
    dice = dice_coeff(pred, target)
    dice_score = dice_coef_loss(dice)
    bce_score = bce_loss(pred, target)
    return bce_score + dice_score
    
def iou_core(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)  # Chuyển logits về xác suất
    # Tính intersection và union theo từng ảnh
    intersection = torch.sum(pred * target, dim=(1, 2, 3))  # Batch_size x 1
    union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou  # Giữ nguyên theo batch, mỗi ảnh 1 giá trị

def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().item()  # Chuyển tensor về CPU và lấy giá trị float
    elif isinstance(value, list):
        return [tensor_to_float(v) for v in value]  # Xử lý danh sách các tensor
    return value  # Nếu không phải tensor, giữ nguyên
def to_numpy(tensor):
    # Move tensor to CPU and convert to NumPy array
    return tensor.cpu().detach().item()
    
def loss_func(*kwargs):
    args = get_args()
    if args.loss == "Dice_loss":
        x = dice_coef_loss(*kwargs)
        return x
    elif args.loss == "BCEDice_loss":
        x = bce_dice_loss(*kwargs)
        return x



    
