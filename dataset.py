from config import*

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		imagePath = self.imagePaths[idx]
		maskPath = self.maskPaths[idx]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE) # với cv2.IMREAD_GRAYSCALE = 0
		if self.transforms:
			# grab the image path from the current index
			augmented = self.transforms(image=image, mask=mask)
			image = augmented["image"]
			mask = augmented["mask"]
			# print("shape_mask1: ", mask.shape)
			mask = (mask > 127).float() 
			# mask = (mask > 127).astype("float32")        # chuyển về float32: giá trị 0.0 hoặc 1.0
			# mask = torch.from_numpy(mask)  
			mask = mask.unsqueeze(0)                     # shape (1, H, W)
			return image, mask, imagePath	
			
def get_dataloaders(augment):    
    if augment:
        print("[INFO] Using AUGMENTATION for training set")
        train_transform = A.Compose([
	     A.Resize(
	        height=256,
	        width=256,
	        interpolation=cv2.INTER_LINEAR,          # cho ảnh
	        mask_interpolation=cv2.INTER_NEAREST     # cho mask
    ),
            A.HorizontalFlip(p=0.5),
            # A.Rotate(limit=15, border_mode=0, p=0.3),
	    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            # A.GaussNoise(var_limit=(10, 50), p=0.2),
            # A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.2),
            # A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.2),
            # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
	    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

            ToTensorV2()
        ])
    else:
        print("[INFO] Not using AUGMENTATION")
        train_transform = A.Compose([
            A.Resize(
	        height=256,
	        width=256,
	        interpolation=cv2.INTER_LINEAR,          # cho ảnh
	        mask_interpolation=cv2.INTER_NEAREST     # cho mask
    ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    valid_transform = A.Compose([
	A.Resize(
	        height=256,
	        width=256,
	        interpolation=cv2.INTER_LINEAR,          # cho ảnh
	        mask_interpolation=cv2.INTER_NEAREST     # cho mask
    ),
        # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
	     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


    g = torch.Generator()
    g.manual_seed(SEED)
    def seed_worker(worker_id):
	    np.random.seed(SEED + worker_id)
	    random.seed(SEED + worker_id)

    trainImagesPaths = sorted(list(paths.list_images(IMAGE_TRAIN_PATH)))
    trainMasksPaths = sorted(list(paths.list_images(MASK_TRAIN_PATH)))

    validImagesPaths = sorted(list(paths.list_images(IMAGE_VALID_PATH)))
    validMasksPaths = sorted(list(paths.list_images(MASK_VALID_PATH)))

    testImagesPaths = sorted(list(paths.list_images(IMAGE_TEST_PATH)))
    testMasksPaths = sorted(list(paths.list_images(MASK_TEST_PATH)))

    trainDS = SegmentationDataset(trainImagesPaths, trainMasksPaths, transforms=train_transform)
    validDS = SegmentationDataset(validImagesPaths, validMasksPaths, transforms=valid_transform)
    testDS = SegmentationDataset(testImagesPaths, testMasksPaths, transforms=valid_transform)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(validDS)} examples in the valid set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
	
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=bach_size, pin_memory=PIN_MEMORY,
        num_workers=4, worker_init_fn=seed_worker,generator=g)
    validLoader = DataLoader(validDS, shuffle=False,
        batch_size=bach_size, pin_memory=PIN_MEMORY,
        num_workers=4, worker_init_fn=seed_worker, generator=g)
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=bach_size, pin_memory=PIN_MEMORY,
        num_workers=4, worker_init_fn=seed_worker, generator=g)
	
    return trainLoader, validLoader, testLoader


# if augment:
# 	print("[INFO] Using AUGMENTATION for training set")
# 	train_transform = A.Compose([
# 	     A.Resize(
# 		height=256,
# 		width=256,
# 		interpolation=cv2.INTER_LINEAR,          # cho ảnh
# 		mask_interpolation=cv2.INTER_NEAREST     # cho mask
# 	),
# 	    A.HorizontalFlip(p=0.5),
# 	    A.Rotate(limit=15, border_mode=0, p=0.3),
# 	    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
# 	    A.GaussNoise(var_limit=(10, 50), p=0.2),
# 	    A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.2),
# 	    A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.2),
# 	    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
# 	    ToTensorV2()
# 	])
# else:
# 	print("[INFO] Not using AUGMENTATION")
#         train_transform = A.Compose([
#             A.Resize(
# 	        height=256,
# 	        width=256,
# 	        interpolation=cv2.INTER_LINEAR,          # cho ảnh
# 	        mask_interpolation=cv2.INTER_NEAREST     # cho mask
# 	),
#             A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
#             ToTensorV2()
#         ])

# valid_transform = A.Compose([
# 	A.Resize(
# 		height=256,
# 		width=256,
# 		interpolation=cv2.INTER_LINEAR,          # cho ảnh
# 		mask_interpolation=cv2.INTER_NEAREST     # cho mask
# 	),
# 	A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
# 	ToTensorV2()
# ])



# trainImagesPaths = sorted(list(paths.list_images(IMAGE_TRAIN_PATH)))
# trainMasksPaths = sorted(list(paths.list_images(MASK_TRAIN_PATH)))

# validImagesPaths = sorted(list(paths.list_images(IMAGE_VALID_PATH)))
# validMasksPaths = sorted(list(paths.list_images(MASK_VALID_PATH)))

# testImagesPaths = sorted(list(paths.list_images(IMAGE_TEST_PATH)))
# testMasksPaths = sorted(list(paths.list_images(MASK_TEST_PATH)))

# trainDS = SegmentationDataset(trainImagesPaths, trainMasksPaths, transforms=train_transform)
# validDS = SegmentationDataset(validImagesPaths, validMasksPaths, transforms=valid_transform)
# testDS = SegmentationDataset(testImagesPaths, testMasksPaths, transforms=valid_transform)
# print(f"[INFO] found {len(trainDS)} examples in the training set...")
# print(f"[INFO] found {len(validDS)} examples in the valid set...")
# print(f"[INFO] found {len(testDS)} examples in the test set...")

# trainLoader = DataLoader(trainDS, shuffle=True,
# 	batch_size=bach_size, pin_memory=PIN_MEMORY,
# 	num_workers=4)
# validLoader = DataLoader(validDS, shuffle=False,
# 	batch_size=bach_size, pin_memory=PIN_MEMORY,
# 	num_workers=4)
# testLoader = DataLoader(testDS, shuffle=False,
# 	batch_size=bach_size, pin_memory=PIN_MEMORY,
# 	num_workers=4)


