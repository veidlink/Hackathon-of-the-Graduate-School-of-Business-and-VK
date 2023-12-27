try:
    import pandas as pd
    import numpy as np
    from torchvision import transforms
    from PIL import Image
    import tqdm
    import torch
    import torchvision.models as models

    print("\nAssuming you provided correct test.csv file in the same folder as this script...\n")

    unf_resnet50 = models.resnet50(pretrained=False)
    unf_resnet50.fc = torch.nn.Linear(unf_resnet50.fc.in_features, 2)

    checkpoint_path_unf = 'UNF_BEST_weights_epoch_29_resnet50.pth'

    checkpoint_unf = torch.load(checkpoint_path_unf)
    unf_resnet50.load_state_dict(checkpoint_unf)

    unf_resnet50.eval()

    test_df = pd.read_csv('test.csv')

    predictions1 = pd.DataFrame(np.arange(0, test_df.shape[0]+1, 1), columns=['label'])
    predictions1

    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resnet standards
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\nWorking... Please wait...\n")

    # Load and preprocess the image
    for i in tqdm.tqdm(range(len(test_df))):
        img = Image.open(test_df.iloc[i, 0])
        img = transform(img)
        img = img.unsqueeze(0)

        # Predict the number of squares
        with torch.no_grad():
            output = unf_resnet50(img)
            number_of_squares = output[0, 0]
            number_of_squares = round(number_of_squares.item())
            predictions1.iloc[i, 0] = number_of_squares  # Assuming the output is a single value

    predictions1.to_csv('TEST_square_preds.csv') #CSV output for our model

    print("\nPredictions made! Check out the TEST_square_preds.csv file.\n")

except Exception as e:
    print(f"\nStopped... Error: {e}\n")