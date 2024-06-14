from dataset import BengaliDatasetTest
import os
import torch
import pandas as pd
import numpy as np
from model_dispatcher import MODEL_DISPATCHER

TEST_BATCH_SIZE = 32
MODEL_MEAN=(0.485,0.465,0.406)
MODEL_STD=(0.229,0.224,0.225)
IMG_HEIGHT=137
IMG_WIDTH=236
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "resnet34"

def model_predict():
    g_pred, v_pred, c_pred = [], [], []
    img_ids_list = [] 
    
    for file_idx in range(4):
        df = pd.read_parquet(f"../input/test_image_data_{file_idx}.parquet")

        dataset = BengaliDatasetTest(df=df,
                                    img_height=IMG_HEIGHT,
                                    img_width=IMG_WIDTH,
                                    mean=MODEL_MEAN,
                                    std=MODEL_STD)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size= TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        for bi, d in enumerate(data_loader):
            image = d["image"]
            img_id = d["image_id"]
            image = image.to(DEVICE, dtype=torch.float)

            g, v, c = model(image)
            #g = np.argmax(g.cpu().detach().numpy(), axis=1)
            #v = np.argmax(v.cpu().detach().numpy(), axis=1)
            #c = np.argmax(c.cpu().detach().numpy(), axis=1)

            for ii, imid in enumerate(img_id):
                g_pred.append(g[ii].cpu().detach().numpy())
                v_pred.append(v[ii].cpu().detach().numpy())
                c_pred.append(c[ii].cpu().detach().numpy())
                img_ids_list.append(imid)
        
    return g_pred, v_pred, c_pred, img_ids_list

if __name__ == "__main__":
    final_g_pred = []
    final_v_pred = []
    final_c_pred = []
    final_img_ids = []

    for i in range(5):
        model = model = MODEL_DISPATCHER[BASE_MODEL](pretrained=False)
        model.load_state_dict(torch.load(f"../models/resnet34_fold{i}_best.pth"))
        model.to(DEVICE)
        model.eval()
        g_pred, v_pred, c_pred, img_ids_list = model_predict()
        final_g_pred.append(g_pred)
        final_v_pred.append(v_pred)
        final_c_pred.append(c_pred)
        if i == 0:
            final_img_ids.extend(img_ids_list)

    final_g = np.argmax(np.mean(np.array(final_g_pred), axis=0), axis=1)
    final_v = np.argmax(np.mean(np.array(final_v_pred), axis=0), axis=1)
    final_c = np.argmax(np.mean(np.array(final_c_pred), axis=0), axis=1)

    predictions = []
    for ii, imid in enumerate(final_img_ids):

        predictions.append((f"{imid}_grapheme_root", final_g[ii]))
        predictions.append((f"{imid}_vowel_diacritic", final_v[ii]))
        predictions.append((f"{imid}_consonant_diacritic", final_c[ii]))


    sub = pd.DataFrame(predictions,columns=["row_id","target"])
    sub.to_csv("../submission.csv",index=False)
