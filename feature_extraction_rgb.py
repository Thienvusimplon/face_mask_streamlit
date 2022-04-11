import pandas as pd

from img_preprocessing import crop_img, resize_img, split_rgb, blurr_img, \
    rgb_hist, search_tresh, treshold, px_rate_rgb


def feature_extract(dataset, im):

    data = []
    i = 0
    limit = 200

    for pic_ref in dataset["name"]:
        df_loc_im = dataset[dataset["name"] == pic_ref]

        for index, df_pic_row in df_loc_im.iterrows():
            blurred_img_rgb = blurr_img(
                split_rgb(
                    resize_img(
                        im
                    )
                )
            )
            blurr_to_rate = px_rate_rgb(
                treshold(
                    search_tresh(
                        rgb_hist(blurred_img_rgb)), blurred_img_rgb
                )
            )
            data.append({"pic_ref": pic_ref,
                         "rate_r": blurr_to_rate[0],
                         "rate_g": blurr_to_rate[1],
                         "rate_b": blurr_to_rate[2],
                         #"classname": df_pic_row["classname"]
                         })

        i += 1
        if i == limit:
            break

    df_data = pd.DataFrame(data).reset_index()

    return df_data
