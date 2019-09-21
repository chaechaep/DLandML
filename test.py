import pandas as pd

df = pd.read_csv('./train_0003.csv')

dict = {
    "filename" : str(df['filename'][0]),
    "regions" : {}
}
#
# for i in df.index:
#     print(df['ID'][i])
# dict_tmp = {str(df['ID'][1]) : '1'}
# dict.update()


for i in df.index:
    dict_tmp = {
        str(df['ID'][i]):{
            "shape_attributes":{
                "name": str(df['name'][i]),
                "all_points_x": df['all_points_x'][i],
                "all_points_y": df['all_points_y'][i]
            }
        }
    }
    dict["regions"].update(dict_tmp)

