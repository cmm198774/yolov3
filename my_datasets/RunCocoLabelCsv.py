import pandas as pd
import json 
raw_path='.//raw_data//coco2014//annotations//'
train_label_file='instances_train2014.json'
val_label_file='instances_val2014.json'
train_tgt_csv_file='train_info_2014.csv'
val_tgt_csv_file='val_info_2014.csv'

def main():
    dt=json.load(open(raw_path+train_label_file))
    df_img_info=pd.DataFrame(dt['images'])
    df_annotations_info=pd.DataFrame(dt['annotations'])
    df_categories_info=pd.DataFrame(dt['categories'])
    df_train_info=pd.merge(df_img_info[['id','file_name']].rename(columns={'id':'image_id'}),df_annotations_info[['image_id','bbox','category_id','id']].rename(columns={'id':'box_id'}),how='left',on='image_id')
    df_train_info=pd.merge(df_train_info,df_categories_info.rename(columns={'id':'category_id'}),how='left',on='category_id')
    df_train_info=df_train_info[df_train_info['category_id'].notnull()].drop_duplicates(subset=['image_id','box_id','category_id'],keep='first')
    dt=json.load(open(raw_path+val_label_file))
    df_img_info=pd.DataFrame(dt['images'])
    df_annotations_info=pd.DataFrame(dt['annotations'])
    df_categories_info=pd.DataFrame(dt['categories'])
    df_val_info=pd.merge(df_img_info[['id','file_name']].rename(columns={'id':'image_id'}),df_annotations_info[['image_id','bbox','category_id','id']].rename(columns={'id':'box_id'}),how='left',on='image_id')
    df_val_info=pd.merge(df_val_info,df_categories_info.rename(columns={'id':'category_id'}),how='left',on='category_id')
    df_val_info=df_val_info[df_val_info['category_id'].notnull()].drop_duplicates(subset=['image_id','box_id','category_id'],keep='first')
    df_category=df_train_info[['category_id']].drop_duplicates().sort_values(by='category_id',ascending=True)
    df_category['category_id_modif']=np.arange(len(df_category))
    df_train_info=pd.merge(df_train_info,df_category,how='left',on='category_id')
    df_val_info=pd.merge(df_val_info,df_category,how='left',on='category_id')
    df_train_info.to_csv(raw_path+train_tgt_csv_file,header=True,index=False)
    df_val_info.to_csv(raw_path+val_tgt_csv_file,header=True,index=False)

if __name__=='__main__':
    main()