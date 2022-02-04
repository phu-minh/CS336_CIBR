import subprocess
import redis_server
subprocess.Popen([redis_server.REDIS_SERVER_PATH])
bovw_path = '/output/bovw.hdf5'
os.system('!python build_redis_index.py --bovw_db '+ bovw_path)

from ntpath import join
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import os 

bovw_path = 'output/bovw.hdf5'
features_path = 'output/features.hdf5'
dataset_path = 'paris'
vocab_path = 'output/vocab.cpickle'
idf_path = 'output/idf.cpickle'

st.set_page_config(
     page_title="CIBR Demo by Streamlit",
)
st.set_option('deprecation.showfileUploaderEncoding', False)


st.title('Streamlit Image Query')
# Upload an image and set some options for demo purposes
st.header("Content Based Image Retrieval Demo")

st.subheader("Click on sidebar, upload a file and query")

st.info('<====== Sidebar is on the left')

img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((500,500))
    st.image(cropped_img)

    if st.button('Crop and Query'):
        cropped_img.save("test.png")

        chosen2_path = 'test.png'
        st.write('Starting query...')   

        import time
        start_time = time.time()
        os.system('python search.py --dataset '+dataset_path+' --features_db '+features_path +' --bovw_db '+ bovw_path +' --codebook ' +vocab_path +' --idf '+ idf_path + '  --query '+ chosen2_path)
        st.write('Done !')
        time_string = str(time.time() - start_time)
        st.write('Searching time:  '  + time_string[:6] +'s')

        import pandas as pd 

        results = pd.read_csv('./results.csv')
        full_dataset_path = './paris'
        #st.dataframe(results)
        for i, (name,score) in enumerate(zip(results['Path'], results['Cosine_Score'])):
          col1 , col2 = st.columns([2,2])
          dir = name[8:-12]
          path = os.path.join(full_dataset_path,dir, name[2:-1])
          #st.write(path)
          img = Image.open(path)
          
          col1.image(img, width = 300)
          col2.write("Rank "+ str(i+1))
          col2.write(score)
          col2.write(name[2:-1])
    