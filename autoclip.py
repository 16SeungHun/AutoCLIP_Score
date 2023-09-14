from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def clipscore(text, image):
    
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    return logits_per_image.tolist()


#def get_tsv_eng(dir:str):
#    df = pd.read_csv(dir, sep='\t', encoding='utf-8')
#    df = df[df.story_id<=2]
#    return df['en'].tolist()

if __name__=="__main__":
    
    #f_root = "/home/shlee/IJCNN/clipauto/"
    
    model_name = 'sd-illust-model'
    model_path = '/home/shlee/skt_finetune/' + model_name
#    model_path = "/home/shlee/textual_inversion/textual_inversion_fairy_tale"
    
    prompts = ["anything..."] #get_tsv_eng(f_root+"fairy_tale_grimm.tsv")
   
    #folder_name
    names = "ft_du_t_photo", "n_du_t_photo"
    #csv file name
    namess = 'clipscores_dt_t.csv'
      
    pipes = (StableDiffusionPipeline.from_pretrained(model_path, torch_dtype= torch.float16, revision="fp32").to("cuda"), 
             StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda"))
    
    scores = [[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], [],[], [], [], [], [], [], [], [], [], []]
    #for num in range(1,11):
    for i, prompt in enumerate(tqdm(prompts)):
      for name, pipe, score in zip(names, pipes, scores):
        for j, sc in enumerate(score):
                  prompt = prompt[:100]
                  image = pipe(prompt, num_inference_steps=100, guidance_scale=8).images[0]
                  sc.append(clp:=clipscore(prompt, image)[0][0])
                  print("\n", f"{name}{i}-{j}:{clp}")
                  image.save(f"./datas/{name}/{i:06d}_{j}.jpg")
    
    n = ["ft0", "ft1", "ft2", "ft3", "ft4","ft5","ft6","ft7","ft8","ft9","ft10","ft11", "ft12", "ft13", "ft14", "ft15","ft16","ft17","ft18","ft19","ft20","ft21","ft22", "ft23", "ft24", "ft25", "ft26","ft27","ft28","ft29","ft30","ft31","ft32","ft33", "ft34", "ft35", "ft36", "ft37","ft38","ft39","ft40", "ft41", "ft42", "ft43", "ft44","ft45","ft46","ft47","ft48","ft49","ft50","ft51", "ft52", "ft53", "ft54", "ft55","ft56","ft57","ft58","ft59","ft60","ft61","ft62", "ft63", "ft64", "ft65", "ft66","ft67","ft68","ft69","ft70","ft71","ft72","ft73", "ft74", "ft75", "ft76", "ft77","ft78","ft79","ft80", "ft81", "ft82", "ft83", "ft84","ft85","ft86","ft87","ft88","ft89","ft90", "ft91", "ft92", "ft93", "ft94","ft95","ft96","ft97","ft98","ft99"], ["r0", "r1", "r2", "r3", "r4","r5","r6","r7","r8","r9","r10","r11", "r12", "r13", "r14", "r15","r16","r17","r18","r19","r20","r21","r22", "r23", "r24", "r25", "r26","r27","r28","r29","r30","r31","r32","r33", "r34", "r35", "r36", "r37","r38","r39","r40", "r41", "r42", "r43", "r44","r45","r46","r47","r48","r49","r50","r51", "r52", "r53", "r54", "r55","r56","r57","r58","r59","r60","r61","r62", "r63", "r64", "r65", "r66","r67","r68","r69","r70","r71","r72","r73", "r74", "r75", "r76", "r77","r78","r79","r80", "r81", "r82", "r83", "r84","r85","r86","r87","r88","r89","r90", "r91", "r92", "r93", "r94","r95","r96","r97","r98","r99"]
    data = {j:dd for i, d in zip(n, scores) for j, dd in zip(i, d)}
    
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f'./{namess}', sep=',')