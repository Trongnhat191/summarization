from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model_name = 'google/flan-t5-large'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype = torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = 'Ngày 27/3 , Cơ_quan Cảnh_sát điều_tra Công_an TP. Hưng_Yên , tỉnh Hưng_Yên cho biết , đơn_vị vừa ra quyết_định khởi_tố vụ án , khởi_tố bị_can đối_với đối_tượng Mai_Văn_Thương ( SN 1989 , trú tại đội 11 , thôn An_Chiểu 1 , xã Liên_Phương , TP. Hưng_Yên ) để điều_tra về hành_vi trộm_cắp tài_sản.'

text = text.replace('_', ' ')
print(text)
inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)