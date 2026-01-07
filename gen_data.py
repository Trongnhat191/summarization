import os
import dotenv
from datasets import load_dataset
import json
import concurrent.futures

dotenv.load_dotenv()

FPT_KEY = os.getenv("FPT_KEY")
BASE_URL = 'https://mkp-api.fptcloud.com'
MODEL_NAME = 'gpt-oss-120b'

def client_init(api_key, base_url):
    """Khởi tạo client để gọi API từ FPT Marketplace"""
    from openai import OpenAI
    client = OpenAI(api_key = api_key, 
                    base_url = base_url)
    return client

def base_prompt(article_content):
    prompt= f"""Bạn là một biên tập viên báo chí chuyên nghiệp. Nhiệm vụ của bạn là tóm tắt bài báo được cung cấp. 
Chú ý yêu cầu:
1. Chỉ sử dụng thông tin có trong bài báo.
2. Không thêm ý kiến cá nhân hoặc thông tin bên ngoài.
3. Tóm tắt dưới dạng một đoạn văn từ 3-5 câu.
4. Ưu tiên giữ lại các con số, tên riêng và sự kiện chính.
Nội dung bài báo như sau:
{article_content}."""

    return prompt

def parse_product_apikey(questions, question_idx,  model_name, client,  
                        temperature = 0.05 , top_p = 0.95, max_completion_tokens = None,
                        max_attemps = 5
                        ):
    
    prompted_question = base_prompt(questions[question_idx])
    for i in range(max_attemps):
            # Xóa print ở đây để tránh log bị lộn xộn khi chạy song song
            # print(f'running question {question_idx + 1}')
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompted_question,
                        },
                    ],
                    temperature=temperature, 
                    top_p=top_p,        
                    frequency_penalty=0.0,  
                    presence_penalty=0.0,  
                    stream=False,
                    max_completion_tokens=max_completion_tokens,
                )
                answer = response.to_dict()['choices'][0]['message']['content'] 
                return answer
            except Exception as e:
                print(f"Error at index {question_idx}: {e}")
                continue
    return None # Trả về None nếu thất bại sau max_attemps

def process_single_item(idx, document, model_name, client):
    """Hàm wrapper để xử lý một item trong luồng"""
    print(f'Processing index {idx}...')
    summary = parse_product_apikey([document], 0, model_name, client, # Truyền list chứa 1 document
                                temperature=0.1, top_p=0.9,
                                max_attemps=3)
    if summary:
        return {
            'Document': document,
            'Summary': summary,
            'Original_Idx': idx # Giữ lại index gốc để sort lại nếu cần
        }
    return None

def calculate_average_length(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_doc_len = sum(len(item['Document'].split()) for item in data)
    total_summary_len = sum(len(item['Summary'].split()) for item in data)
    
    avg_doc_len = total_doc_len / len(data)
    avg_summary_len = total_summary_len / len(data)
    
    print(f'Average document length: {avg_doc_len}')
    print(f'Average summary length: {avg_summary_len}')
    # max, min length
    max_doc_len = max(len(item['Document'].split()) for item in data)
    min_doc_len = min(len(item['Document'].split()) for item in data)
    max_summary_len = max(len(item['Summary'].split()) for item in data)
    min_summary_len = min(len(item['Summary'].split()) for item in data)
    print(f'Max document length: {max_doc_len}, Min document length: {min_doc_len}')
    print(f'Max summary length: {max_summary_len}, Min summary length: {min_summary_len}')

    #count documents with length < 800
    count = sum(1 for item in data if len(item['Document'].split()) < 800)
    print(f'Number of documents with length < 800: {count}')
    
    # create new json file with documents with length < 800
    new_data = [item for item in data if len(item['Document'].split()) < 800]
    with open('filtered_' + json_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    json_file = 'new_data.json'
    calculate_average_length(json_file)
    d
    client = client_init(FPT_KEY, BASE_URL)
    dataset = load_dataset('OpenHust/vietnamese-summarization')
    train_data = dataset['train']

    # select subset with document length < 700 words and > 600 words
    train_data = train_data.filter(lambda x: len(x['Document'].split()) < 750 and len(x['Document'].split()) > 650)
    train_data = train_data.select(range(100))
    # calculate averaghe length of document and summary
    total_doc_len = sum(len(doc.split()) for doc in train_data['Document'])
    total_summary_len = sum(len(summary.split()) for summary in train_data['Summary'])
    print(f'Average document length: {total_doc_len/len(train_data)}')
    print(f'Average summary length: {total_summary_len/len(train_data)}')

    documents = train_data['Document']

    summaries = train_data['Summary'] # Lấy summary gốc để tính độ dài tham khảo

    new_data = []
    original_gt_len = []
    new_gt_len = []
    
    MAX_WORKERS = 1 
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Tạo danh sách các task
        futures = {executor.submit(process_single_item, idx, doc, MODEL_NAME, client): idx for idx, doc in enumerate(documents)}
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                new_data.append(result)
                # Tính toán độ dài cho thống kê
                idx = result['Original_Idx']
                original_gt_len.append(len(summaries[idx].split()))
                new_gt_len.append(len(result['Summary'].split()))

    # Sắp xếp lại data theo index gốc nếu cần thứ tự đúng
    new_data.sort(key=lambda x: x['Original_Idx'])
    # Xóa key tạm Original_Idx trước khi lưu
    for item in new_data:
        del item['Original_Idx']

    with open('new_data_seq2seq.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    if original_gt_len:
        print(f'Original GT length: {sum(original_gt_len)/len(original_gt_len)}')
        print(f'New GT length: {sum(new_gt_len)/len(new_gt_len)}')