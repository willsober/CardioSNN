import os
import requests
from tqdm import tqdm

def download_file(url, local_file):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        
        with open(local_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✓ 已下载: {local_file}")  # 新增：打印文件保存路径
        return True
    except Exception as e:
        print(f"\n下载 {os.path.basename(local_file)} 时出错: {str(e)}")
        if os.path.exists(local_file):
            os.remove(local_file)
        return False

def download_mitbih_data():
    
    os.makedirs('data/raw', exist_ok=True)
    
   
    base_url = "https://physionet.org/files/mitdb/1.0.0"
    
   
    record_numbers = range(100, 125)
    
    extensions = ['.dat', '.hea', '.atr']
    
  
    
    total_files = len(record_numbers) * len(extensions)
    success_count = 0
    
    with tqdm(total=total_files, desc="time") as pbar:
        for record in record_numbers:
            for ext in extensions:
                filename = f"{record}{ext}"
                url = f"{base_url}/{filename}"
                local_file = os.path.join('data/raw', filename)
                
                # 如果文件已存在，跳过下载
                if os.path.exists(local_file):
                    success_count += 1
                    pbar.update(1)
                    continue
                
                # 下载文件
                if download_file(url, local_file):
                    success_count += 1
                pbar.update(1)
    
  
    missing_files = []
    for record in record_numbers:
        for ext in extensions:
            filename = f"{record}{ext}"
            local_file = os.path.join('data/raw', filename)
            if not os.path.exists(local_file):
                missing_files.append(filename)
    
    if missing_files:
        print("\:")
        for filename in missing_files:
            print(f"- {filename}")
    else:
        print("\!")
    
    

if __name__ == '__main__':
    download_mitbih_data()