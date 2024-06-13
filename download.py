
from HFUtils.HFUtilDownload import HFUtilDownload

import requests,os
from tqdm import tqdm
import platform


def getProxy():
    if platform.system() == "Linux":
            return "192.168.0.77:18808"
    else:
        return "127.0.0.1:10809"

os.environ['HTTP_PROXY'] = getProxy()
os.environ['HTTPS_PROXY'] = getProxy()


def download_file(url, local_filename=None):
    """
    下载文件并显示进度条
    
    :param url: 文件的URL
    :param local_filename: 本地保存的文件名，如果不提供则从URL中提取
    """
    if local_filename is None:
        local_filename = url.split('/')[-1]
    # 确保保存目录存在
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # 确保请求成功
        total_size = int(r.headers.get('content-length', 0))  # 获取文件总大小
        with open(local_filename, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk:  # 过滤掉keep-alive新行
                    f.write(chunk)
                    bar.update(len(chunk))

downLoadUtil = HFUtilDownload()


# url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
# localFilePath = "./whisper/tiny.pt"
# if not os.path.exists(localFilePath):
#     download_file(url, localFilePath)

# url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
# localFilePath = "./face-parse-bisent/resnet18-5c106cde.pth"
# if not os.path.exists(localFilePath):
#     download_file(url, localFilePath)



# repo_id="TMElyralab/MusePose"
# folder_name="MusePose"
local_dir="./models/musetalk"
# downLoadUtil.download_folder_from_repo(repo_id=repo_id, 
#                                          folder_name=folder_name, 
#                                          local_dir=local_dir)

repo_id="hengtuibabai/MuseTalk.models"
local_dir="./models/"
downLoadUtil.download_folder_from_repo(repo_id=repo_id, 
                                         local_dir=local_dir)

import zipfile
zip_file_path="./models/models.zip"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # 解压所有内容到指定目录
    zip_ref.extractall(local_dir)
    print(f"文件已成功解压到 {local_dir}")


# repo_id="yzd-v/DWPose"
# local_dir="./models/dwpose"
# downLoadUtil.download_folder_from_repo(repo_id=repo_id, 
#                                          local_dir=local_dir)

# repo_id="yzd-v/DWPose"
# dw_dir = f"{local_dir}/dwpose/"
# downLoadUtil.download_folder_from_repo(repo_id=repo_id, 
#                                          local_dir=dw_dir)


# repo_id="lambdalabs/sd-image-variations-diffusers"
# downLoadUtil.download_folder_from_repo(repo_id=repo_id, 
#                                          local_dir=local_dir)



