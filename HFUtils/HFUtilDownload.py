import os

from huggingface_hub import snapshot_download
from tqdm import tqdm

from .HFUtilBase import HFUtilBase
import platform



class HFUtilDownload(HFUtilBase):

    def download_folder_from_repo(self, repo_id: str, local_dir: str,folder_name: str = None):
        fileList = self.get_file_list(repo_id)
        if folder_name:
            files_list = [file for file in list(self.get_file_list(repo_id)) if f"{folder_name}/" in file]
        else:
            files_list = [file for file in list(self.get_file_list(repo_id))]

        for file in tqdm(files_list):
            self.hf_api.hf_hub_download(repo_id=repo_id, filename=file, token=self._read_token,
                                        local_dir=local_dir, local_dir_use_symlinks=False)

    def _download_file_from_repo(self, repo_id: str, local_dir: str, filename: str):
        files_list = os.listdir(local_dir)
        if filename not in files_list:
            print("downloading", filename)
        self.hf_api.hf_hub_download(repo_id=repo_id, filename=filename, token=self._read_token, local_dir=local_dir,
                                    local_dir_use_symlinks=False)

    def download_tokenizer_from_repo(self, repo_id: str, local_dir: str):
        print(local_dir)
        to_download = [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json"
        ]

        for file in to_download:
            self._download_file_from_repo(repo_id=repo_id, local_dir=local_dir, filename=file)

        files_list = os.listdir(local_dir)

        tokenizer_exist = False
        for filename in files_list:
            if ".model" in filename:
                tokenizer_exist = True

        if not tokenizer_exist:
            print("downloading tokenizer model")
            snapshot_download(repo_id=repo_id, allow_patterns="*.model", local_dir=local_dir,
                              local_dir_use_symlinks=False)

    def download_config_from_repo(self, repo_id: str, local_dir: str):
        file_list = os.listdir(local_dir)
        if "config.json" not in file_list:
            print("downloading config.json")
            self.hf_api.hf_hub_download(repo_id=repo_id, filename="config.json", token=self._read_token,
                                        local_dir=local_dir, local_dir_use_symlinks=False)


if __name__ == "__main__":
    hf_util = HFUtilDownload()
    hf_util.download_tokenizer_from_repo("BAAI/bge-m3", "/Users/mouse/Documents/GitHub/HFUtils/Downloads/onnx")
