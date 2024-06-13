import os
from typing import List

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile


class HFUtilBase:
    def __init__(self, private_endpoint: str = None):
        self._read_token = None
        self._write_token = None
        self.hf_api = None
        if not private_endpoint:
            self.endpoint = "https://huggingface.co"
        else:
            self.endpoint = private_endpoint
        self.login_hf()

    def login_hf(self) -> None:
        print("Loading HF Tokens")

        # Load HF tokens from .env file
        if not os.environ.get('HF_WRITE_TOKEN') or not os.environ.get('HF_READ_TOKEN'):
            load_dotenv()
            if not os.environ.get('HF_WRITE_TOKEN') or not os.environ.get('HF_READ_TOKEN'):
                raise RuntimeError("HF tokens not found in .env file")

        # Set HF Tokens
        self._read_token = os.environ['HF_READ_TOKEN']
        self._write_token = os.environ['HF_WRITE_TOKEN']

        # Initialize HF API
        if not self.hf_api:
            self.hf_api = HfApi(
                endpoint=self.endpoint,  # change this to a private endpoint if needed
            )

        print("Loaded HF Tokens")

    def check_model_exist(self, repo_id: str) -> bool:
        return self.hf_api.repo_exists(repo_id=repo_id, repo_type="model", token=self._read_token)

    def get_file_list(self, repo_id: str) -> List[RepoFile]:
        return self.hf_api.list_repo_files(repo_id=repo_id, token=self._read_token)

    def get_filepaths_list(self, repo_id: str, filter_by: str = None) -> List[str]:
        if not filter_by:
            return [file.path for file in self.get_file_list(repo_id)]
        elif filter_by == "onnx":
            return [file.path for file in self.get_file_list(repo_id) if ".onnx" in file.path]
        elif filter_by == "tokenizer":
            return [file.path for file in self.get_file_list(repo_id) if "tokenizer" in file.path or ".model" in file.path]
        elif filter_by == "config":
            return [file.path for file in self.get_file_list(repo_id) if "config.json" in file.path]
        else:
            raise NotImplementedError(f"Filter by {filter_by} not supported. Must be one of 'onnx', 'tokenizer, or 'config'")


if __name__ == "__main__":
    base = HFUtilBase()
    print(base.check_model_exist("lemousehunter/Mixtral-8x7B-v0.1-inf2-bs12-1024-f16-tp16"))