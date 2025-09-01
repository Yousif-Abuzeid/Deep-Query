from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
from .ProjectController import ProjectController
import re
import os
class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576  # 1 MB

    def validate_uploaded_file(self, file: UploadFile):
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value
        return True, ResponseSignal.FILE_VALIDATION_SUCCESS.value

    def generate_unique_filename(self, orig_file_name: str, project_id: str) -> str:
        random_key = self.generate_random_string(16)
        project_controller = ProjectController()
        project_path = project_controller.get_project_path(project_id)

        cleaned_file_name = self.get_clean_file_name(orig_file_name)

        new_file_path = os.path.join(
            project_path,
            random_key + "_" + cleaned_file_name)

        while os.path.exists(new_file_path):
            random_key = self.generate_random_string(16)
            new_file_path = os.path.join(
                project_path, random_key + "_" + cleaned_file_name)

        return new_file_path

    def get_clean_file_name(self, orig_file_name: str) -> str:
        cleaned_file_name = re.sub(r'[^\w.]','',orig_file_name.strip())
        return cleaned_file_name