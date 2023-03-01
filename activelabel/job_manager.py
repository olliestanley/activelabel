from activelabel import LabelJob, ModelWrapper
from activelabel.text.jobs import TextClassificationLabelJob
from activelabel.text.models import Word2VecSVCTextClassifier


class JobManager:
    def __init__(self, mode: str, label_type: str):
        self.mode = mode
        self.label_type = label_type

    def get_job(self, interval: int = 50) -> LabelJob:
        model = self.get_model()
        label_job = self.get_label_job(model, interval)
        return label_job

    def get_model(self) -> ModelWrapper:
        # TODO: remove hardcoding
        if self.mode == "text":
            if self.label_type == "class":
                return Word2VecSVCTextClassifier(["+", "-", "="])

        raise ValueError("Unknown mode/label type combination")

    def get_label_job(self, model: ModelWrapper, interval: int) -> LabelJob:
        # TODO: remove hardcoding
        if self.mode == "text":
            if self.label_type == "class":
                return TextClassificationLabelJob(model, interval=interval)

        raise ValueError("Unknown mode/label type combination")
