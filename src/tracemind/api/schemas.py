from pydantic import AwareDatetime, BaseModel, model_validator

from tracemind.models import Incident


class DiagnosisRequest(BaseModel):
    incident: Incident
    investigation_start: AwareDatetime
    investigation_end: AwareDatetime

    @model_validator(mode="after")
    def validate_window(self) -> "DiagnosisRequest":
        if self.investigation_end <= self.investigation_start:
            raise ValueError("investigation_end must be after investigation_start")
        return self
