from typing import Optional

from pydantic import BaseModel, Field


# -------------------- Models --------------------
class ParseRequest(BaseModel):
    prompt: str = Field(
        ...,
        examples=["Write query here."],
    )


class Experience(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None


class CandidateData(BaseModel):
    name: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    optionalSkills: list[str] = Field(default_factory=list)
    instituteName: list[str] = Field(default_factory=list)
    course: list[str] = Field(default_factory=list)
    experience: Experience = Field(default_factory=Experience)
    phoneNumber: Optional[str] = None
    email: Optional[str] = None


class ParseResponse(BaseModel):
    data: CandidateData
