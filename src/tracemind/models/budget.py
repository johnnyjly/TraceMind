"""Admission estimates kept separate from provider-reported token usage."""

from pydantic import BaseModel, Field


class TokenLedger(BaseModel):
    used: int = Field(default=0, ge=0)
    reserved: int = Field(default=0, ge=0)
    unknown: int = Field(default=0, ge=0)

    @property
    def committed(self):
        return self.used + self.reserved + self.unknown

    def recover(self):
        self.unknown += self.reserved
        self.reserved = 0

    def settle(self, reservation, actual):
        self.reserved -= reservation
        if actual is None:
            self.unknown += reservation
        else:
            self.used += actual
