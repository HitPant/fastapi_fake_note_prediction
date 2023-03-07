from pydantic import BaseModel

class Noteparam(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float
    