"""Spacing in displayed prose only; scientific identifiers/data stay unchanged."""
import re

def readable(text):
    text=str(text)
    text=re.sub(r'\b(epoch|Epoch|Figure|Table|df|All|all|full|across|of|at|is|rates|batches|and|minus|descriptive|prespecified|Linux|IoU|Car|person|the|were|from|to|for|including|ISSN)(?=\d)',r'\1 ',text)
    text=re.sub(r'\b(Figure|Table)(?=S\d)',r'\1 ',text)
    text=re.sub(r'(?<=[가-힣])(?=\d)',' ',text)
    return re.sub(r'(?<=\d)(?=(?:cells|cases|endpoints|hours|images|pairs|replications|point)\b)',' ',text)
