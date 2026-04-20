"""
Political QA environment.
"""

from .politicalqa_dataloader import create_political_qa_dataset, PoliticalQALoader

__all__ = [
    'create_political_qa_dataset',
    'PoliticalQALoader',
]
