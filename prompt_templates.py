# prompt_templates.py
"""Dynamic prompt system with simplified explanations and unified base template."""
import re
from typing import List

class DynamicPromptTemplates:
    """Dynamic prompt template system (Unified Base + Exclusion Templates)"""

    @staticmethod
    def select_template(excluded_memory=None, dataset_type=None):
        """
        Selects the appropriate base prompt template.
        The actual memory features will be formatted separately and inserted.
        Exclusion logic is now primarily handled by memory_system.get_tagged_features not returning
        features for excluded components. This method provides the overall structure.

        Args:
            excluded_memory: Name of the memory component to conceptually exclude.
                             (Actual exclusion happens in feature generation).
            dataset_type: No longer used to select different base templates.

        Returns:
            Appropriate prompt template string (in English).
        """
        # Since memory feature generation is now simplified and direct,
        # the base template can be mostly unified.
        # If specific wording changes were needed based on exclusion,
        # similar to get_template_without_sequential_memory, those could be kept.
        # For now, let's assume the main unified template is sufficient as the
        # memory_features placeholder will contain appropriately filtered features.

        # Example if you still wanted slightly different phrasing based on exclusion:
        # if excluded_memory == "sequential":
        #     return DynamicPromptTemplates.get_template_without_sequential_memory_info()
        # elif excluded_memory == "working":
        #     return DynamicPromptTemplates.get_template_without_working_memory_info()
        # elif excluded_memory == "long":
        #     return DynamicPromptTemplates.get_template_without_long_memory_info()
        
        return DynamicPromptTemplates.get_unified_base_template_for_llm()

    @staticmethod
    def get_unified_base_template_for_llm():
        """
        Provides the base structure for the LLM prompt.
        The {query} and {memory_features} will be filled by PersonalizedGenerator.
        The specific instructions for generating phrases are now part of the
        prompt_content in PersonalizedGenerator.
        """
        # This template is now very minimal, as most instructions are in PersonalizedGenerator's prompt_content
        return """User Query: {query}

Memory Features (derived from user's interaction history):
{memory_features}
"""
        # The instruction for "Personalized Descriptive Phrases:" and the format
        # will be appended by PersonalizedGenerator.

    # The following specific templates might be less critical if the main instruction
    # in PersonalizedGenerator is robust and memory_features are already filtered.
    # However, keeping them as stubs or slightly varied intros can be an option.
    # For maximum simplicity now, they will also return the unified base.

    @staticmethod
    def get_template_without_sequential_memory_info():
        """Base template when sequential memory is conceptually absent."""
        return DynamicPromptTemplates.get_unified_base_template_for_llm()
        # Or:
        # return """User Query: {query}
#
# Memory Features (Sequential Memory features might be limited or absent):
# {memory_features}
# """

    @staticmethod
    def get_template_without_working_memory_info():
        """Base template when working memory is conceptually absent."""
        return DynamicPromptTemplates.get_unified_base_template_for_llm()

    @staticmethod
    def get_template_without_long_memory_info():
        """Base template when long-term memory is conceptually absent."""
        return DynamicPromptTemplates.get_unified_base_template_for_llm()

    @staticmethod
    def format_memory_features(tagged_feature_strings: List[str]) -> str:
        """
        MODIFIED: Simply joins the list of already formatted tagged feature strings.
        The input strings are expected to be like:
        "[TAG_NAME] Feature Type: keyword1, keyword2"
        """
        if not tagged_feature_strings:
            return "No relevant memory features available." # Or an empty string

        # Each string in tagged_feature_strings is already a complete line.
        # Just join them with newlines.
        return "\n".join(tagged_feature_strings)

