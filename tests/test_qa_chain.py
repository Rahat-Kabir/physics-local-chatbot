import pytest
from app import get_conversational_chain
from langchain.schema import Document  # Assuming this is the type expected

class TestQAChain:
    def test_qa_chain_response(self, mock_st_session):
        """Test question-answering chain response generation"""
        chain = get_conversational_chain()
        
        # Use a Document object as expected by LangChain
        context = "Newton's second law states that the force acting on an object is equal to the mass of the object times its acceleration."
        input_documents = [Document(page_content=context)]  # Assuming LangChain's Document class is correct

        question = "What is Newton's second law?"
        response = chain({
            "input_documents": input_documents,
            "context": "",
            "question": question
        }, return_only_outputs=True)
        
        assert 'force' in response['output_text'], "Expected key phrase 'force' not found in the response."
