from enum import Enum

class ResponseSignal(Enum):

    FILE_VALIDATION_SUCCESS = "file_validate_successfully"
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
    PROCESSING_SUCCESS = "processing_success"
    PROCESSING_FAILED = "processing_failed"
    NO_FILES_ERROR = "not_found_files"
    FILE_ID_ERROR = "no_file_found_with_this_id"
    PROJECT_NOT_FOUND = "project_not_found"
    INSERT_INTO_VECTOR_DB_ERROR = "insert_into_vector_db_error"
    INSERT_INTO_VECTOR_DB_SUCCESS = "insert_into_vector_db_success"
    VECTOR_DB_COLLECTION_RETRIEVED = "vector_db_collection_retrieved"
    VECTOR_DB_COLLECTION_NOT_FOUND = "vector_db_collection_not_found"
    VECTOR_DB_SEARCH_ERROR = "vector_db_search_error"
    VECTOR_DB_SEARCH_SUCCESS = "vector_db_search_success"
    RAG_ANSWER_ERROR = "rag_answer_error"
    RAG_ANSWER_SUCCESS = "rag_answer_success"
    CHAT_ANSWER_ERROR = "chat_answer_error"
    CHAT_ANSWER_SUCCESS = "chat_answer_success"