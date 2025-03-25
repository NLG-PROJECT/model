import os
import json
from typing import Dict, Any, Optional, List, BinaryIO
import logging
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from io import BytesIO
from .base import HardStorageInterface

logger = logging.getLogger(__name__)

class GoogleDriveStorage(HardStorageInterface):
    """Google Drive implementation of hard storage."""
    
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self, credentials_path: str, root_folder_id: Optional[str] = None):
        """Initialize Google Drive storage.
        
        Args:
            credentials_path: Path to service account credentials file
            root_folder_id: Optional root folder ID for organizing files
        """
        self.credentials_path = credentials_path
        self.root_folder_id = root_folder_id
        self.service = self._get_drive_service()
        self._ensure_root_folder()
    
    def _get_drive_service(self):
        """Get Google Drive service instance."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.SCOPES
            )
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise
    
    def _ensure_root_folder(self):
        """Ensure root folder exists and get its ID."""
        if not self.root_folder_id:
            # Create root folder if it doesn't exist
            folder_metadata = {
                'name': 'DocumentStorage',
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            self.root_folder_id = folder.get('id')
    
    def _create_folder(self, name: str, parent_id: str) -> str:
        """Create a folder and return its ID."""
        folder_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = self.service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        return folder.get('id')
    
    def _get_file_by_name(self, name: str, parent_id: str) -> Optional[Dict[str, Any]]:
        """Get file by name in a specific folder."""
        results = self.service.files().list(
            q=f"name='{name}' and '{parent_id}' in parents",
            fields="files(id, name, mimeType)"
        ).execute()
        items = results.get('files', [])
        return items[0] if items else None
    
    def _get_file_by_doc_id(self, doc_id: str) -> Optional[str]:
        """Find a file by its doc_id in metadata.
        
        Args:
            doc_id: Document ID to search for
            
        Returns:
            Google Drive file ID if found, None otherwise
        """
        try:
            # Search for files with matching doc_id in appProperties
            query = f"appProperties has {{ key='doc_id' and value='{doc_id}' }}"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, appProperties)"
            ).execute()
            
            files = results.get('files', [])
            if not files:
                logger.warning(f"No file found with doc_id {doc_id}")
                return None
            
            # Return the first matching file's ID
            file_id = files[0]['id']
            logger.info(f"Found file with doc_id {doc_id}: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error searching for file with doc_id {doc_id}: {e}")
            return None
    
    def store_document(self, content: bytes, filename: str, metadata: Dict[str, Any]) -> str:
        """Store a document in Google Drive.
        
        Args:
            content: Document content
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        try:
            file_metadata = {
                'name': filename,
                'parents': [self.root_folder_id] if self.root_folder_id else []
            }
            
            # Add custom metadata
            for key, value in metadata.items():
                file_metadata[f'appProperties.{key}'] = str(value)
            
            # Determine content type
            content_type = metadata.get('content_type')
            if not content_type:
                # Try to determine from filename
                ext = Path(filename).suffix.lower()
                content_types = {
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    '.xls': 'application/vnd.ms-excel',
                    '.txt': 'text/plain',
                    '.csv': 'text/csv',
                    '.json': 'application/json',
                    '.html': 'text/html',
                    '.htm': 'text/html',
                    '.xml': 'application/xml',
                    '.zip': 'application/zip',
                    '.rar': 'application/x-rar-compressed',
                    '.7z': 'application/x-7z-compressed',
                    '.tar': 'application/x-tar',
                    '.gz': 'application/gzip'
                }
                content_type = content_types.get(ext, 'application/octet-stream')
            
            logger.info(f"Storing document {filename} with content type: {content_type}")
            
            # Create a BytesIO object from the content
            fh = BytesIO(content)
            media = MediaIoBaseUpload(fh, mimetype=content_type)
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file['id']
            logger.info(f"Successfully stored document {filename} with ID: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store document {filename} in Google Drive: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document from Google Drive.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            # Get file metadata and content in parallel
            file = self.service.files().get(
                fileId=doc_id,
                fields='id, name, mimeType, size, createdTime, modifiedTime, appProperties'
            ).execute()
            
            # Download content
            request = self.service.files().get_media(fileId=doc_id)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            # Extract metadata from appProperties
            metadata = {}
            for key, value in file.get('appProperties', {}).items():
                metadata[key] = value
            
            result = {
                'id': file['id'],
                'content': fh.getvalue(),
                'metadata': {
                    'filename': file['name'],
                    'content_type': file['mimeType'],
                    'size': int(file['size']),
                    'created_at': file['createdTime'],
                    'modified_at': file['modifiedTime'],
                    **metadata
                }
            }
            
            logger.debug(f"Successfully retrieved document {doc_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve document from Google Drive: {e}")
            return None
    
    def store_chunk(self, chunk_id: str, content: str, doc_id: str, metadata: Dict[str, Any] = None) -> str:
        """Store a document chunk.
        
        Args:
            chunk_id: Unique chunk identifier
            content: Chunk content
            doc_id: Parent document ID
            metadata: Additional metadata
            
        Returns:
            Chunk ID
        """
        try:
            # Find document folder
            doc_folder = self._get_file_by_name(doc_id, self.root_folder_id)
            if not doc_folder:
                raise Exception(f"Document folder not found for doc_id: {doc_id}")
            
            # Create chunks folder if it doesn't exist
            chunks_folder = self._get_file_by_name('chunks', doc_folder['id'])
            if not chunks_folder:
                chunks_folder = self._create_folder('chunks', doc_folder['id'])
            
            # Prepare chunk metadata
            chunk_metadata = {
                'name': f"{chunk_id}.txt",
                'parents': [chunks_folder['id']]
            }
            
            # Add custom metadata
            if metadata:
                for key, value in metadata.items():
                    chunk_metadata[f'appProperties.{key}'] = str(value)
            
            # Store chunk
            chunk_file = BytesIO(content.encode('utf-8'))
            file = self.service.files().create(
                body=chunk_metadata,
                media_body=MediaIoBaseUpload(chunk_file, mimetype='text/plain'),
                fields='id'
            ).execute()
            
            logger.info(f"Successfully stored chunk {chunk_id} with ID: {file['id']}")
            return file['id']
            
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_id}: {e}")
            raise
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document chunk."""
        try:
            # Search for chunk file
            results = self.service.files().list(
                q=f"name='{chunk_id}.txt'",
                fields="files(id, name, parents)"
            ).execute()
            items = results.get('files', [])
            if not items:
                return None
            
            chunk_file = items[0]
            content = self.service.files().get_media(
                fileId=chunk_file['id']
            ).execute().decode('utf-8')
            
            # Get doc_id from parent folder structure
            doc_folder = self.service.files().get(
                fileId=chunk_file['parents'][0],
                fields='parents'
            ).execute()
            doc_id = self.service.files().get(
                fileId=doc_folder['parents'][0],
                fields='name'
            ).execute()['name']
            
            return {
                "content": content,
                "doc_id": doc_id
            }
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunk IDs associated with a document."""
        try:
            # Find document folder
            doc_folder = self._get_file_by_name(doc_id, self.root_folder_id)
            if not doc_folder:
                return []
            
            # Find chunks folder
            chunks_folder = self._get_file_by_name('chunks', doc_folder['id'])
            if not chunks_folder:
                return []
            
            # List all chunk files
            results = self.service.files().list(
                q=f"'{chunks_folder['id']}' in parents",
                fields="files(name)"
            ).execute()
            
            return [f['name'].replace('.txt', '') for f in results.get('files', [])]
        except Exception as e:
            logger.error(f"Error getting chunks for document {doc_id}: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from Google Drive.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            self.service.files().delete(fileId=doc_id).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to delete document from Google Drive: {e}")
            return False
    
    def store_file(self, file_id: str, file_content: BinaryIO, metadata: Dict[str, Any]) -> bool:
        """Store a file (e.g., PDF, DOCX)."""
        try:
            # Create files folder if it doesn't exist
            files_folder = self._get_file_by_name('files', self.root_folder_id)
            if not files_folder:
                files_folder = self._create_folder('files', self.root_folder_id)
            
            # Determine content type from metadata or file extension
            content_type = metadata.get('content_type')
            if not content_type and hasattr(file_content, 'name'):
                ext = Path(file_content.name).suffix.lower()
                content_types = {
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.doc': 'application/msword',
                    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    '.xls': 'application/vnd.ms-excel',
                    '.txt': 'text/plain',
                    '.csv': 'text/csv',
                    '.json': 'application/json',
                    '.html': 'text/html',
                    '.htm': 'text/html',
                    '.xml': 'application/xml',
                    '.zip': 'application/zip',
                    '.rar': 'application/x-rar-compressed',
                    '.7z': 'application/x-7z-compressed',
                    '.tar': 'application/x-tar',
                    '.gz': 'application/gzip'
                }
                content_type = content_types.get(ext, 'application/octet-stream')
            
            logger.info(f"Storing file {file_id} with content type: {content_type}")
            
            # Create a BytesIO object from the content if it's not already a file-like object
            if isinstance(file_content, bytes):
                file_content = BytesIO(file_content)
            
            # Store file
            file_metadata = {
                'name': file_id,
                'parents': [files_folder['id']]
            }
            self.service.files().create(
                body=file_metadata,
                media_body=MediaIoBaseUpload(file_content, mimetype=content_type),
                fields='id'
            ).execute()
            
            # Store metadata
            metadata_file = BytesIO(json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8'))
            metadata_file_metadata = {
                'name': f"{file_id}.json",
                'parents': [files_folder['id']]
            }
            self.service.files().create(
                body=metadata_file_metadata,
                media_body=MediaIoBaseUpload(metadata_file, mimetype='application/json'),
                fields='id'
            ).execute()
            
            logger.info(f"Successfully stored file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing file {file_id}: {e}")
            return False
    
    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a file and its metadata."""
        try:
            # Find files folder
            files_folder = self._get_file_by_name('files', self.root_folder_id)
            if not files_folder:
                return None
            
            # Get file
            file = self._get_file_by_name(file_id, files_folder['id'])
            if not file:
                return None
            
            # Get file content
            content = self.service.files().get_media(
                fileId=file['id']
            ).execute()
            
            # Get metadata
            metadata_file = self._get_file_by_name(f"{file_id}.json", files_folder['id'])
            if not metadata_file:
                return None
            
            metadata = json.loads(self.service.files().get_media(
                fileId=metadata_file['id']
            ).execute().decode('utf-8'))
            
            return {
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving file {file_id}: {e}")
            return None
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the root folder.
        
        Returns:
            List of document information
        """
        try:
            query = f"'{self.root_folder_id}' in parents" if self.root_folder_id else None
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, createdTime, modifiedTime, appProperties)"
            ).execute()
            
            documents = []
            for file in results.get('files', []):
                metadata = {}
                for key, value in file.get('appProperties', {}).items():
                    metadata[key] = value
                
                documents.append({
                    'id': file['id'],
                    'metadata': {
                        'filename': file['name'],
                        'content_type': file['mimeType'],
                        'size': int(file['size']),
                        'created_at': file['createdTime'],
                        'modified_at': file['modifiedTime'],
                        **metadata
                    }
                })
            
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents from Google Drive: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if the storage is healthy.
        
        Returns:
            True if healthy
        """
        try:
            # Try to list files to verify connection
            self.service.files().list(pageSize=1).execute()
            return True
        except Exception as e:
            logger.error(f"Google Drive health check failed: {e}")
            return False 