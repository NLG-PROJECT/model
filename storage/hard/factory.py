from typing import Optional
from .base import HardStorageInterface
from .filesystem import FilesystemStorage
from .gdrive import GoogleDriveStorage

class HardStorageFactory:
    """Factory for creating hard storage implementations."""
    
    @staticmethod
    def create_storage(storage_type: str, **kwargs) -> Optional[HardStorageInterface]:
        """Create a hard storage implementation based on type.
        
        Args:
            storage_type: Type of storage to create ('filesystem' or 'gdrive')
            **kwargs: Additional arguments for the storage implementation
            
        Returns:
            An instance of HardStorageInterface or None if type is not supported
        """
        if storage_type.lower() == 'filesystem':
            base_path = kwargs.get('base_path', 'storage')
            return FilesystemStorage(base_path=base_path)
        elif storage_type.lower() == 'gdrive':
            credentials_path = kwargs.get('credentials_path')
            root_folder_id = kwargs.get('root_folder_id')
            if not credentials_path:
                raise ValueError("credentials_path is required for Google Drive storage")
            return GoogleDriveStorage(credentials_path, root_folder_id)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}") 